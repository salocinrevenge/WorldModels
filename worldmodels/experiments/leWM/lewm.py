import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
import stable_pretraining as spt

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x

class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class LeWM_backbone(nn.Module):
    def __init__(
        self,
        encoder=None,
        predictor=None,
        action_encoder=None,
        projector=None,
        pred_proj=None,
        history_size=3,
        num_preds=1,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.history_size = history_size
        self.num_preds = num_preds


        # Standart values for the LeWM model components if not provided
        if encoder == None:
            self.encoder = spt.backbone.utils.vit_hf("small", patch_size=16, image_size=224, pretrained=False)
        if action_encoder == None:
            self.action_encoder = Embedder(input_dim=2, smoothed_dim=16, emb_dim=256)
        if predictor == None:
            self.predictor = ARPredictor(num_frames=3, depth=4, heads=8, mlp_dim=1024, input_dim=256, hidden_dim=256)
        if projector == None:
            self.projector = MLP(input_dim=384, hidden_dim=512, output_dim=256, norm_fn=torch.nn.BatchNorm1d)
        if pred_proj == None:
            self.pred_proj = MLP(input_dim=256, hidden_dim=512, output_dim=256, norm_fn=torch.nn.BatchNorm1d)

    def encode(self, pixels, action=None):
        """Codifica imagens e ações em embeddings latentes."""
        b, t = pixels.shape[:2]
        pixels_flat = rearrange(pixels, "b t ... -> (b t) ...")
        
        output = self.encoder(pixels_flat)
        # Extrai o token CLS se for um modelo Vision Transformer (ViT)
        if hasattr(output, "last_hidden_state"):
            pixels_emb = output.last_hidden_state[:, 0]
        else:
            pixels_emb = output

        emb = self.projector(pixels_emb)
        emb = rearrange(emb, "(b t) d -> b t d", b=b, t=t)

        act_emb = None
        if action is not None:
            act_emb = self.action_encoder(action)

        return emb, act_emb

    def forward(self, x, action):
        """
        x: Imagens (B, T, C, H, W)
        action: Sequência de Ações (B, T, action_dim)
        """
        emb, act_emb = self.encode(x, action)

        ctx_len = self.history_size
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]

        # Previsão autorregressiva do estado futuro no espaço latente
        pred_raw = self.predictor(ctx_emb, ctx_act)
        pred_emb = self.pred_proj(rearrange(pred_raw, "b t d -> (b t) d"))
        pred_emb = rearrange(pred_emb, "(b t) d -> b t d", b=x.size(0))

        # Retorna dicionário contendo tanto a predição quanto o embedding real
        return {
            "pred_emb": pred_emb,
            "emb": emb,
            "num_preds": self.num_preds,
            "history_size": self.history_size,
        }

class LeWM_decoder(nn.Module):
    """Decoder convolucional opcional para reconstruir pixels a partir do espaço latente."""
    def __init__(self, emb_dim=256, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 512 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1), # 224x224
            nn.Sigmoid(),
        )

    def forward(self, backbone_output):
        if isinstance(backbone_output, dict):
            emb = backbone_output["pred_emb"]
        else:
            emb = backbone_output

        b, t, d = emb.shape
        emb_flat = emb.reshape(b * t, d)

        x = self.fc(emb_flat)
        x = x.view(b * t, 512, 7, 7)
        rec = self.deconv(x)
        return rearrange(rec, "(b t) c h w -> b t c h w", b=b, t=t)

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    ############################### ATENÇÃO ################################
    # Obtido de module do repositório LeWM, checar lá se algo quebrar por conta dessa função.

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

class LeWM_loss(nn.Module):
    def __init__(self, sigreg_kwargs=None, sigreg_weight=1.0):
        super().__init__()
        sigreg_kwargs = sigreg_kwargs or {"knots": 17, "num_proj": 1024}
        self.sigreg = SIGReg(**sigreg_kwargs)
        self.lambd = sigreg_weight

    def forward(self, output, target):
        # Caso o decoder de pixels esteja ativado
        if isinstance(output, torch.Tensor) and output.dim() == 5:
            return F.mse_loss(output, target)

        # Treinamento Latente Padrão do LeWM
        pred_emb = output["pred_emb"]
        emb = output["emb"]
        n_preds = output.get("num_preds", 1)

        tgt_emb = emb[:, n_preds:]  # Alvo latente real
        
        # 1. Erro de predição MSE
        pred_loss = F.mse_loss(pred_emb, tgt_emb)

        # 2. Regularização de Espaço Latente (SIGReg)
        sigreg_loss = self.sigreg(emb.transpose(0, 1))

        return pred_loss + self.lambd * sigreg_loss

class LeWMDataModule(L.LightningDataModule):
    def __init__(self, dataset, batch_size=16, train_split=0.9, seed=42, num_workers=2):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_split = train_split
        self.seed = seed
        self.num_workers = num_workers

    def setup(self, stage=None):
        gen = torch.Generator().manual_seed(self.seed)
        self.train_set, self.val_set = spt.data.random_split(
            self.dataset, lengths=[self.train_split, 1.0 - self.train_split], generator=gen
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
        )

    def _collate_fn(self, batch):
        """Garante a formatação da tupla (input, action, target) exigida pelo main.py."""
        if isinstance(batch, dict):
            pixels = batch["pixels"]
            actions = torch.nan_to_num(batch["action"], 0.0)
            return pixels, actions, pixels

        pixels = torch.stack([item["pixels"] for item in batch])
        actions = torch.stack([torch.nan_to_num(item["action"], 0.0) for item in batch])
        return pixels, actions, pixels
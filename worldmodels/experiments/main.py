import multiprocessing

multiprocessing.set_start_method("fork", force=True)

import torch
from torch.utils.data import Subset
import lightning as L
import stable_pretraining as spt
import stable_worldmodel as swm
from leWM.lewm import LeWM_backbone, LeWM_loss, LeWMDataModule
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
import os
import time
from pathlib import Path
import numpy as np
import hydra
from omegaconf import OmegaConf, open_dict
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class WorldModel(L.LightningModule):
    def __init__(self, backbone, decoder = None, load_weights=None, loss = None, learning_rate=1e-4, use_decoder=False, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.load_weights = load_weights
        self.loss = loss
        self.learning_rate = learning_rate
        self.using_decoder = use_decoder
        self.freeze_backbone = freeze_backbone

        if load_weights:
            self.load_state_dict(torch.load(load_weights))

    # Para ser usado como modelo em CEMSolver:
    def _get_expected_action_channels(self) -> int:
        """Retorna o número de canais de ação esperados pelo Embedder (padrão: 2 para PushT)."""
        try:
            if hasattr(self.backbone, "action_encoder"):
                ae = self.backbone.action_encoder
                if hasattr(ae, "patch_embed") and hasattr(ae.patch_embed, "in_channels"):
                    return ae.patch_embed.in_channels
        except Exception:
            pass
        return 2

    def _format_action(self, action: torch.Tensor) -> torch.Tensor:
        if action is None or not torch.is_tensor(action):
            return None

        # Get target in_channels expected by the Conv1d patch embedder (typically 2)
        target_dim = self._get_expected_action_channels()

        # Flatten CEM batch and sample dimensions if 4D: (B, S, T, A) -> (B*S, T, A)
        if action.ndim == 4:
            b, s, t, a = action.shape
            action = action.reshape(b * s, t, a)

        if action.ndim == 3:
            # Check if already in (Batch, Channels, Length) format: (300, 2, T)
            if action.shape[1] == target_dim:
                return action
            
            # If in (Batch, Length, Channels) format: (300, T, 2) -> transpose to (300, 2, T)
            elif action.shape[2] == target_dim:
                return action.transpose(1, 2)
            
            # If action dimension > target_dim (e.g., shape is 300, 10, 5), slice & transpose to (300, 2, 10)
            elif action.shape[2] > target_dim:
                return action[:, :, :target_dim].transpose(1, 2)
            elif action.shape[1] > target_dim:
                return action[:, :target_dim, :]

        elif action.ndim == 2:
            # (Batch, Action_Dim) -> (Batch, Action_Dim, 1)
            if action.shape[1] == target_dim:
                return action.unsqueeze(-1)
            else:
                return action[:, :target_dim].unsqueeze(-1)

        return action

    def encode(self, info: dict):
        # 1. Localiza o tensor de imagem no dicionário
        pixels = None
        for key in ["pixels", "goal", "goal_pixels"]:
            if key in info and torch.is_tensor(info[key]):
                pixels = info[key]
                break

        if pixels is None:
            raise KeyError(
                f"Nenhum tensor de imagem ('pixels', 'goal', 'goal_pixels') foi encontrado. Chaves disponíveis: {list(info.keys())}"
            )

        orig_shape = pixels.shape
        orig_ndim = pixels.ndim

        # 2. Normaliza as dimensões de pixels para o formato 5D esperado pelo LeWM: (B_eff, T, C, H, W)
        if orig_ndim == 3:  # (C, H, W)
            pixels_5d = pixels.unsqueeze(0).unsqueeze(0)
            leading_shape = (1, 1)
        elif orig_ndim == 4:  # (B, C, H, W)
            pixels_5d = pixels.unsqueeze(1)
            leading_shape = (orig_shape[0], 1)
        elif orig_ndim == 5:  # (B, T, C, H, W)
            pixels_5d = pixels
            leading_shape = (orig_shape[0], orig_shape[1])
        else:  # 6D+, ex: (B, S, T, C, H, W)
            c, h, w = orig_shape[-3:]
            t_dim = orig_shape[-4]
            b_eff = 1
            for d_size in orig_shape[:-4]:
                b_eff *= d_size
            pixels_5d = pixels.reshape(b_eff, t_dim, c, h, w)
            leading_shape = orig_shape[:-3]

        # 3. Ajusta o tensor de ação se presente
        action = info.get("action", None)
        formatted_action = self._format_action(action) if action is not None else None

        # 4. Passa os tensores formatados para o backbone
        emb, act_emb = self.backbone.encode(pixels_5d, action=formatted_action)

        # 5. Restaura o formato original caso venha do CEM (ex: B, S, T, D)
        if orig_ndim > 5:
            d_dim = emb.shape[-1]
            emb = emb.reshape(*leading_shape, d_dim)

        return {"emb": emb, "act_emb": act_emb}

    def rollout(self, info_dict: dict, action_candidates: torch.Tensor, history_size: int = 3):
        if hasattr(self.backbone, "rollout"):
            return self.backbone.rollout(info_dict, action_candidates, history_size=history_size)

        if "emb" not in info_dict:
            encoded = self.encode(info_dict)
            info_dict["emb"] = encoded["emb"]

        emb = info_dict["emb"]

        # 1. Determina dinamicamente o histórico de contexto disponível
        if emb.ndim == 4:  # (B, S, T_ctx, D)
            b, s, t_ctx, d = emb.shape
            curr_history = min(t_ctx, history_size)
            ctx_emb = emb[:, :, :curr_history].reshape(b * s, curr_history, d)
            b_act_eff = b * s
        elif emb.ndim == 3:  # (B, T_ctx, D)
            b, t_ctx, d = emb.shape
            curr_history = min(t_ctx, history_size)
            ctx_emb = emb[:, :curr_history]
            b_act_eff = b
        else:
            raise ValueError(f"Dimensão inesperada para o embedding: {emb.ndim}")

        # 2. Formata ações candidatas do CEM para (B_eff, T_act, Action_Dim)
        formatted_actions = self._format_action(action_candidates)

        if action_candidates.ndim == 4:
            b_act, s_act = action_candidates.shape[:2]
            if ctx_emb.shape[0] != b_act * s_act:
                ctx_emb = ctx_emb.repeat_interleave(s_act, dim=0)
        else:
            b_act, s_act = b_act_eff, 1

        # 3. Predição no espaço latente usando a sequência completa de ações do planejamento
        ctx_act_emb = self.backbone.action_encoder(formatted_actions)
        pred_raw = self.backbone.predictor(ctx_emb, ctx_act_emb)

        # 4. Projeta e restaura o formato original das predições
        t_pred = pred_raw.shape[1]
        pred_emb = self.backbone.pred_proj(rearrange(pred_raw, "b t d -> (b t) d"))

        if action_candidates.ndim == 4:
            pred_emb = rearrange(pred_emb, "(b_s t) d -> b_s t d", t=t_pred)
            pred_emb = pred_emb.reshape(b_act, s_act, t_pred, -1)
        else:
            pred_emb = rearrange(pred_emb, "(b t) d -> b t d", t=t_pred)

        info_dict["predicted_emb"] = pred_emb
        return info_dict

    def criterion(self, info_dict: dict):
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]

        while goal_emb.ndim < pred_emb.ndim:
            goal_emb = goal_emb.unsqueeze(1)
        goal_emb = goal_emb.expand_as(pred_emb)

        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        device = next(self.parameters()).device # Descobre o device

        for k in list(info_dict.keys()): # Manda td pra o device
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] if (torch.is_tensor(v) and v.ndim > 1) else v for k, v in info_dict.items()} # Obtem o primeiro frame, sendo ele o objetivo

        if "pixels" not in goal: # Padroniza a informação: goal["pixels"] contém os pixels do objetivo
            if "goal" in goal:
                goal["pixels"] = goal.pop("goal")
            elif "goal_pixels" in goal:
                goal["pixels"] = goal.pop("goal_pixels")

        goal.pop("action", None) # Remove a coluna de ações, se existir. Objetivo não tem ação.

        goal_out = self.encode(goal) # Codifica o objetivo para obter o embedding do objetivo
        info_dict["goal_emb"] = goal_out["emb"] # Adiciona o embedding do objetivo ao dicionário de informações

        info_dict = self.rollout(info_dict, action_candidates) # Executa a simulação do modelo de mundo para obter as predições futuras para as ações candidatas
        return self.criterion(info_dict) # Compara o estado latente previsto com o estado latente do objetivo e retorna o custo associado

    # Fim dos métodos para CEMSolver

    def format_action(action: torch.Tensor, target_channels: int = 2) -> torch.Tensor:
        if action is None:
            return None

        # Handle 2D: (B, A) -> (B, A, 1)
        if action.ndim == 2:
            return action.unsqueeze(-1) if action.shape[1] == target_channels else action.unsqueeze(1)

        # Handle 3D: (B, T, A) -> (B, A, T)
        elif action.ndim == 3:
            if action.shape[1] != target_channels and action.shape[2] == target_channels:
                return action.transpose(1, 2)  # Converts (300, 1, 2) -> (300, 2, 1)
            return action

        # Handle 4D (CEM candidates): (B, S, T, A) -> (B*S, A, T)
        elif action.ndim == 4:
            b, s, t, a = action.shape
            action = action.reshape(b * s, t, a)
            if action.shape[2] == target_channels:
                return action.transpose(1, 2)
            return action

        return action


    def set_use_decoder(self, use_decoder):
        self.using_decoder = use_decoder

    def set_freeze_backbone(self, freeze_backbone):
        self.freeze_backbone = freeze_backbone
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters()) # Only optimize parameters that require gradients
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        return optimizer

    def forward(self, x, action):
        h = self.backbone(x, action)
        if self.using_decoder and self.decoder is not None:
            return self.decoder(h)
        else:
            return h

    def set_loss(self, new_loss):
        # If a new loss is necessary when changing the decoder
        self.loss = new_loss

    def training_step(self, batch, batch_idx):
        # batch é a tupla (input, action, target)
        output = self.forward(batch[0], batch[1])
        loss = self.loss(output, batch[2])
        self.log("train_loss", loss, prog_bar=True)
        return loss

def train_wm(model, max_epochs=10, accelerator="gpu", devices=1, datamodule=None):

    trainer = L.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=devices)

    trainer.fit(model, datamodule)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def evaluate_reconstruction_wm(model, dataloader):

    trainer = L.Trainer(accelerator="gpu", devices=1)

    results = trainer.test(model, dataloader)

    return results


def evaluate_simulation_wm(
    model, 
    ambient: str = "pusht", 
    dataset_path: str = None, 
    custom_config: dict = None
) -> float:
    os.environ["MUJOCO_GL"] = "egl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configurações padrão incluindo os parâmetros do CEMSolver
    cfg = custom_config or {
        "env_name": "swm/PushT-v1",
        "dataset_name": "pusht_expert_train",
        "keys_to_cache": ["action", "proprio", "state"],
        "eval_budget": 50,
        "num_eval": 50,
        "goal_offset_steps": 25,
        "img_size": 224,
        "seed": 42,
        "horizon": 5,
        "receding_horizon": 5,
        "action_block": 5,
        "num_samples": 300,
        "n_steps": 30,
        "topk": 30,
        "var_scale": 1.0,
        "callables": [
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
    }

    # 1. Carregamento Direto do Dataset Local
    if dataset_path and Path(dataset_path).exists():
        dataset = HDF5Dataset(path=dataset_path, keys_to_cache=cfg["keys_to_cache"])
    else:
        raw_dir = os.environ.get("LOCAL_DATASET_DIR") or os.environ.get("STABLEWM_HOME")
        base_dir = Path(raw_dir) if raw_dir else Path(swm.data.utils.get_cache_dir())
        
        possible_paths = [
            base_dir / f"{cfg['dataset_name']}.h5",
            base_dir / "datasets" / f"{cfg['dataset_name']}.h5",
            Path.home() / ".stable_worldmodel" / "datasets" / f"{cfg['dataset_name']}.h5",
        ]
        h5_path = next((p for p in possible_paths if p.exists()), None)
        
        if h5_path:
            dataset = HDF5Dataset(path=h5_path, keys_to_cache=cfg["keys_to_cache"])
        else:
            raise FileNotFoundError(
                f"Dataset não encontrado localmente. "
                f"Passe o caminho exato via `dataset_path='/caminho/para/{cfg['dataset_name']}.h5'`."
            )

    # 2. Transformações de Imagem
    img_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=(cfg["img_size"], cfg["img_size"])),
    ])
    transform = {"pixels": img_transform, "goal": img_transform, "goal_pixels": img_transform,}

    # 3. Pré-processadores de Colunas do Dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg["keys_to_cache"]:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # 4. Preparação do Modelo de Mundo
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True

    # 5. Instanciação do Solver CEM com Parâmetros de Planejamento
    solver = swm.solver.cem.CEMSolver(
        model=model,
        num_samples=cfg.get("num_samples", 300),
        n_steps=cfg.get("n_steps", 30),
        topk=cfg.get("topk", 30),
        var_scale=cfg.get("var_scale", 1.0),
        device=device,
        seed=cfg["seed"],
    )
    plan_config = swm.PlanConfig(
        horizon=cfg["horizon"],
        receding_horizon=cfg["receding_horizon"],
        action_block=cfg["action_block"],
    )

    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=plan_config, process=process, transform=transform
    )

    # 6. Instanciação do Ambiente
    world = swm.World(
        env_name=cfg["env_name"],
        num_envs=cfg["num_eval"],
        max_episode_steps=2 * cfg["eval_budget"],
        image_shape=(cfg["img_size"], cfg["img_size"]),
    )
    world.set_policy(policy)

    # 7. Seleção de Episódios Válidos
    step_idx = dataset.get_col_data("step_idx")
    ep_col_data = dataset.get_col_data(col_name)

    lengths = np.array([np.max(step_idx[ep_col_data == ep_id]) + 1 for ep_id in ep_indices])
    max_start_idx = lengths - cfg["goal_offset_steps"] - 1
    max_start_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}

    max_start_per_row = np.array([max_start_dict[ep_id] for ep_id in ep_col_data])
    valid_mask = step_idx <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(cfg["seed"])
    sampled_indices = g.choice(len(valid_indices) - 1, size=cfg["num_eval"], replace=False)
    random_episode_indices = np.sort(valid_indices[sampled_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    # 8. Avaliação
    metrics = world.evaluate(
        dataset=dataset, # Source dataset for dataset-driven eval
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg["goal_offset_steps"], # = 25
        eval_budget=cfg["eval_budget"], # = 50
        episodes_idx=eval_episodes.tolist(), # Dataset episode indices, one per env
        callables=cfg["callables"], # = [ {"method": "_set_state", "args": {"state": {"value": "state"}}}, {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},],
        video="./videos_pusht", # Directory to write one mp4 per episode/env
    )# reset mode = 'wait' (freeze terminated envs and stop when all are done)
    """Run the attached policy and return aggregated metrics.

        Two modes of operation:

        * **Episodic (default)**: set ``episodes`` to the number of
          episodes to roll out. Terminated envs are auto-reset until the
          target count is reached.

        * **Dataset-driven**: pass ``dataset`` with ``episodes_idx`` /
          ``start_steps`` / ``goal_offset`` / ``eval_budget``. Each env
          is seeded from one dataset episode, starts at
          ``start_steps[i]`` and targets the state at
          ``start_steps[i] + goal_offset``. Run length is capped at
          ``eval_budget`` steps. Requires ``num_envs == len(episodes_idx)``.

        Args:
            episodes: Total episodes to roll out (episodic mode).
            seed: Base seed. Per-env seeds are derived by offsetting it.
            options: Reset options forwarded to ``envs.reset``.
            video: Directory to write one mp4 per episode/env (optional).
            reset_mode: ``'auto'`` (reset terminated envs) or ``'wait'``
                (freeze terminated envs and stop when all are done).
                Defaults to ``'auto'`` for episodic eval and ``'wait'``
                for dataset eval.
            dataset: Source dataset for dataset-driven eval.
            episodes_idx: Dataset episode indices, one per env.
            start_steps: Starting step within each dataset episode.
            goal_offset: Offset from each start step that defines the goal.
            eval_budget: Max env steps per episode in dataset mode.
            callables: Per-env setup calls applied on the unwrapped env
                after reset. Each spec is
                ``{'method': name, 'args': {arg_name: {'value': ...,
                'in_dataset': bool}}}``; if ``in_dataset`` is True, the
                ``value`` names a key in the sliced dataset state and the
                per-env value is deep-copied in.

        Returns:
            A dict with ``'success_rate'`` (percent), ``'episode_successes'``
            (per-episode bool/uint array), and ``'seeds'`` used for reset.
        """


    return metrics



if __name__ == "__main__":

    print("Starting World Model Training...")
    model = WorldModel(backbone=LeWM_backbone(), loss=LeWM_loss(sigreg_weight=1.0), learning_rate=1e-4)

    print("Model initialized.")
    # 4. Carregar Dados de Exemplo
    dataset = HDF5Dataset(
    path="/home/nicolas.silva/WorldModels/worldmodels/experiments/leWM/le-wm/datasets/pusht_expert_train.h5")
    print("Dataset loaded.")
    dataset = Subset(dataset, range(1000))
    datamodule = LeWMDataModule(dataset, batch_size=8, num_workers=1)

    print("DataModule created.")
    # 5. Treinar e Salvar
    train_wm(model, max_epochs=1, accelerator="cpu", datamodule=datamodule)
    print("Training completed.")
    save_model(model, "world_model.pth")

    print("Model saved########################################.")
    sr = evaluate_simulation_wm(model, ambient='pusht', dataset_path="/home/nicolas.silva/WorldModels/worldmodels/experiments/leWM/le-wm/datasets/pusht_expert_train.h5")
    print(f"Success Rate: {sr}")
    # evaluate_reconstruction_wm(model, dataloader=None)



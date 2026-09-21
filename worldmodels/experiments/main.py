import torch
import lightning as L
import stable_pretraining as spt
import stable_worldmodel as swm
from leWM.lewm import LeWM_backbone, LeWM_loss, LeWMDataModule
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset

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

def evaluate_simulation_wm(model, ambient):
    success_rate = model.evaluate(ambient)
    return success_rate

def evaluate_reconstruction_wm(model, dataloader):

    trainer = L.Trainer(accelerator="gpu", devices=1)

    results = trainer.test(model, dataloader)

    return results


if __name__ == "__main__":

    print("Starting World Model Training...")
    model = WorldModel(backbone=LeWM_backbone(), loss=LeWM_loss(sigreg_weight=1.0), learning_rate=1e-4)

    print("Model initialized.")
    # 4. Carregar Dados de Exemplo
    dataset = HDF5Dataset(
    path="/home/nicolas.silva/WorldModels/worldmodels/experiments/leWM/le-wm/datasets/pusht_expert_train.h5")
    print("Dataset loaded.")
    datamodule = LeWMDataModule(dataset, batch_size=8)

    print("DataModule created.")
    # 5. Treinar e Salvar
    train_wm(model, max_epochs=5, accelerator="cpu", datamodule=datamodule)
    print("Training completed.")
    save_model(model, "world_model.pth")


    # evaluate_simulation_wm(model, ambient=None)
    # evaluate_reconstruction_wm(model, dataloader=None)
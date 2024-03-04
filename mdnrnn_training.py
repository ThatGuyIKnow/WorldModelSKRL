import os
import numpy as np
from pathlib import Path

import torch
import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torchvision

from Models.MDNRNN import MDNRNN
from Models.VAE import VAE

from Utils.EpisodeDataset import EpisodeDataset
from Utils.TransformerWrapper import TransformWrapper
from Utils.utils import transpose_2d

# Data Paths
DATASET_PATH = 'data/carracing-v2/main_details_simulation.csv'
CHECKPOINT_PATH = 'runs'

# Training parameters
SEED = 42
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 16
NUM_WORKERS = 4
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 30
VAL_SPLIT = 0.1

# Model parameters
ACTION_SPACE = 3
LATENT_DIM =  32
SEQ_LENGTH = 32
H_SPACE = 64
GAUSSIAN_SPACE = 5

# Logging and Model Saving
WANDB_KWARGS = {
    'log_model': "all", 
    'prefix': 'mdnrnn', 
    'project': 'world_model_mdnrnn',
}
VAE_CHECKPOINT_REFERENCE = "team-good-models/model-registry/WorldModelVAE:latest"

# Setting the seed
L.seed_everything(SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Device:", DEVICE)

wandb_logger = WandbLogger(**WANDB_KWARGS)
vae_dir = wandb_logger.download_artifact(VAE_CHECKPOINT_REFERENCE, artifact_type="model")
encoding_model = VAE.load_from_checkpoint(Path(vae_dir) / "model.ckpt").to(DEVICE)
encoding_model.freeze()

transform = TransformWrapper.transform

class GenerateCallback(L.Callback):
    def __init__(self, input_imgs, vae, action_shape, every_n_epochs=1):
        super().__init__()
        self.action_shape = action_shape
        
        self.encoder = vae.encoder.to(DEVICE)
        self.decoder = vae.decoder.to(DEVICE)
        images, actions, _, _, next_images = input_imgs
        self.input_actions = torch.stack(actions).to(DEVICE)
        self.input_imgs = torch.stack([img[-1] for img in next_images]).unsqueeze(dim=1).to(DEVICE)
        self.sample_count = len(self.input_imgs)

        self.input_latent = torch.stack([self._to_latent(img.to(DEVICE)) for img in images])  # Latents to reconstruct during training
        
        self.image_reconst = self.decoder(self.input_latent[:, -1])
        self.every_n_epochs = every_n_epochs

    def _to_latent(self, image):
        _, _, latent = self.encoder(image)
        return latent.squeeze(dim=0).clone().detach()

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_latent = self.input_latent.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                mus, sigmas, logpis, _, _, _ = pl_module.forward(input_latent, self.input_actions)

                mus = mus[:, -1]
                sigmas = sigmas[:, -1]
                logpis = logpis[:, -1]

                next_latents = pl_module.sample(mus, sigmas, logpis)
                reconst_imgs = self.decoder(next_latents)
                pl_module.train()

            imgs = torch.stack([self.input_imgs[:, -1], self.image_reconst, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=3, normalize=True)
            trainer.logger.log_image(key="Reconstructions_Next_Step", images=[grid], step=trainer.global_step)

def get_car_racing_dataset():
    train_dataset = EpisodeDataset(DATASET_PATH, transform=transform, action_space=ACTION_SPACE, seq_length=SEQ_LENGTH, encoder=encoding_model.encoder, device=DEVICE)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [1-VAL_SPLIT, VAL_SPLIT])

    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, num_workers=NUM_WORKERS, multiprocessing_context='spawn')
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, multiprocessing_context='spawn')

    return train_loader, val_loader

def get_seq_input_imgs(n=8):
    dataset = EpisodeDataset(DATASET_PATH, transform=transform, action_space=ACTION_SPACE, seq_length=SEQ_LENGTH, encoder=encoding_model.encoder)
    idx = np.random.randint(0, len(dataset), size=n)
    return transpose_2d([dataset.get_image_seq(i) for i in idx])

def train_mdnrnn():
    train_loader, val_loader = get_car_racing_dataset()
    input_seqs = get_seq_input_imgs()

    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "mdnrnn_%i" % LATENT_DIM),
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            GenerateCallback(input_seqs, encoding_model, ACTION_SPACE, 1),
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor='val_loss', mode='min', patience=EARLY_STOPPING_PATIENCE, check_on_train_epoch_end=False)
        ],
        logger=wandb_logger,
        accelerator="auto",
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    pretrained_filename = os.path.join(CHECKPOINT_PATH, "mdnrnn_best.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = MDNRNN.load_from_checkpoint(pretrained_filename)
    else:
        model = MDNRNN(LATENT_DIM, ACTION_SPACE, H_SPACE, GAUSSIAN_SPACE)

        
        trainer.fit(model, train_loader, val_loader)
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    result = {"val": val_result}
    return model, result


if __name__ == '__main__':
    train_mdnrnn()
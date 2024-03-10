import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import gymnasium as gym
import torchvision
import wandb

from Models.MDNRNN import MDNRNN
from Models.VAE import VAE
from Models.WorldModel import WorldModel

from Utils.EpisodeDataset import EpisodeDataset
from Utils.TransformerWrapper import TransformWrapper
from Utils.utils import transpose_2d

# Data Paths
CHECKPOINT_PATH = 'runs'

# Training parameters
SEED = 42
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
NUM_ENVS = 5
BATCH_SIZE = 16
NUM_WORKERS = 0
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 30
VAL_SPLIT = 0.1

# Model parameters
ACTION_SPACE = 3
LATENT_DIM = 32
IMG_CHANNELS = 1
HIDDEN_DIM = 256
SEQ_LENGTH = 1000
GAUSSIAN_SPACE = 5
WANDB_KWARGS = {
    'log_model': "all", 
    'prefix': 'world_model', 
    'project': 'full_world_model',
}
print("Device:", DEVICE)
wandb_logger = WandbLogger(**WANDB_KWARGS)
# Setting the seed
L.seed_everything(SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GenerateCallback(L.Callback):
    def __init__(self, dataset, model, every_n_epochs=1):
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            states, actions, rewards, dones, next_states = self.dataset[0]

            with torch.no_grad():
                pl_module.eval()
                latents, hiddens, reconstruction_obs, posterior_z, est_reward = pl_module.forward_full(states, actions)
                recon_posterior = self.model.vae.decoder(posterior_z.view(-1, LATENT_DIM)).view(SEQ_LENGTH, NUM_ENVS, 64, 64)
                reconst_obs = torch.concat([v for v in reconstruction_obs.swapaxes(0, 1)], dim=-2)
                reconst_obs = reconst_obs.squeeze(1)
                reconst_post_obs = torch.concat([v for v in recon_posterior.swapaxes(0, 1)], dim=-2)
                pl_module.train()
            state_img = torch.concat([v for v in states.swapaxes(0, 1)], dim=-2)
            imgs = torch.concat([state_img, reconst_obs, reconst_post_obs], dim=-1)
            imgs = imgs.unsqueeze(1).cpu().numpy()
            imgs = np.clip(np.repeat(imgs, 3, axis=1) * 255, 0, 255).astype(np.uint8)
            trainer.logger.log_video(key="video", videos=[imgs,], step=trainer.global_step)


def train_world_model():
    train_loader = EpisodeDataset(lambda: TransformWrapper(gym.make("CarRacing-v2", render_mode='rgb_array')), 
                                  num_envs=NUM_ENVS, max_steps=SEQ_LENGTH, skip_first=50, device=DEVICE)
    
    
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "mdnrnn_best.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = WorldModel.load_from_checkpoint(pretrained_filename)
    else:
        model = WorldModel(train_loader.envs.observation_space, 
                           train_loader.envs.action_space, 
                           IMG_CHANNELS, 
                           LATENT_DIM, 
                           HIDDEN_DIM,
                           GAUSSIAN_SPACE,
                           SEQ_LENGTH)


    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "wm_%i" % LATENT_DIM),
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            GenerateCallback(train_loader, model, 5),
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor='train_loss', mode='min', patience=EARLY_STOPPING_PATIENCE)
        ],
        logger=wandb_logger,
        accelerator="auto",
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

        
    trainer.fit(model, train_loader)
    val_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    result = {"val": val_result}
    return model, result


if __name__ == '__main__':
    train_world_model()
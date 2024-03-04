import os
from pathlib import Path
import numpy as np

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.utils.data as data
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
import torchvision

from Models.VAE import VAE
from Utils.DataOnlyLoader import DataOnlyLoader
from Utils.TransformerWrapper import TransformWrapper

# Hyperparameters
# Paths
DATASET_PATH = 'data'  # Path to the folder where the datasets are/should be downloaded
CHECKPOINT_PATH = 'runs'  # Path to the folder where the pretrained models are saved

# Device parameters
SEED = 42
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Model parameters
LATENT_DIM = 32
IMG_CHANNELS = 1

# Training parameters
BATCH_SIZE = 256
NUM_WORKERS = 4
MAX_EPOCHS = 100
SAVE_EVERY_N_EPOCHS = 5
VAL_SPLIT = 0.1
IMAGE_VIS_COUNT = 8
EARLY_STOPPING_PATIENCE = 10

# Setting the seed for reproducibility
L.seed_everything(SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GenerateCallback(Callback):
    """
    Custom callback for generating and logging reconstructed images during training.
    """
    def __init__(self, input_imgs, every_n_epochs=1):
        """
        Initialize the callback with input images for visualization.

        Args:
            input_imgs (torch.Tensor): Images to reconstruct during training.
            every_n_epochs (int): Log reconstructed images every N epochs.
        """
        super().__init__()
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Generate and log reconstructed images at the end of each training epoch.

        Args:
            trainer: PyTorch Lightning trainer object.
            pl_module: PyTorch Lightning module being trained.
        """
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs, _, _, _ = pl_module(input_imgs)
                pl_module.train()
            imgs = torch.concat([input_imgs, reconst_imgs], dim=-2)
            grid = torch.concat([img for img in imgs], dim=-1)
            trainer.logger.log_image(key="Reconstructions", images=[grid], step=trainer.global_step)

def get_car_racing_dataset():
    """
    Load and split the Car Racing dataset into training and validation sets.

    Returns:
        tuple: Validation set, training DataLoader, and validation DataLoader.
    """
    transform = TransformWrapper.transform
    train_dataset = torchvision.datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [1 - VAL_SPLIT, VAL_SPLIT])

    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)

    train_loader = DataOnlyLoader(train_loader)
    val_loader = DataOnlyLoader(val_loader)
    return val_set, train_loader, val_loader

def train_vae():
    """
    Train the Variational Autoencoder (VAE) model.
    """
    val_set, train_loader, val_loader = get_car_racing_dataset()

    # Create a PyTorch Lightning trainer with the generation callback
    wandb_logger = WandbLogger(log_model="all", prefix='vae')
    idx = np.random.randint(0, len(val_set), size=IMAGE_VIS_COUNT)
    training_images = torch.stack([val_set[i][0] for i in idx], dim=0)

    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "vae_%i" % LATENT_DIM),
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback(training_images, every_n_epochs=SAVE_EVERY_N_EPOCHS),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor='val_loss', mode='min', patience=EARLY_STOPPING_PATIENCE, check_on_train_epoch_end=False),
        ],
        logger=wandb_logger,
        gradient_clip_val=0.5
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    # Check if a pretrained model exists; if not, train a new one
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "vae_best.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = VAE.load_from_checkpoint(pretrained_filename)
    else:

        vae_checkpoint_reference = "team-good-models/lightning_logs/model-fe6xxfhz:v33"
        vae_dir = wandb_logger.download_artifact(vae_checkpoint_reference, artifact_type="model")
        model = VAE.load_from_checkpoint(Path(vae_dir) / "model.ckpt")
        #model = VAE(IMG_CHANNELS, LATENT_DIM)
        trainer.fit(model, train_loader, val_loader)
    
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    result = {"val": val_result}
    return model, result

train_vae()

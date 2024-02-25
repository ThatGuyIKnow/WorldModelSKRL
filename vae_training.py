import os

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


# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = 'data'
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = 'runs'

# Setting the seed
L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

class GenerateCallback(Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs, _, _, _ = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True)
            trainer.logger.log_image(key="Reconstructions", images=[grid], step=trainer.global_step)



def get_car_racing_dataset():
    # Transformations applied on each image => only make them a tensor
    transform = TransformWrapper.transform

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = torchvision.datasets.ImageFolder(DATASET_PATH, transform=transform)    
    train_set, val_set = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)

    train_loader = DataOnlyLoader(train_loader)
    val_loader = DataOnlyLoader(val_loader)
    return train_dataset, train_loader, val_loader


def train_vae(latent_dim=32):
    train_dataset, train_loader, val_loader = get_car_racing_dataset()
    # Create a PyTorch Lightning trainer with the generation callback
    wandb_logger = WandbLogger(log_model="all")
    training_images = torch.stack([train_dataset[i][0] for i in range(8)], dim=0)
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "vae_%i" % latent_dim),
        accelerator="auto",
        devices=1,
        max_epochs=100,
        limit_train_batches=100,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback(training_images, every_n_epochs=10),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor='train_loss', mode='min', patience=10)
        ],
        logger=wandb_logger,
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "vae_best.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = VAE.load_from_checkpoint(pretrained_filename)
    else:
        model = VAE(1, latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    result = {"val": val_result}
    return model, result

train_vae()
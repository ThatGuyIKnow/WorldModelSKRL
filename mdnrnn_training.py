import datetime
import os
from random import randint

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from Models.MDNRNN import MDNRNN
from Models.VAE import VAE

from Utils.EpisodeDataset import EpisodeDataset, get_car_racing_dataset, car_preprocessing_transform
from Utils.ModelCheckpoint import retrieve_wandb_checkpoint

from pytorch_lightning.loggers import WandbLogger

# Batch size
BATCH_SIZE = 4
# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = 'data/carracing-v2/details_simulation.csv'
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = 'runs'
# Path to the encoding/decoding VAE model
ENC_PATH = retrieve_wandb_checkpoint(user='team-good-models', project='world_model', model_run='model-zuhrjnxl')

# Setting the seed
L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

encoding_model = VAE.load_from_checkpoint(ENC_PATH)


class GenerateImaginedCallback(L.Callback):
    def __init__(self, input_imgs, vae, action_shape, every_n_epochs=1):
        super().__init__()
        self.action_shape = action_shape
        
        self.encoder = vae.encoder
        self.decoder = vae.decoder

        self.input_actions = nn.utils.rnn.pack_sequence([img['actions'] for img in input_imgs])
        self.input_imgs = torch.stack([img['images'][-1] for img in input_imgs]).unsqueeze(dim=1)
        self.sample_count = len(self.input_imgs)

        input_latent = torch.stack([self._to_latent(img['images']) for img in input_imgs])  # Latents to reconstruct during training
        self.input_latent = nn.utils.rnn.pack_sequence(input_latent[:, :-1])
        
        self.image_reconst = self.decoder(input_latent[:, -1])
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def _to_latent(self, image):
        _, _, latent = self.encoder(image)
        return latent.squeeze(dim=0).clone().detach()

    def _get_random_actions(self, c, s):
        return torch.stack([F.one_hot(randint(c), c) for _ in range(s)])

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_latent = self.input_latent.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
#                actions = self._get_random_actions(self.action_space, self.sample_count)
                mus, sigmas, logpis, _, _, _ = pl_module({'latent':input_latent, 'actions':self.input_actions})

                mus = torch.stack(mus)[:, -1]
                sigmas = torch.stack(sigmas)[:, -1]
                logpis = torch.stack(logpis)[:, -1]

                next_latents = pl_module.sample(mus, sigmas, logpis)
                reconst_imgs = self.decoder(torch.stack(next_latents))
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([self.input_imgs[:, -1], self.image_reconst, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=3, normalize=True).cpu().numpy()
            trainer.logger.log_image("Reconstructions", images=[np.moveaxis(grid, 0, -1)])


def get_seq_input_imgs(n=2, p=0.5):
    dataset = EpisodeDataset(DATASET_PATH, transform=car_preprocessing_transform)
    seqs = []
    for i in range(n):
        seq = dataset[i]
        seq_len = int(seq['images'].shape[0] * p)

        seq['images'] = seq['images'][:seq_len]
        seq['actions'] = seq['actions'][:seq_len]
        seq['rewards'] = seq['rewards'][:seq_len]
        seq['dones'] = seq['dones'][:seq_len]

        seqs.append(seq)
    return seqs

def train_mdnrnn(latent_dim=32, action_space=5, h_space=64, gaussian_space=64):
    train_loader, val_loader = get_car_racing_dataset(DATASET_PATH, BATCH_SIZE, encoding_model=encoding_model.encoder)
    input_seqs = get_seq_input_imgs()
    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project='world_model', name=f'mdnrnn_{latent_dim}_{datetime.datetime.today()}', log_model="all")

    # add your batch size to the wandb config
    wandb_logger.experiment.config["batch_size"] = BATCH_SIZE
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "mdnrnn_%i" % latent_dim),
        accelerator="auto",
        devices=1,
        max_epochs=500,
        callbacks=[
            GenerateImaginedCallback(input_seqs, encoding_model, action_space, 10),
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor='val_loss', mode='min', patience=30)
        ],
        logger=wandb_logger
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "mdnrnn_best.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = MDNRNN.load_from_checkpoint(pretrained_filename)
    else:
        model = MDNRNN(latent_dim, action_space, h_space, gaussian_space, device=device)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    result = {"val": val_result}
    return model, result

train_mdnrnn(32, 5, 64, 64)
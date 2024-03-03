import os
from random import randint
import numpy as np

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torchvision
from torchvision import transforms
from Models.MDNRNN import MDNRNN
from Models.VAE import VAE
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path

from Utils.EpisodeDataset import EpisodeDataset
from Utils.TransformerWrapper import TransformWrapper

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

# Model parameters
ACTION_SPACE = 3
LATENT_DIM =  32
SEQ_LENGTH = 32
H_SPACE = 64
GAUSSIAN_SPACE = 5

# Logging and Model Saving
WANDB_KWARGS = {'log_model': "all"}
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

transform = TransformWrapper.transform

class GenerateCallback(L.Callback):
    def __init__(self, input_imgs, vae, action_shape, every_n_epochs=1):
        super().__init__()
        self.action_shape = action_shape
        
        self.encoder = vae.encoder
        self.decoder = vae.decoder

        self.input_actions = nn.utils.rnn.pack_sequence([img['actions'].to(DEVICE) for img in input_imgs])
        self.input_imgs = torch.stack([img['images'][-1] for img in input_imgs]).unsqueeze(dim=1).to(DEVICE)
        self.sample_count = len(self.input_imgs)

        input_latent = torch.stack([self._to_latent(img['images'].to(DEVICE)) for img in input_imgs])  # Latents to reconstruct during training
        self.input_latent = nn.utils.rnn.pack_sequence(input_latent[:, :-1].to(DEVICE))
        
        self.image_reconst = self.decoder(input_latent[:, -1])
        self.every_n_epochs = every_n_epochs

    def _to_latent(self, image):
        _, _, latent = self.encoder(image)
        return latent.squeeze(dim=0).clone().detach()

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_latent = self.input_latent.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                mus, sigmas, logpis, _, _, _ = pl_module({'latent':input_latent, 'actions':self.input_actions})

                mus = torch.stack(mus)[:, -1]
                sigmas = torch.stack(sigmas)[:, -1]
                logpis = torch.stack(logpis)[:, -1]

                next_latents = pl_module.sample(mus, sigmas, logpis)
                reconst_imgs = self.decoder(torch.stack(next_latents))
                pl_module.train()

            imgs = torch.stack([self.input_imgs[:, -1], self.image_reconst, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=3, normalize=True)
            trainer.logger.log_image(key="Reconstructions_Next_Step", images=[grid], step=trainer.global_step)

def get_car_racing_dataset():
    train_dataset = EpisodeDataset(DATASET_PATH, transform=transform, action_space=ACTION_SPACE, seq_length=SEQ_LENGTH)    
    train_set, val_set = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, num_workers=NUM_WORKERS, collate_fn=EpisodeDataset.collate_fn)
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, collate_fn=EpisodeDataset.collate_fn)

    return train_loader, val_loader

def get_seq_input_imgs(n=8, p=0.5):
    dataset = EpisodeDataset(DATASET_PATH, transform=transform, action_space=ACTION_SPACE)
    seqs = []
    idx = np.random.randint(0, len(dataset), size=n)
    for i in idx:
        seq = dataset[i]
        seq_len = int(seq['images'].shape[0] * p)

        seq['images'] = seq['images'][:seq_len]
        seq['actions'] = seq['actions'][:seq_len]
        seq['rewards'] = seq['rewards'][:seq_len]
        seq['dones'] = seq['dones'][:seq_len]

        seqs.append(seq)
    return seqs

def train_mdnrnn():
    train_loader, val_loader = get_car_racing_dataset()
    input_seqs = get_seq_input_imgs()

    wandb_logger = WandbLogger(log_model="all", prefix='mdnrnn')
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
        model.encoding = encoding_model.encoder
    else:
        model = MDNRNN(LATENT_DIM, ACTION_SPACE, H_SPACE, GAUSSIAN_SPACE, device=DEVICE)
        model.encoding = encoding_model.encoder
        trainer.fit(model, train_loader, val_loader)
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    result = {"val": val_result}
    return model, result

train_mdnrnn()
from typing import Optional, Tuple, Union
import gym
import gymnasium
import torch

from Models.VAE import VAE
from Models.MDNRNN import MDNRNN
from skrl.models.torch import Model, DeterministicMixin

from Utils.utils import prefix_keys, transpose_2d

"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import lightning as L
import wandb

class WorldModel(L.LightningModule):
    """ Variational Autoencoder """
    def __init__(self, observation_space, action_space, img_channels, latent_size, hidden_space, gaussian_space, sequence_length):
        super().__init__()
        self.save_hyperparameters()

        self.observation_space = observation_space
        self.action_space = action_space.shape[-1]
        self.img_channels = img_channels
        self.latent_size = latent_size
        self.gaussian_space = gaussian_space
        self.hidden_space = hidden_space
        self.sequence_length = sequence_length

        self.vae = VAE(img_channels, latent_size, hidden_space)
        self.mdnrnn = MDNRNN(latent_size, self.action_space, hidden_space, gaussian_space, include_reward_loss=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
    
    def prepare_loss_dict(self, *losses):
        names = ['recon_loss', 'reg_loss', 'gmm_loss', 'termination_loss', 'reward_loss']
        log = {k: v for k, v in zip(names, losses)}
        
        loss = sum(losses)
        log['loss'] = loss
        return log

    # ============================================
    # ============== PASS FUNCTIONS ==============
    # ============================================
    def _get_losses(
        self, images: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
        dones: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden = self.mdnrnn.initial_state(batch_size=images.shape[1]) if hidden is None else hidden
        
        losses = []
        outputs = []
        for image, action, reward, done in zip(images, actions, rewards, dones):
            image = image.squeeze(0).unsqueeze(1)
            action = action.squeeze(0)
            
            recon_loss, reg_loss, z_mu = self.vae.get_loss(image, hidden[0])
            gmm_loss, termination_loss, reward_loss, next_hidden = self.mdnrnn.cell.get_loss(z_mu, action, reward, done, hidden)
            
            loss_dict = self.prepare_loss_dict(recon_loss, reg_loss, gmm_loss, termination_loss, reward_loss)
            losses.append(loss_dict)
            outputs.append([z_mu, hidden])
            hidden = next_hidden

        loss_dict = {k: sum([d[k] for d in losses])/len(losses) for k in losses[0].keys()}

        outputs = transpose_2d(outputs)
        latents = torch.stack(outputs[0]).squeeze()
        hiddens = outputs[1]

        return loss_dict, latents, hiddens


    def forward(
        self, images: torch.Tensor, actions: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        latents, hiddens, reconstruction_obs, post_reconstruction_obs, est_reward = self.forward_full(images, actions, hidden)
        return latents, hiddens
        
        
    def forward_full(
        self, images: torch.Tensor, actions: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden = self.mdnrnn.initial_state(batch_size=images.shape[1]) if hidden is None else hidden
        
        outputs = []
        for image, action in zip(images, actions):
            image = image.squeeze(0).unsqueeze(1)
            action = action.squeeze(0)
            
            reconstruction, mu, logsigma, z = self.vae(image, hidden[0])
            mus, sigmas, logpi, reward, done, next_hidden = self.mdnrnn.cell(mu, action, hidden)
            posterior_z = self.mdnrnn.cell.sample(mus, sigmas, logpi)

            outputs.append([mu, hidden, reconstruction, posterior_z, reward])
            hidden = next_hidden

        outputs = transpose_2d(outputs)
        latents = torch.stack(outputs[0])
        hiddens = outputs[1]
        reconstruction_obs = torch.stack(outputs[2])
        posterior_z = torch.stack(outputs[3])
        est_rewards = torch.stack(outputs[4])

        return latents, hiddens, reconstruction_obs, posterior_z, est_rewards



    # ============================================
    # ============== STEP FUNCTIONS ==============
    # ============================================
    def step(self, batch, batch_idx):
        states, actions, rewards, dones, next_states = batch
        losses, latents, hiddens = self._get_losses(states, actions, rewards, dones)

        return losses
        
    def training_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        self.log_dict(prefix_keys(losses, 'train_'), on_step=True)
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        self.log_dict(prefix_keys(losses, 'val_'), on_step=True)
        return losses['loss']
        
    def test_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        self.log_dict(prefix_keys(losses, 'test_'), on_step=True)
        return losses['loss']
    
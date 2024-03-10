from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.distributions import (Normal as Gaussian, Categorical, MixtureSameFamily, Independent)
import torch.nn.functional as F

import lightning as L
from Losses.GMMLoss import bce_with_logits_list, gaussian_mixture_loss, mse_loss_list

from Utils.ReduceLROnPlateau import ReduceLROnPlateau
from Utils.utils import prefix_keys, transpose_2d

class MDRNNCell_(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians)
        self.prediction_heads = nn.Linear(latents + hiddens, 2)
        
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

        stride = self.gaussians * self.latents
        self.splits = [stride, stride, gaussians]

    def forward(self, batch, action, hidden): # pylint: disable=arguments-differ
        """ ONE STEP forward.

        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: (BSIZE, LSIZE) torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        """
        #in_al = in_al.view(*in_al.shape[1:])

        preds = self.prediction_heads(torch.cat([batch, hidden[0]], dim=-1))
        reward, done = preds[:,0], preds[:,1]
        next_hidden = self.rnn(torch.cat([batch, action], dim=-1), hidden)

        out_full = self.gmm_linear(hidden[0])

        mus, sigmas, pi = torch.split(out_full, self.splits, dim=1)

        mus = mus.view(-1, self.gaussians, self.latents)
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        pi = pi.view(-1, self.gaussians)

        sigmas = torch.exp(sigmas)
        logpi = F.log_softmax(pi, dim=-1)

        return mus, sigmas, logpi, reward, done, next_hidden

    def get_loss(self, latent, actions, rewards, dones, hidden):
        mus, sigmas, logpi, rs, ds, next_hidden = self.forward(latent, actions, hidden)

        gmm_loss = gaussian_mixture_loss(latent, mus, sigmas, logpi)
        termination_loss = bce_with_logits_list(ds, dones)
        reward_loss = mse_loss_list(rs, rewards)
        
        loss = gmm_loss + termination_loss + reward_loss

        return gmm_loss, termination_loss, reward_loss, next_hidden

    def sample(self, mus, sigmas, logpi):
        cat = Categorical(logpi)
        coll = Independent(Gaussian(mus, sigmas), 1)

        mixture = MixtureSameFamily(cat, coll)
        return mixture.sample()

class MDNRNN(L.LightningModule):
    # ============================================
    # ================= SETUP ====================
    # ============================================
    def __init__(self, latent_space, action_space, hidden_space, gaussian_space, cell = None, include_reward_loss=False):
        super().__init__()
        
        self.save_hyperparameters()

        self.action_space = action_space
        self.latent_space = latent_space
        self.hidden_space = hidden_space
        if cell is None:
            #self.cell = MDRNNCell_(latent_space, action_space, gaussian_space, hidden_space)
            self.cell = MDRNNCell_(latent_space, action_space, hidden_space, gaussian_space)
        else:
            self.cell = cell

        self.include_reward_loss = include_reward_loss

    def configure_optimizers(self):
        optimizer = RMSprop(self.parameters(), lr=1e-3, alpha=0.9)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


    # ============================================
    # ================== LOSSES ==================
    # ============================================
    def _get_losses(self, batch, include_reward = False):    
        latent, actions, rewards, dones, next_latent = batch
        norm_factor = latent[0].shape[-1]
        
        mus, sigmas, logpi, rs, ds, _ = self.forward(latent, actions)

        gmm_loss = gaussian_mixture_loss(next_latent, mus, sigmas, logpi)
        termination_loss = bce_with_logits_list(ds, dones)
        
        if include_reward:
            reward_loss = mse_loss_list(rs, rewards)
        else:
            reward_loss = 0

        return gmm_loss, termination_loss, reward_loss, norm_factor        



    # ============================================
    # ============== PASS FUNCTIONS ==============
    # ============================================
    def forward(
        self, latent: torch.Tensor, action: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        inputs = torch.cat([latent, action], dim=-1)
        hidden = self.initial_state(batch_size=inputs.shape[1]) if hidden is None else hidden
        outputs = []
        for i in range(len(inputs)):
            output = self.cell(inputs[i], hidden)
            out, hidden = output[:-1], output[-1]
            outputs += [out]
        outputs = transpose_2d(outputs)
        mus, sigmas, logpis, rs, ds = [torch.stack(v).squeeze() for v in outputs]

        return mus, sigmas, logpis, rs, ds, hidden

    def initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros((batch_size, self.hidden_space), device=self.device), torch.zeros((batch_size, self.hidden_space), device=self.device)

    def sample(self, mus, sigmas, logpi):
        cat = Categorical(logpi)
        coll = Independent(Gaussian(mus, sigmas), 1)

        mixture = MixtureSameFamily(cat, coll)
        return mixture.sample()


    # ============================================
    # ============== STEP FUNCTIONS ==============
    # ============================================
    def step(self, batch, batch_idx):
        gmm_loss, termination_loss, reward_loss, norm_factor = self._get_losses(batch)
        
        if self.include_reward_loss:
            loss = (gmm_loss + termination_loss + reward_loss) / (norm_factor + 2)
        else:
            loss = (gmm_loss + termination_loss) / (norm_factor + 1)

        losses = {
            'gmm_loss': gmm_loss / norm_factor,
            'termination_loss': termination_loss,
            'reward_loss': reward_loss,
            'loss_scaling': norm_factor + 2 if self.include_reward_loss else norm_factor + 1,
            'loss': loss
        }
        return losses
        
    def training_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        self.log_dict(prefix_keys(losses, 'train_'))
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        self.log_dict(prefix_keys(losses, 'val_'))
        return losses['loss']
        
    def test_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        self.log_dict(prefix_keys(losses, 'test_'))
        return losses['loss']
    
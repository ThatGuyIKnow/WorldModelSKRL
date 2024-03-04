from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import jit
from torch.optim import RMSprop
from torch.distributions import (Normal as Gaussian, Categorical, MixtureSameFamily, Independent)

import lightning as L
from Losses.GMMLoss import bce_with_logits_list, gaussian_mixture_loss, mse_loss_list

from Utils.ReduceLROnPlateau import ReduceLROnPlateau
from Utils.utils import prefix_keys, transpose_2d


class MDNRNNCell(jit.ScriptModule):
    def __init__(self, latents, actions, gaussians, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.latents = latents
        self.actions = actions
        self.gaussians = gaussians

        stride = self.gaussians * self.latents
        self.splits = [stride, stride, gaussians, 1, 1]

        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, self.latents + self.actions))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

        output_size = (2 * latents + 1) * gaussians + 2
        self.gmm = nn.Parameter(torch.randn(output_size, hidden_size))
        self.bias_gmm = nn.Parameter(torch.randn(output_size))


    @jit.script_method
    def forward(
        self, input: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = state
        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        
        zy = torch.mm(hy, self.gmm.t()) + self.bias_gmm
        mus, sigmas, logpi, r, d = torch.split(zy, self.splits, dim=1)

        mus = mus.view(-1, self.gaussians, self.latents)
        sigmas = mus.view(-1, self.gaussians, self.latents)
        logpi = logpi.view(-1, self.gaussians)
        
        sigmas = torch.exp(sigmas)
        logpi = torch.log_softmax(logpi, dim=-1)

        return mus, sigmas, logpi, r, d, (hy, cy)




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
            self.cell = MDNRNNCell(latent_space, action_space, gaussian_space, hidden_space)
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
        hidden_space = self.hidden_space
        return torch.zeros((batch_size, hidden_space), device=self.device), torch.zeros((batch_size, hidden_space), device=self.device)

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
    
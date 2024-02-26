from typing import Tuple, Union
import gym
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.distributions import (Normal as Gaussian, Categorical, MixtureSameFamily, Independent)

import lightning as L
from Losses.GMMLoss import bce_with_logits_list, gaussian_mixture_loss, mse_loss_list

from Utils.ReduceLROnPlateau import ReduceLROnPlateau

class MDRNNCell(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 2)
        
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden): # pylint: disable=arguments-differ
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
        if action.shape[-1] == 1:
            action = F.one_hot(action, self.actions).view(action.shape[0], self.actions)       
        #action = actin.view((*action.shape, 1))
        in_al = torch.cat([action, latent], dim=-1)

        #in_al = in_al.view(*in_al.shape[1:])

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = F.log_softmax(pi, dim=-1)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden

    def sample(self, mus, sigmas, logpi):
        cat = Categorical(logpi)
        coll = Independent(Gaussian(mus, sigmas), 1)

        mixture = MixtureSameFamily(cat, coll)
        return mixture.sample()


class MDNRNN(L.LightningModule):
    def __init__(self, 
                 latent_space: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                 action_space: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                 h_space: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                 gaussian_space: Union[int, Tuple[int], gym.Space, gymnasium.Space],                 
                 lookahead: int = 1,
                 temperature: float = 0.2,
                 encoding = None,
                 device: Union[str, torch.device] = 'cuda'):
        super().__init__()

        self.save_hyperparameters()

        self.latent_space = latent_space
        self.gaussian_space = gaussian_space
        self.h_space = h_space

        self.action_space = action_space
        if isinstance(action_space, gymnasium.spaces.Discrete):
            self.action_space = action_space.n
        
        self.temperature = temperature
        
        assert lookahead > 0
        self.lookahead = lookahead

        self.cell = MDRNNCell(self.latent_space, self.action_space, self.h_space, self.gaussian_space)

        self.encoding = encoding

    @property
    def rnn_space(self):
        """The rnn_space property."""
        return (self.latent_space + self.action_space.n, self.h_space)
    
    # @property
    # def mdn_space(self):
    #     """The mdn_space property."""
    #     return (2 * self.latent_space + 1) * self.gaussian_space + 2
    
    # inital
    def initial_state(self, batch_size: int = 1, mode: str = 'zero'):
        hidden_space = self.h_space
        if mode == 'zero': 
            return torch.zeros((batch_size, hidden_space), device=self.device), torch.zeros((batch_size, hidden_space), device=self.device)

    def sample(self, mus, sigmas, logpis):
        return [self.cell.sample(mu, sigma, logpi) for mu, sigma, logpi in zip(mus, sigmas, logpis)] 

    def forward(self, inputs):
        """
        The function takes inputs and a role, performs computations using a recurrent neural network
        (RNN) and a mixture density network (MDN), and returns the resulting values.
        
        :param inputs: The `inputs` parameter is a tensor containing the input data. It has a shape of
        `(seq_len, input_size)`, where `seq_len` is the length of the sequence and `input_size` is the
        size of each input element
        :param role: Nothing
        :return: the following variables: mus, sigmas, logpi, rs, ds.
        """        
        (actions, latents), seq_lengths = self._unpack_variable_length_batch(inputs)
        
        if 'hidden_state' in inputs:
            hidden = inputs['hidden_state']
        else:
            hidden  = self.initial_state(batch_size=actions.shape[1])

        state = []
        for action, latent in zip(actions, latents):            
            # Assuming embeds to be the proper input to the LSTM
            mu, sigma, logpi, r, d, hidden = self.cell(action, latent, hidden)
            state.append([mu, sigma, logpi, r, d])
        states = list(zip(*state))
        mus, sigmas, logpis, rs, ds = self._unpack_processed_batch(*[torch.stack(v) for v in states], seq_lengths)

        return mus, sigmas, logpis, rs, ds, hidden

    def _unpack_value(self, value, seq_lengths):
        value = torch.split(value, 1, dim=1)
        value = [v.squeeze(dim=1)[:len] for v, len in zip(value, seq_lengths)]
        return value

    def _unpack_processed_batch(self, mus, sigmas, logpis, rs, ds, seq_lengths):
        mus = self._unpack_value(mus, seq_lengths)
        sigmas = self._unpack_value(sigmas, seq_lengths)
        logpis = self._unpack_value(logpis, seq_lengths)
        rs = self._unpack_value(rs, seq_lengths)
        ds = self._unpack_value(ds, seq_lengths)

        return mus, sigmas, logpis, rs, ds

    def _unpack_variable_length_batch(self, inputs):
        action = nn.utils.rnn.unpack_sequence(inputs['actions'])
        latent = nn.utils.rnn.unpack_sequence(inputs['latent'])

        seq_lengths = [a.shape[0] for a in action]

        action = nn.utils.rnn.pad_sequence(action)
        latent = nn.utils.rnn.pad_sequence(latent)

        return (action, latent), seq_lengths

    def configure_optimizers(self):
        optimizer = RMSprop(self.parameters(), lr=1e-3, alpha=0.9)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def _get_losses(self, batch):    
        batch_latent = batch['images']
        batch_actions = batch['actions']
        batch_terminations = batch['dones']
        batch_rewards = batch['rewards']
        norm_factor = batch_latent[0].shape[-1] + 2 
        
        mus, sigmas, logpi, rs, ds, _ = self.forward({"latent": batch_latent, "actions": batch_actions})
        
        unpacked_batch_terminations = nn.utils.rnn.unpack_sequence(batch_terminations)
        unpacked_batch_reward = nn.utils.rnn.unpack_sequence(batch_rewards)
        # ALL THESE LOSSES LOSSES THE GRADIENT. FIX
        gmm_loss = gaussian_mixture_loss(batch_latent, mus, sigmas, logpi)
        termination_loss = bce_with_logits_list(ds, unpacked_batch_terminations)
        reward_loss = mse_loss_list(rs, unpacked_batch_reward)

        return gmm_loss, termination_loss, reward_loss, norm_factor        


    def set_requires_grad_for_packed_sequence(self, packed_sequence):
        """
        Sets the requires_grad attribute of the tensors within a PackedSequence to True.
        
        Parameters:
        - packed_sequence: An instance of PackedSequence whose data tensor's requires_grad will be set to True.
        
        Returns:
        - A new PackedSequence with the same data as the input, but with requires_grad set to True for its data tensor.
        """
        # Check if the input is indeed a PackedSequence
        if not isinstance(packed_sequence, nn.utils.rnn.PackedSequence):
            raise ValueError("The input must be a PackedSequence.")
        
        # Set requires_grad to True for the data tensor
        data_with_grad = packed_sequence.data.requires_grad_(True)
        
        # Create a new PackedSequence with the modified data tensor
        # but keep the other properties (batch_sizes, sorted_indices, unsorted_indices) the same.
        packed_sequence_with_grad = nn.utils.rnn.PackedSequence(data_with_grad, 
                                                    packed_sequence.batch_sizes,
                                                    packed_sequence.sorted_indices, 
                                                    packed_sequence.unsorted_indices)
        
        return packed_sequence_with_grad

    def training_step(self, batch, batch_idx):
        batch = { key : self.set_requires_grad_for_packed_sequence(value) for key, value in batch.items()}
        gmm_loss, termination_loss, reward_loss, norm_factor = self._get_losses(batch)
        
        loss = (gmm_loss + termination_loss + reward_loss) / (norm_factor + 2)
        
        self.log_dict({
            'gmm_loss': gmm_loss,
            'termination_loss': termination_loss,
            'reward_loss': reward_loss,
            'loss_scaling': norm_factor + 2,
            'train_loss': loss
        })
        
        return loss


    def validation_step(self, batch, batch_idx):
        gmm_loss, termination_loss, reward_loss, norm_factor = self._get_losses(batch)
        
        loss = (gmm_loss + termination_loss + reward_loss) / (norm_factor + 2)
        
        self.log_dict({
            'gmm_loss': gmm_loss,
            'termination_loss': termination_loss,
            'reward_loss': reward_loss,
            'loss_scaling': norm_factor + 2,
            'val_loss': loss
        })
        return loss
        
    def test_step(self, batch, batch_idx):
        gmm_loss, termination_loss, reward_loss, norm_factor = self._get_losses(batch)

        loss = (gmm_loss + termination_loss + reward_loss) / (norm_factor + 2)
        
        self.log_dict({
            'gmm_loss': gmm_loss,
            'termination_loss': termination_loss,
            'reward_loss': reward_loss,
            'loss_scaling': norm_factor + 2,
            'test_loss': loss
        })


    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.encoding is not None:
            batch = self.encoding(batch)
        return batch
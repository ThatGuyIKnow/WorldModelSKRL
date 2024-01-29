from typing import Optional, Tuple, Union
import gym
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from skrl.models.torch import Model, MultivariateGaussianMixin

from Models.VAE import VAE

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
        in_al = torch.cat([action, latent], dim=1)

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




class MDNRNN(Model):
    def __init__(self, 
                 latent_space: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                 action_space: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                 h_space: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                 gaussian_space: Union[int, Tuple[int], gym.Space, gymnasium.Space],                 
                 lookahead: int = 1,
                 temperature: float = 0.2,
                 device: Union[str, torch.device] = 'cuda'):
        Model.__init__(self, latent_space, latent_space + h_space, device)

        self.latent_space = latent_space
        self.gaussian_space = gaussian_space
        self.h_space = h_space
        self.action_space = action_space
        self.temperature = temperature
        
        assert lookahead > 0
        self.lookahead = lookahead

        self.cell = MDRNNCell(self.latent_space, self.action_space.n, self.h_space, self.gaussian_space)

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
        ins_space, hidden_space = self.rnn_space
        if mode == 'zero': 
            return torch.zeros((batch_size, hidden_space), device=self.device), torch.zeros((batch_size, hidden_space), device=self.device)
    
    def act(self, inputs, role):
        """
        The function takes inputs and a role, performs computations using a recurrent neural network
        (RNN) and a mixture density network (MDN), and returns the resulting values.
        
        :param inputs: The `inputs` parameter is a tensor containing the input data. It has a shape of
        `(seq_len, input_size)`, where `seq_len` is the length of the sequence and `input_size` is the
        size of each input element
        :param role: Nothing
        :return: the following variables: mus, sigmas, logpi, rs, ds.
        """        
        if 'hidden_state' in inputs:
            hidden_state = inputs['hidden_state']
        else:
            hidden_state = self.initial_state()

        state = []
        for action, latent in zip(inputs['actions'], inputs['vae']['latent']):
            
            action = torch.unsqueeze(action, dim=0)
            latent = torch.unsqueeze(latent, dim=0)

            mu, sigma, logpi, r, d, hidden_state = self.cell(action, latent, hidden_state)
            state.append([mu, sigma, logpi, r, d])
        mus, sigmas, logpis, rs, ds = list(zip(*state))
        
        return mus, sigmas, logpis, rs, ds, hidden_state

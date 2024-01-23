from typing import Optional, Tuple, Union
import gym
import gymnasium
import torch
import torch.nn as nn

from skrl.models.torch import Model, MultivariateGaussianMixin

from Models.VAE import VAE


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

        self.rnn = nn.LSTM(*self.rnn_space)
        self.mdn =  nn.Linear(self.h_space, self.mdn_space)
    
    @property
    def rnn_space(self):
        """The rnn_space property."""
        print(self.latent_space + self.action_space.n, self.h_space)
        return (self.latent_space + self.action_space.n, self.h_space)
    
    @property
    def mdn_space(self):
        """The mdn_space property."""
        return (2 * self.latent_space + 1) * self.gaussian_space + 2
    
    
    def extract_outs(self, outs, seq_len, action_space):
        stride = self.latent_space * self.gaussian_space
        
        mus    = outs[:, :, :stride]
        mus    = mus.view(seq_len, action_space, self.gaussian_space, self.latent_space)
        
        sigmas = outs[:, :, stride:2*stride]
        sigmas = sigmas.view(seq_len, action_space, self.gaussian_space, self.latent_space)
        
        pi     = outs[:, :, 2 * stride: 2 * stride + self.gaussian_space]
        pi     = pi.view(seq_len, action_space, self.gaussian_space)
        
        rs     = outs[:, :, -2]
        ds     = outs[:, :, -1]
        
        return mus, sigmas, pi, rs, ds
    
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
        seq_len, action_space = inputs['actions'].size(0), inputs['actions'].size(1)
        
        ins = torch.concat([inputs['latent'][2], inputs['actions']], dim=-1)
        # if 'hidden_state' in inputs:
        #     hidden_state = inputs['hidden_state']
        # else:
        #     hidden_state = self.initial_state()
        
        output, (hidden_state, cell_state) = self.rnn(ins)
        gmm_outs = self.mdn(output)
        print(gmm_outs.shape)
        mus, sigmas, pi, rs, ds = self.extract_outs(gmm_outs, seq_len, action_space) 

        sigmas = torch.exp(sigmas)
        logpi = nn.functional.log_softmax(pi, dim=-1)
        
        return mus, sigmas, logpi, rs, ds, hidden_state

from typing import Optional, Tuple, Union
import gym
import gymnasium
import torch

from Models.VAE import VAE
from Models.MDNRNN import MDNRNN
from skrl.models.torch import Model, DeterministicMixin


class WorldModel(Model):
    def __init__(self, 
                 observation_space: Union[int, Tuple[int], gym.Space, gymnasium.Space], 
                 action_space: Union[int, Tuple[int], gym.Space, gymnasium.Space], 
                 latent_space: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                 gaussian_space: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                 img_channels: int,
                 h_space: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                 lookahead: int = 1,
                 temperature: float = 0.2,
                 device: Union[str, torch.device] = 'cuda'):
        Model.__init__(self, observation_space, latent_space + h_space, device)

        self.latent_space = latent_space
        self.h_space = h_space
        
        self.lookahead = lookahead
        
        self.vae = VAE(img_channels, latent_space, observation_space.shape)
        self.mdnrnn = MDNRNN(latent_space, action_space, h_space, gaussian_space, lookahead, temperature, device)


    def act(self, inputs, role):
        recon_x, mu, logsigma, z = self.vae(inputs['states'])
        inputs['vae'] = {
            'recon_x' : recon_x,
            'mu' : mu,
            'logsigma' : logsigma,
            'latent' : z
        }
        mus, sigmas, logpis, rs, ds, (hidden_state, _) = self.mdnrnn(inputs)
        return z,  hidden_state
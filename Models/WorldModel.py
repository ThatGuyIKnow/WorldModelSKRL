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
                 latent_space: Union[int, Tuple[int], gym.Space, gymnasium.Space] = None,
                 gaussian_space: Union[int, Tuple[int], gym.Space, gymnasium.Space] = None,
                 img_channels: int = None,
                 h_space: Union[int, Tuple[int], gym.Space, gymnasium.Space] = None,
                 lookahead: int = 1,
                 temperature: float = 0.2,
                 device: Union[str, torch.device] = 'cuda',
                 vae: Model = None,
                 mdnrnn: Model = None):
        super().__init__(observation_space = observation_space, action_space = action_space)
        self.latent_space = latent_space
        self.h_space = h_space

        self.lookahead = lookahead
        if vae is None:
            self.vae = VAE(img_channels, latent_space, observation_space.shape)
        else:
            self.vae = vae

        if mdnrnn is None:
            self.mdnrnn = MDNRNN(latent_space, action_space, h_space, gaussian_space, lookahead, temperature, device)
        else:
            self.mdnrnn = mdnrnn

        self.vae.to(device)
        self.mdnrnn.to(device)

    def act(self, inputs, role):
        recon_x, mu, logsigma, z = self.vae(inputs['states'])
        inputs['latent'] = z
        mus, sigmas, logpis, rs, ds, (hidden_state, _) = self.mdnrnn(inputs)
        return z,  hidden_state

    def initial_state(self):
        return self.mdnrnn.initial_state()
    
    def to_latent(self, x):
        _, _, z = self.vae.encoder(x)
        return z
    
    def step(self, action, latent, hidden_state):
        _, _, _, _, _, hidden = self.mdnrnn.cell(action, latent, hidden_state)
        return hidden
from typing import Optional, Tuple, Union
import gym
import gymnasium
import numpy as np
import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

from Models.WorldModel import WorldModel


# [start-config-dict-torch]
WORLD_MODEL_DEFAULT_CONFIG = {
    "latent_space": 32,      # size of the latent space in the VAE
    "h_space": 64,           # size of the saved h when predicting the future
    "gaussian_space": 64,    # size of the saved h when predicting the future
    "lookahead": 1,          # the amount of lookahead when predicting the future
    "temperature": 0.2,      # the MDN-RNN temparure for the mixture density network
    "device": 'cpu',        # allocated device
    "img_channels": 1        # img_channels (rgb: 3 or greyscale: 1)
}
# [end-config-dict-torch]

class WorldModelController(DeterministicMixin, Model):
    def __init__(self, 
                 observation_space: Union[int, Tuple[int], gym.Space, gymnasium.Space], 
                 action_space: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                 device: Union[str, torch.device] = 'cuda', 
                 clip_actions: bool = False,
                 cfg: Optional[dict] = None):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.cfg = WORLD_MODEL_DEFAULT_CONFIG
        if cfg is not None:
            self.cfg.update(cfg)
            
        
        self.world_model = WorldModel(observation_space, action_space, **self.cfg)
        
        # shared layers/network
        self.net = nn.Linear(self.cfg['latent_space'] + self.cfg['h_space'], action_space.n)
        
    # forward the input to compute model output according to the specified role
    def compute(self, inputs, role):
        z, h = self.world_model(inputs)
        x = torch.concat([z, h], dim=-1)
        return self.net(x), {}
        


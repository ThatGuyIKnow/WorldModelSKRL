import collections
import copy
from functools import partial
import math
from typing import Dict, Optional, Tuple, Union
import gym
import gymnasium
from skrl.agents.torch import Agent
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.memories.torch import Memory
from skrl.models.torch import Model
import torch
from torch import optim
from torch.nn import functional as F
from Utils.EarlyStopping import EarlyStopping
from Utils.ReduceLROnPlateau import ReduceLROnPlateau
from Utils.ModelCheckpoint import ModelCheckpoint

from Losses.GMMLoss import gaussian_mixture_loss


def preprocess(shape):
    return lambda x, train=False: x.view(-1, *shape) / 255

def to_latent(encoder, mdnrnn, shape, batch_size):
    state_preprocessor = preprocess(shape)
    def func(x, hidden_state = None):
        x = state_preprocessor(x)

        _, _, z = encoder(x)
        if hidden_state is None:
            hidden_state = mdnrnn.initial_state(batch_size)

        for action, latent in zip(inputs['actions'], inputs['latent']):

            action = torch.unsqueeze(action, dim=0)
            latent = torch.unsqueeze(latent, dim=0)

            mu, sigma, logpi, r, d, hidden_state = self.cell(action, latent, hidden_state)
        _, _, z = encoder(x)
        _, _, _, _, _, h = mdnrnn.cell(z, h)
        return torch.concat([z, h], dim=1)
    return func

# [start-config-dict-torch]
WorldModelDQNDefaultConfig = {
    'learning_starts': 2e6,
    'batch_size': 32,
    'img_channels': 1,
    'vae': {
        'learning_starts': 1000,
        'learning_stops': 1e6,
        'learning_interval': 1,
        'learning_rate': 1e-4,
        'learning_rate_eps': 0.01,
        'optimizer': optim.Adam,
        'lr_schedule': partial(ReduceLROnPlateau, **{'mode': 'min', 'factor': 0.5, 'patience': 5}),
        'track_recon_interval': 200,
        'checkpoint_interval': 10000
        },
    'mdnrnn': {
        'learning_starts': 1100,
        #'learning_starts': 1e6,
        'learning_stops': 2e6,
        'learning_interval': 1,
        'learning_rate': 1e-3,
        'learning_rate_eps': 0.01,
        'optimizer': partial(torch.optim.RMSprop, **{'lr':1e-3, 'alpha':.9}),
        'lr_schedule': partial(ReduceLROnPlateau, **{'mode': 'min', 'factor': 0.5, 'patience': 5}),
        'track_recon_interval': 2000,
        'checkpoint_interval': 10000
        },
}
WorldModelDQNDefaultConfig.update(DQN_DEFAULT_CONFIG)
# [end-config-dict-torch]

class WorldModelDQN(DQN):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:

        _cfg = copy.deepcopy(WorldModelDQNDefaultConfig)
        _cfg.update(cfg if cfg is not None else {})
        _cfg['vae']['state_preprocessor'] = preprocess
        _cfg['vae']['state_preprocessor_kwargs'] = {'shape': (_cfg['img_channels'], *observation_space.shape)}

        _cfg['mdnrnn']['state_preprocessor'] = preprocess
        _cfg['mdnrnn']['state_preprocessor_kwargs'] = _cfg['vae']['state_preprocessor_kwargs']

        _cfg['state_preprocessor'] = to_latent
        _cfg['state_preprocessor_kwargs'] = {'mdnrnn':  models['world_model'].world_model.mdnrnn,
                                             'encoder': models['world_model'].world_model.vae.encoder,
                                              **_cfg['vae']['state_preprocessor_kwargs'] }
        super().__init__(models, memory, observation_space, action_space, device, _cfg)
        self._cfg = _cfg

        self._learning_starts = self._cfg['learning_starts']

        self.network =
        self.previous_action = torch.ones(self.action_space.n) / self.action_space.n

        self.network.world_model.vae.to(device)
        self.network.world_model.mdnrnn.to(device)

        self._tracking_media = collections.defaultdict(None)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        states = self._state_preprocessor(states)
        onehot_previous_action = torch.squeeze(F.one_hot(self.previous_action.long(), num_classes=self.action_space.n), dim=1)

        if not self._exploration_timesteps:
            self.previous_action = torch.argmax(self.network.act({"states": states, 'actions': onehot_previous_action}, role="controller")[0], dim=1, keepdim=True)
            return self.previous_action, None, None

        # sample random actions
        actions = self.network.random_act({"states": states}, role="controller")[0]
        if timestep < self._random_timesteps:
            self.previous_action = actions
            return actions, None, None

        # sample actions with epsilon-greedy policy
        epsilon = self._exploration_final_epsilon + (self._exploration_initial_epsilon - self._exploration_final_epsilon) \
                * math.exp(-1.0 * timestep / self._exploration_timesteps)

        indexes = (torch.rand(states.shape[0], device=self.device) >= epsilon).nonzero().view(-1)
        if indexes.numel():
            actions[indexes] = torch.argmax(self.network.act({"states": states[indexes], 'actions': onehot_previous_action}, role="controller")[0], dim=1, keepdim=True)
        self.previous_action = actions

        # record epsilon
        self.track_data("Exploration / Exploration epsilon", epsilon)

        return actions, None, None



    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """


        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)
        self.write_tracking_media(timestep)

    def track_media(self, tag, data, type='image'):
        self._tracking_media[tag] = {'type': type, 'data': data}

    def write_tracking_media(self, timestep : int):
        for k, v in self._tracking_media.items():
            if v['type'] == 'video':
                self.writer.add_video(k, v['data'], timestep)
            elif v['type'] == 'image':
                self.writer.add_image(k, v['data'], timestep)
            elif v['type'] == 'images':
                self.writer.add_images(k, v['data'], timestep)
        self._tracking_media.clear()

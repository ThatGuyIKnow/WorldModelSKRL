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

def to_latent(encoder, mdnrnn, shape):
    state_preprocessor = preprocess(shape)
    def func(x):
        x = state_preprocessor(x)
        _, _, z = encoder(x)
        _, _, _, _, _, h = mdnrnn.act(z, role='preprocessor')
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
        _cfg['state_preprocessor_kwargs'] = {'mdnrnn':  models['world_model_controller'].world_model.mdnrnn, 
                                             'encoder': models['world_model_controller'].world_model.vae.encoder,
                                              **_cfg['vae']['state_preprocessor_kwargs'] }
        super().__init__(models, memory, observation_space, action_space, device, _cfg)
        self._cfg = _cfg
        
        self._learning_starts = self._cfg['learning_starts']
        
        self._vae_learning_starts = self._cfg['vae']['learning_starts']
        self._vae_learning_stops = self._cfg['vae']['learning_stops']
        self._vae_learning_interval = self._cfg['vae']['learning_interval']
        self._vae_learning_rate = self._cfg['vae']['learning_rate']
        self._vae_learning_rate_eps = self._cfg['vae']['learning_rate_eps']
        self._vae_lr_schedule = self._cfg['vae']['lr_schedule']
        self._vae_track_recon_interval = self._cfg['vae']['track_recon_interval']
        self._vae_checkpoint_interval = self._cfg['vae']['checkpoint_interval']
        self._vae_state_preprocessor = self._cfg['vae']['state_preprocessor'](**self._cfg['vae']['state_preprocessor_kwargs'])

        self._mdnrnn_learning_starts = self._cfg['mdnrnn']['learning_starts']
        self._mdnrnn_learning_stops = self._cfg['mdnrnn']['learning_stops']
        self._mdnrnn_learning_interval = self._cfg['mdnrnn']['learning_interval']
        self._mdnrnn_learning_rate = self._cfg['mdnrnn']['learning_rate']
        self._mdnrnn_learning_rate_eps = self._cfg['mdnrnn']['learning_rate_eps']
        self._mdnrnn_lr_schedule = self._cfg['mdnrnn']['lr_schedule']
        self._mdnrnn_track_recon_interval = self._cfg['mdnrnn']['track_recon_interval']
        self._mdnrnn_checkpoint_interval = self._cfg['mdnrnn']['checkpoint_interval']
        self._mdnrnn_state_preprocessor = self._cfg['mdnrnn']['state_preprocessor'](**self._cfg['mdnrnn']['state_preprocessor_kwargs'])

        self.network = self.models['world_model_controller']
        self.previous_action = torch.ones(self.action_space.n) / self.action_space.n
        
        self.network.world_model.vae.to(device)
        self.network.world_model.mdnrnn.to(device)

        # OPTIMIZERS 
        self._vae_optimizer = self._cfg['vae']['optimizer'](self.network.world_model.vae.parameters())
        self._vae_lr_scheduler = self._cfg['vae']['lr_schedule'](optimizer=self._vae_optimizer)
        self._vae_stopping_criteria = EarlyStopping('min', patience=30)
        self._vae_checkpoint  = ModelCheckpoint(self.network.world_model.vae, self._vae_checkpoint_interval, self.experiment_dir, 'VAE')


        self._mdnrnn_optimizer    = self._cfg['mdnrnn']['optimizer'](self.network.world_model.mdnrnn.parameters())
        self._mdnrnn_lr_scheduler = self._cfg['mdnrnn']['lr_schedule'](optimizer=self._mdnrnn_optimizer)
        self._mdnrnn_stopping_criteria = EarlyStopping('min', patience=30)
        self._mdnrnn_checkpoint  = ModelCheckpoint(self.network.world_model.mdnrnn, self._mdnrnn_checkpoint_interval, self.experiment_dir, 'MDNRNN')

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
        if timestep >= self._vae_learning_starts and timestep < self._vae_learning_stops and (timestep % self._vae_learning_interval) == 0:
            self._update_vae(timestep)
            
        if timestep >= self._mdnrnn_learning_starts and timestep < self._mdnrnn_learning_stops and not timestep % self._mdnrnn_learning_interval:
            self._update_mdnrnn(timestep)
            
        if timestep >= self._learning_starts and timestep % self._update_interval == 0:
            self._update_controller(timestep, timesteps)

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)
        self.write_tracking_media(timestep)
        
    def _update_vae(self, timestep : int) -> bool:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # Prevent overfitting, early stopping signal
        #if self._vae_stopping_criteria.stop:
        #    return True
        
        # get VAE
        vae = self.network.world_model.vae
        
        # sample a batch from memory
        sampled_states, _, _, _, _ = \
            self.memory.sample(names=self.tensors_names, batch_size=self._batch_size)[0]

        sampled_states = self._vae_state_preprocessor(sampled_states, train=True)


        # compute target values
        recon_x, mu, logsigma, _ = vae(sampled_states)

        reconstruction_loss = F.mse_loss(recon_x, sampled_states, reduction='sum')
        #reg_loss = 0.5 * (mu ** 2 + torch.exp(logsigma) - logsigma - 1).sum(dim=1).mean()
        reg_loss = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        loss = reconstruction_loss + reg_loss
        
        # optimize Q-network
        self._vae_optimizer.zero_grad()
        loss.backward()
        self._vae_optimizer.step()
        
        # update learning rate
        if self._vae_lr_scheduler is not None:
            self._vae_lr_scheduler.step(loss)

        if self._vae_stopping_criteria is not None:
            self._vae_stopping_criteria.step(loss)

        # record data
        self.track_data("VAE / Reconstruction loss", reconstruction_loss.item())
        self.track_data("VAE / Reg. loss", reg_loss.item())
        self.track_data("VAE / Loss", loss.item())

        if self._vae_lr_scheduler is not None:
            self.track_data("VAE / Learning rate", self._vae_lr_scheduler.get_last_lr()[0])
        
        if timestep % self._vae_track_recon_interval == 0:
            next_states = sampled_states.view(-1, 1, *self.observation_space.shape)
            data = torch.stack([next_states[0], recon_x[0]])
            self.track_media('VAE / Reconstruction', data, type='images')
        
        self._vae_checkpoint.save_checkpoint(timestep, loss)

        return False
                
    def _update_mdnrnn(self, timestep : int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # get Mixture Density Network RNN
        encoder = self.network.world_model.vae.encoder
        decoder = self.network.world_model.vae.decoder
        mdnrnn = self.network.world_model.mdnrnn

        # sample a batch from memory
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_terminated = \
            self.memory.sample(names=self.tensors_names, batch_size=1, sequence_length=self._batch_size)[0]

        sampled_states = torch.swapaxes(sampled_states, 0, 1)
        sampled_next_states = torch.swapaxes(sampled_next_states, 0, 1)
        
        sampled_states = self._mdnrnn_state_preprocessor(sampled_states, train=True)
        sampled_next_states = self._mdnrnn_state_preprocessor(sampled_next_states, train=True)

        _, _, z = encoder(sampled_states)
        _, _, next_z = encoder(sampled_next_states)
        
        sampled_actions = F.one_hot(sampled_actions[:,0], self.action_space.n)
        
        # compute target values
        with torch.no_grad():
            mus, sigmas, logpi, rs, ds, _ = mdnrnn.act({"latent": z, "actions": sampled_actions}, role="mdnrnn")
            
        gmm_loss = gaussian_mixture_loss(next_z, mus, sigmas, logpi)    
        reward_loss = F.mse_loss(rs, sampled_rewards)
        terminated_loss = F.binary_cross_entropy_with_logits(ds, sampled_terminated.float())
        
        loss = gmm_loss + reward_loss + terminated_loss
        
        # optimize MDNRNN
        self._mdnrnn_optimizer.zero_grad()
        loss.backward()
        self._mdnrnn_optimizer.step()
        
        # update learning rate
        if self._mdnrnn_lr_scheduler:
            self._mdnrnn_lr_scheduler.step(loss)

        # record data
        self.track_data("MDNRNN / GMM loss", gmm_loss.item())
        self.track_data("MDNRNN / Reward loss", reward_loss.item())
        self.track_data("MDNRNN / Terminated loss", terminated_loss.item())
        self.track_data("MDNRNN / Loss", loss.item())

        if self._mdnrnn_lr_scheduler:
            self.track_data("MDNRNN / Learning rate", self._mdnrnn_lr_scheduler.get_last_lr()[0])
                
         
        if timestep % self._mdnrnn_track_recon_interval == 0:
            sampled_states = sampled_states.view(-1, 1, *self.observation_space.shape)
            sampled_next_states = sampled_next_states.view(-1, 1, *self.observation_space.shape)
            recon_next_x = decoder(next_z)
            data = torch.stack([sampled_states[0], sampled_next_states[0], recon_next_x[0]])
            self.track_media('MDNRNN / Reconstruction (x_t, x_{t+1}, x\'_{t+1})', data, type='images')
        
        self._mdnrnn_checkpoint.save_checkpoint(timestep, loss)

        return False
    
    def _update_controller(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # sample a batch from memory
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = \
            self.memory.sample(names=self.tensors_names, batch_size=self._batch_size)[0]

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            sampled_states = self._state_preprocessor(sampled_states, train=True)
            sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

            # compute target values
            with torch.no_grad():
                next_q_values, _, _ = self.target_q_network.act({"states": sampled_next_states}, role="target_q_network")

                target_q_values = torch.max(next_q_values, dim=-1, keepdim=True)[0]
                target_values = sampled_rewards + self._discount_factor * sampled_dones.logical_not() * target_q_values

            # compute Q-network loss
            q_values = torch.gather(self.q_network.act({"states": sampled_states}, role="q_network")[0],
                                    dim=1, index=sampled_actions.long())

            q_network_loss = F.mse_loss(q_values, target_values)

            # optimize Q-network
            self.optimizer.zero_grad()
            q_network_loss.backward()
            self.optimizer.step()

            # update target network
            if not timestep % self._target_update_interval:
                self.target_q_network.update_parameters(self.q_network, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

            # record data
            self.track_data("Loss / Q-network loss", q_network_loss.item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

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
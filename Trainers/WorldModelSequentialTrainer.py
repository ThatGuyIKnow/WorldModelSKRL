from typing import List, Optional, Union

import copy
import tqdm
import numpy as np

import torch
import torch.nn.functional as f

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import Model


SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "random_timesteps": 10000, 
}


class WorldModelSequentialTrainer(SequentialTrainer):
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent]],
                 world_model: Model,
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None,
                 device = 'cuda') -> None:
        """Sequential trainer

        Train agents sequentially (i.e., one after the other in each interaction with the environment)

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``).
                    See SEQUENTIAL_TRAINER_DEFAULT_CONFIG for default values
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        self.world_model = world_model
        self.action_space = env.action_space
        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)
        
        self.device = device
    
    def rand_argmax(self, tens):
        _, max_inds = torch.where(tens == tens.max())
        return np.random.choice(max_inds)
    
    def train(self) -> None:
        """Train the agents sequentially

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")


        # reset env
        states, infos = self.env.reset()
        states = states.to(self.device)
        latent = self.world_model.to_latent(torch.Tensor(states))
        h_state = self.world_model.initial_state()
        enc_states = torch.concat([latent, h_state[0]], dim=-1)

        actions = self.agents.act(enc_states, timestep=0, timesteps=self.timesteps)[0] 

        h_state = self.world_model.step(actions, latent, h_state)
                
        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):

            # pre-interaction
            if self.num_simultaneous_agents > 1:
                for agent in self.agents:
                    agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            else:
                self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # compute actions
            with torch.no_grad():
                # step the environments

                selected_action = actions.item()
                next_states, rewards, terminated, truncated, infos = self.env.step(selected_action)
                rewards = torch.Tensor([rewards])
                terminated = torch.Tensor([terminated])
                truncated = torch.Tensor([truncated])
                next_latent = self.world_model.to_latent(torch.Tensor(next_states))

                next_h_state = self.world_model.step(actions, next_latent, h_state)
                next_enc_states = torch.concat([next_latent, next_h_state[0]], dim=-1)
                

                actions = self.agents.act(enc_states, timestep=0, timesteps=self.timesteps)[0] 

                
                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(states=enc_states,
                                        actions=selected_action,
                                        rewards=rewards,
                                        next_states=next_enc_states,
                                        terminated=terminated,
                                        truncated=truncated,
                                        infos=infos,
                                        timestep=timestep,
                                        timesteps=self.timesteps)

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            with torch.no_grad():
                if terminated.any() or truncated.any():
                    states, infos = self.env.reset()
                    latent = self.world_model.to_latent(torch.Tensor(states))
                    h_state = self.world_model.initial_state()
                    enc_states = torch.concat([latent, h_state[0]], dim=-1)
                    
                    actions = self.agents.act(enc_states, timestep=0, timesteps=self.timesteps)[0] 

                    
                    h_state = self.world_model.step(f.one_hot(actions, self.action_space.n)[0], latent, h_state)
                
                else:
                    latent = next_latent
                    h_state = next_h_state
                    enc_states = next_enc_states

    def eval(self) -> None:
        """Train the agents sequentially

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")


        # reset env
        states, infos = self.env.reset()
        latent = self.world_model.to_latent(torch.Tensor(states))
        h_state = self.world_model.initial_state()
        enc_states = torch.concat([latent, h_state[0]], dim=-1)

        actions = self.agents.act(enc_states, timestep=0, timesteps=self.timesteps)[0] 

        h_state = self.world_model.step(actions, latent, h_state)
                
        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):

            # pre-interaction
            if self.num_simultaneous_agents > 1:
                for agent in self.agents:
                    agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            else:
                self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # compute actions
            with torch.no_grad():
                # step the environments

                selected_action = actions.item()
                next_states, rewards, terminated, truncated, infos = self.env.step(selected_action)
                rewards = torch.Tensor([rewards])
                terminated = torch.Tensor([terminated])
                truncated = torch.Tensor([truncated])
                next_latent = self.world_model.to_latent(torch.Tensor(next_states))

                next_h_state = self.world_model.step(actions, next_latent, h_state)
                next_enc_states = torch.concat([next_latent, next_h_state[0]], dim=-1)
                

                actions = self.agents.act(enc_states, timestep=0, timesteps=self.timesteps)[0] 

                
                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(states=enc_states,
                                        actions=selected_action,
                                        rewards=rewards,
                                        next_states=next_enc_states,
                                        terminated=terminated,
                                        truncated=truncated,
                                        infos=infos,
                                        timestep=timestep,
                                        timesteps=self.timesteps)

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            with torch.no_grad():
                if terminated.any() or truncated.any():
                    states, infos = self.env.reset()
                    latent = self.world_model.to_latent(torch.Tensor(states))
                    h_state = self.world_model.initial_state()
                    enc_states = torch.concat([latent, h_state[0]], dim=-1)
                    
                    actions = self.agents.act(enc_states, timestep=0, timesteps=self.timesteps)[0] 

                    
                    h_state = self.world_model.step(f.one_hot(actions, self.action_space.n)[0], latent, h_state)
                
                else:
                    latent = next_latent
                    h_state = next_h_state
                    enc_states = next_enc_states

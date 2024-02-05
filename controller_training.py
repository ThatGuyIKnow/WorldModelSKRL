# import the agent and its default configuration
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG

import torch
import torch.nn as nn

from skrl.models.torch import Model, DeterministicMixin


# define the model
class MLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# instantiate the model (assumes there is a wrapped environment: env)
critic = MLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=False)
class MLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Linear(self.observation_space.shape[0], self.action_space.n)

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# instantiate the model (assumes there is a wrapped environment: env)
controller = MLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=False)

# instantiate the agent's models
models = {}
models["q_network"] = controller
# models["target_q_network"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = DQN_DEFAULT_CONFIG.copy()

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = DQN(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)

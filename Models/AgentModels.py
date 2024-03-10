
import numpy as np
from skrl.models.torch import Model, DeterministicMixin, GaussianMixin
import torch
import torch.nn as nn

# define the model
class CriticMLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.observation_space.shape[0], 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 1)).to(device)

    def compute(self, inputs, role):
        return self.net(inputs["states"]).to(self.device), {}



class ActorMLP(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.observation_space.shape[0], 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, self.num_actions)).to(self.device)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def to_action(self, input):
        return (input +  torch.tensor( [0, 1, 1] ).to(self.device)) / torch.Tensor([1, 2, 2]).to(self.device)
    
    def compute(self, inputs, role):
        return self.to_action(torch.tanh(self.net(inputs["states"]))), self.log_std_parameter, {}
    
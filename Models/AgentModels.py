
from skrl.models.torch import Model, DeterministicMixin, GaussianMixin
import torch
import torch.nn as nn

# define the model
class CriticMLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.observation_space.shape[0], 400),
                                 nn.Tanh(),
                                 nn.Linear(400, 300),
                                 nn.Tanh(),
                                 nn.Linear(300, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}



class ActorMLP(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.observation_space.shape[0], 400),
                                 nn.Tanh(),
                                 nn.Linear(400, 300),
                                 nn.Tanh(),
                                 nn.Linear(300, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}
    
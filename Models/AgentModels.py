
from skrl.models.torch import Model, DeterministicMixin, CategoricalMixin
import torch
import torch.nn as nn

# define the model
class CriticMLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.observation_space.shape[0], 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}



class ActorMLP(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.net = nn.Sequential(nn.Linear(self.observation_space.shape[0], 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
    
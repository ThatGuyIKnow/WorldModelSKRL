
from math import sqrt
import torch
import torch.nn as nn
from Models.utils.LambdaLayer import LambdaLayer

# import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, Model


class Encoder(Model):
    def __init__(self, observation_space, img_channels, latent_space, device):
        Model.__init__(self, observation_space, latent_space, device)
        
        self.latent_space = latent_space
        
        self.encoder = nn.Sequential(
                nn.Conv2d(img_channels, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
        )
        self.logsigma = nn.Sequential(
                nn.Linear(3136, self.latent_space),
                nn.ReLU(),
            )
        self.mu = nn.Sequential(
                nn.Linear(3136, self.latent_space),
                nn.ReLU(),
            )

    def act(self, inputs, role):
        # permute (samples, width * height * channels) -> (batch, channels, width, height)
        x = self.encoder(inputs["states"].view(-1, 1, *self.observation_space.shape))
        mu = self.mu(x)
        
        logsig = self.logsigma(x)
        sig = logsig.exp()
        
        rand = torch.randn_like(sig)
        
        z = rand * sig + mu
        
        return mu, sig, z

class Decoder(Model):
    def __init__(self, latent_space, img_channels, device):
        Model.__init__(self, latent_space, img_channels, device)
        
        new_dim = int(sqrt(3136/64))
        self.deconv = nn.Sequential(
                nn.Linear(latent_space, 3136),
                nn.ReLU(),
                LambdaLayer(lambda x: x.view(-1, 64, new_dim, new_dim)),
                nn.ConvTranspose2d(64, 64, 8, stride=4),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, img_channels, 3, stride=1),
                nn.Sigmoid()
        )

    def act(self, inputs, role):
        return self.deconv(inputs)


class VAE(Model):
    def __init__(self, observation_space, latent_space, img_channels, device):
        Model.__init__(self, latent_space, img_channels, device)
        
        self.encoder = Encoder(observation_space, img_channels, latent_space, device)
        self.decoder = Decoder(latent_space, img_channels, device)

    def act(self, inputs, role):
        mu, sig, z = self.encoder(inputs)
        
        y = self.decoder(z)
        
        return mu, sig, z, y

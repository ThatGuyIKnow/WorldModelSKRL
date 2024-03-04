
from functools import partial
import gymnasium as gym
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class Crop(object):
    def __init__(self, top=0, bottom=-1, left=0, right=-1):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right 

    def __call__(self, sample):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        return sample[:, self.top:self.bottom, self.left:self.right]
    

class Interpolate(object):
    def __init__(self, min_v, max_v, target_min_v, target_max_v) -> None:
        self.interp = partial(np.interp, xp=[min_v, max_v], fp=[target_min_v, target_max_v])

    def __call__(self, sample):
        return self.interp( sample )
    
class EnsureType(object):
    def __init__(self, type) -> None:
        self.type = type
    def __call__(self, sample):
        return sample.to(self.type)

class TransformWrapper(gym.ObservationWrapper):
    transform = transforms.Compose([
            Interpolate(0, 255, -1, 1),
            transforms.ToTensor(), 
            transforms.Grayscale(), 
            Crop(bottom=-50),
            transforms.Resize((64, 64), antialias=True),
            EnsureType(torch.float)
        ])
    def __init__(self, env):
        super().__init__(env)
        
        
    def observation(self, obs):
        obs = Image.fromarray(obs)
        return self.transform(obs)
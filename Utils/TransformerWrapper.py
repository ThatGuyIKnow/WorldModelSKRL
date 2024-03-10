
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
        self.min_v = min_v
        self.factor = max_v - min_v

        self.scale_factor = target_max_v - target_min_v
        self.target_min_v = target_min_v

    def __call__(self, sample):
        p = (sample - self.min_v) / self.factor
        return self.scale_factor * p + self.target_min_v
    
class EnsureType(object):
    def __init__(self, type) -> None:
        self.type = type
    def __call__(self, sample):
        return sample.to(self.type)


class TransformWrapper(gym.ObservationWrapper):
    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Grayscale(), 
            Crop(bottom=-50),
            transforms.Resize((64, 64), antialias=True),
        ])
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low = np.zeros((64, 64)), high = np.ones((64, 64)), dtype=np.float32)
        
        
    def observation(self, obs):
        obs = Image.fromarray(obs)
        return self.transform(obs)
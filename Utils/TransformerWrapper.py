
import gymnasium as gym
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
    
class TransformWrapper(gym.ObservationWrapper):
    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Grayscale(), 
            transforms.Normalize((0.5,), (0.5,)), 
            Crop(bottom=-50),
            transforms.Resize((64, 64), antialias=True),
        ])
    def __init__(self, env):
        super().__init__(env)
        
        
    def observation(self, obs):
        obs = Image.fromarray(obs)
        return self.transform(obs)
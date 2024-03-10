
import gymnasium as gym

import torch
from torch.utils.data import Dataset

class EpisodeDataset(Dataset):
    def __init__(self, env_fn, num_envs=4, max_steps=1000, episodes_per_epoch=2, device='cpu'):
        self.env_fn = env_fn
        self.num_envs = num_envs
        self.device = device
        self.max_steps = max_steps
        self.episodes_per_epoch = episodes_per_epoch
        self.envs = gym.vector.AsyncVectorEnv([lambda: env_fn() for _ in range(num_envs)])
        self.device = device

    def __getitem__(self, idx):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        obs, _ = self.envs.reset()
        
        for _ in range(self.max_steps):
            action = self.envs.action_space.sample()
            #self.envs.step_async(action)
            next_obs, rs, ds, truncated, infos = self.envs.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(rs)
            next_states.append(next_obs)
            dones.append(ds)
            
            obs = next_obs
            
            if any(dones[-1]) or any(truncated):
                break
        
        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        
        return states, actions, rewards, dones, next_states

    def __len__(self):
        return self.episodes_per_epoch

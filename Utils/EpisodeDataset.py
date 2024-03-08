import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import ast
import re

from tqdm import tqdm


class EpisodeDataset(Dataset):
    def __init__(self, csv_file, transform=None, encoder = None, action_space = None, seq_length=16, device='cpu', limit_size=None):
        """
        Initializes the dataset.
        
        Parameters:
        csv_file (str): Path to the CSV file containing episode data.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        print('RETREIVING DATASET')
        if limit_size is None:
            self.episode_data = pd.read_csv(csv_file)
        else:
            self.episode_data = pd.read_csv(csv_file)[:limit_size]
            
        self.episode_data['Action'] = self.episode_data['Action'].apply(lambda x: ast.literal_eval(re.sub(r"\s+", ',', x)))
        self.encoder = encoder.to(device)

        self.image_paths = pd.concat([self.episode_data['ImagePath'], self.episode_data['NextImagePath']]).unique()
        self.image_idx = pd.Series(range(len(self.image_paths)), index=self.image_paths)
        
        images = torch.stack([transform(Image.open(path)) for path in tqdm(self.image_paths)])
        enc_images = []
        for batch in tqdm(torch.split(images, 64)):
            _, _, z = encoder(batch.to(device))
            enc_images.append(z)
        self.images = torch.concat(enc_images)

        self.actions = self.episode_data['Action'].apply(torch.tensor, by_row=False).to(device)
        self.rewards = torch.tensor(self.episode_data['Reward'], device=device)
        self.dones = torch.tensor(self.episode_data['Done'], device=device)

        if action_space is None:
            self.action_space = len(self.episode_data['Action'].unique())
        else:
            self.action_space = action_space
        self.seq_length = seq_length

        # Drop last SEQ length to ensure equal length training data
        allowed = self.episode_data.groupby(['Episode', 'Worker'], as_index=False).apply(lambda x: x.iloc[:-seq_length])
        self.allowed_idx = [idx[1] for idx in allowed.index]

        self.device = device

    def __len__(self):
        return len(self.allowed_idx) // self.seq_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        min_idx = self.allowed_idx[self.seq_length * idx]
        max_idx = min_idx + self.seq_length

        episode_data = self.episode_data[min_idx:max_idx]
        images = []
        next_images = []
        actions = []
        rewards = []
        dones = []

        for i, row in episode_data.iterrows():
            path = self.image_idx[row['ImagePath']]
            next_path = self.image_idx[row['NextImagePath']]
            image = self.images[path]
            next_image = self.images[next_path]           
            images.append(image)
            next_images.append(next_image)

        images = torch.stack(images)
        next_images = torch.stack(next_images)
        actions = self.actions[min_idx:max_idx]
        rewards = self.rewards[min_idx:max_idx]
        dones = self.dones[min_idx:max_idx]

        
        return images, actions, rewards, dones, next_images


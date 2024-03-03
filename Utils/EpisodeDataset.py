import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import ast
import re


class EpisodeDataset(Dataset):
    def __init__(self, csv_file, transform=None, encoding = None, action_space = None, seq_length=16):
        """
        Initializes the dataset.
        
        Parameters:
        csv_file (str): Path to the CSV file containing episode data.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.episode_data = pd.read_csv(csv_file)
        self.transform = transform
        if action_space is None:
            self.action_space = len(self.episode_data['Action'].unique())
        else:
            self.action_space = action_space
        self.encoding = encoding
        self.seq_length = seq_length

        # Drop last SEQ length to ensure equal length training data
        allowed = self.episode_data.groupby(['Episode', 'Worker'], as_index=False).apply(lambda x: x.iloc[:-seq_length])
        self.allowed_idx = [idx[1] for idx in allowed.index]

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

        for _, row in episode_data.iterrows():
            image = Image.open(row['ImagePath'])
            next_image = Image.open(row['NextImagePath'])
            if self.transform:
                image = self.transform(image)
                next_image = self.transform(next_image)
            if self.encoding is not None:
                _, _, image = self.encoding(image)
                _, _, next_image = self.encoding(next_image)
                image = image.squeeze(dim=0).clone().detach()
                next_image = next_image.squeeze(dim=0).clone().detach()
            action = row['Action']
            if isinstance(action, str):
                action = ast.literal_eval(re.sub(r"\s+", ',', action))
            
            images.append(image)
            next_images.append(next_image)
            actions.append(action)
            rewards.append(row['Reward'])
            dones.append(row['Done'])

        actions = torch.tensor(actions, dtype=torch.long)

        images = torch.stack(images)
        next_images = torch.stack(next_images)
        if actions.shape[-1] == 1:
            actions = F.one_hot(actions, self.action_space).float() 
        else:
            actions = actions.float()
        rewards =  torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        sample = {'images': images, 
                  'next_images': next_images, 
                  'actions': actions, 
                  'rewards': rewards,
                  'dones': dones}
        
        return sample


    def collate_key(value, seq_lengths, order_indicies):
        value = nn.utils.rnn.pad_sequence(value, batch_first=True)
        value = value[order_indicies].clone().detach()
        
        value = nn.utils.rnn.pack_padded_sequence(value, seq_lengths, batch_first=True)

        return value

    def collate_fn(batch):
        seq_lengths = torch.tensor([seq['images'].shape[0] for seq in batch])
        order_indicies = torch.argsort(seq_lengths, descending=True)
        
        #Order sequence
        seq_lengths = seq_lengths[order_indicies]


        keys = ['images', 'next_images', 'actions', 'rewards', 'dones']
        batch = {key : [sample[key] for sample in batch] for key in keys}
        coll_batch = {key : EpisodeDataset.collate_key(batch[key], seq_lengths, order_indicies) for key in keys}
        return coll_batch
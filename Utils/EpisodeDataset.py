import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn

class EpisodeDataset(Dataset):
    def __init__(self, csv_file, transform=None, encoding = None, action_space = None, device="cuda"):
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
        self.device = device
        
    def __len__(self):
        return len(self.episode_data['Episode'].unique())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        episode_data = self.episode_data[self.episode_data['Episode'] == idx]
        images = []
        actions = []
        rewards = []
        dones = []

        for _, row in episode_data.iterrows():
            image = Image.open(row['ImagePath'])
            if self.transform:
                image = self.transform(image)
            if self.encoding is not None:
                _, _, image = self.encoding(image.to(self.device))
                image = image.squeeze(dim=0).clone().detach()
            images.append(image)
            actions.append(row['Action'])
            rewards.append(row['Reward'])
            dones.append(row['Done'])

        actions = torch.tensor(actions, dtype=torch.long)

        images = torch.stack(images)
        actions = F.one_hot(actions, self.action_space).float() 
        rewards =  torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)
        
        images.requires_grad = True
        actions.requires_grad = True
        rewards.requires_grad = True
        dones.requires_grad = True
        

        sample = {'images': images, 
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


        keys = ['images', 'actions', 'rewards', 'dones']
        batch = {key : [sample[key] for sample in batch] for key in keys}
        coll_batch = {key : EpisodeDataset.collate_key(batch[key], seq_lengths, order_indicies) for key in keys}
        return coll_batch
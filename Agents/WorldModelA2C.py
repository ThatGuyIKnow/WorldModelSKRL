from Models.VAE import VAE
from Models.MDNRNN import MDNRNN

import torch
import torch.nn.functional as F
from skrl.models.torch import Model


class WorldModelAgent(object): 
    def __init__(self, vae_dir, mdnrnn_dir, policy, action_space):
        super().__init__()


        self.vae = VAE.load_from_checkpoint(vae_dir)
        self.mdnrnn = MDNRNN.load_from_checkpoint(mdnrnn_dir)
        self.policy = policy

        self.hidden_state = self.mdnrnn.initial_state()
        self.action = None
        self.action_space =  action_space

    def __call__(self, obs):
        return self.act(obs)
    
    def act(self, obs):
        _, _, z = self.vae.encoder(obs)

        enc_state = torch.concat([z, self.hidden_state[0]], dim=-1)
        self.action = self.policy({'states': enc_state})[0].item()

        one_hot_action = F.one_hot(torch.Tensor([self.action]).to(torch.int64), self.action_space.n)

        mus, sigmas, logpi, r, d, next_hidden = self.mdnrnn.cell(one_hot_action, z, self.hidden_state)
        self.hidden_state = next_hidden
               
        return self.action
    

    def reset(self):
        self.hidden_state = self.mdnrnn.initial_state()
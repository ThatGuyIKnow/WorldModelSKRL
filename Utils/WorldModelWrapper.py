import json
import gymnasium as gym
import numpy as np
from moviepy.editor import ImageSequenceClip
import torch
from torchvision import transforms
import wandb

from Models.MDNRNN import MDNRNN
from Models.VAE import VAE

class WorldModelWrapper(gym.Wrapper):
    def __init__(self, env, vae_model: VAE, mdnrnn_model: MDNRNN, output_dim=32+64, episode_trigger=lambda x: False, use_wandb = False, device = 'cpu'):
        super().__init__(env)

        self.vae_model = vae_model
        self.mdnrnn_model = mdnrnn_model

        self.observation_space = gym.spaces.Box(low = -np.ones(output_dim,),  high = np.ones(output_dim,),dtype = np.float16)
        self.action_space = self.env.action_space
        self.hidden_state = self.mdnrnn_model.initial_state()

        self.episode_id = 0
        self.step_id = 0
        self.episode_trigger = episode_trigger
        self.recording = False
        self.frames = []
        self.path = 'videos/CarRacing/rl-video-episode-{0}-reconstruction{1}'

        self.num_agents = 1

        self.wandb = use_wandb
        self.device = device

    def reset(self):
        obs, info = self.env.reset()
        _, _, latent = self.vae_model.encoder(torch.tensor(obs, device=self.device))  # Assuming encode method exists in your VAE model
        self.hidden_state = self.mdnrnn_model.initial_state()

        self.step_id += 1
        self.close_recording()
        self.record_reconstruction(latent)

        observation = torch.concat([latent, self.hidden_state[0]], dim=-1)
        return observation, info

    def step(self, action):
        next_obs, reward, done, truncated, info = self.env.step(action)

        action = torch.Tensor(action).view(1, -1)

        _, _, latent_next = self.vae_model.encoder(torch.tensor(next_obs, device=self.device))  # Assuming encode method exists in your VAE model
        input =  torch.cat([latent_next, action], dim=-1)
        _, _, _, _, _, next_hidden = self.mdnrnn_model.cell(input, self.hidden_state)  # Assuming predict method exists in your MDNRNN model
        self.hidden_state = next_hidden
        next_observation = torch.concat([latent_next, self.hidden_state[0]], dim=-1)

        self.record_reconstruction(latent_next)

        return next_observation, reward, done, truncated, info

    def record_reconstruction(self, latent):
        if self.recording is False:
            self.recording = self.episode_trigger(self.episode_id)
        
        if self.recording is False:
            return

        # Append the image to the video frames list
        self.frames.append(latent)

    def write_metadata(self):
        """Writes metadata to metadata path."""
        metadata = {"step_id": self.step_id, "episode_id": self.episode_id, "content_type": "video/mp4"}
        with open(self.path.format(self.episode_id, '.meta.json'), "w") as f:
            json.dump(metadata, f)

    def close_recording(self):
        self.episode_id += 1

        if self.recording is False:
            return
        
        self.write_metadata()
        # Decode latent representation into an image
        decoded_observation = self.vae_model.decoder(torch.concat(self.frames))  # Assuming decode method exists in your VAE model

        # Convert image to uint8 and resize to fit the video dimensions
        transform = transforms.Resize((400, 400))
        imgs = torch.stack([transform(img) for img in decoded_observation])
        imgs = (imgs.detach().numpy()*255).astype(np.uint8)
        imgs = np.repeat(imgs, 3, axis=1)
        imgs = np.moveaxis(imgs, 1, -1)
        # Create a video writer object
        fps = 50
        clip = ImageSequenceClip([i for i in imgs], fps=fps)
        clip.write_videofile(self.path.format(self.episode_id, '.mp4'), fps=fps)
        
        # Log video to wandb
        if self.wandb:
            wandb.log({"video": wandb.Video(self.path.format(self.episode_id, '.mp4'), fps=fps)})


        # Clear frames list
        self.frames = []
        self.recording = False
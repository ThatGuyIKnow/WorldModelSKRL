import os
from pathlib import Path
import gymnasium as gym
import numpy as np
from PIL import Image
import csv
import torch

from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm
from Agents.WorldModelA2C import WorldModelAgent
from Models.A2CModel import ActorMLP
from Utils.TransformerWrapper import TransformWrapper

def rollout_and_save_images_with_csv(env_name, output_folder, total_steps=1000, max_eps_length=50, env_kwargs={}):
    """
    Simulate a gym environment for a specified number of steps, save each frame as an image, 
    and record details (reward, done, action, image paths) in a CSV file.

    Parameters:
    env_name (str): The name of the gym environment to simulate.
    output_folder (str): The directory where the images and CSV will be saved.
    total_steps (int): The total number of steps to simulate across all episodes.
    """
    # Initialize the environment
    env = gym.make(env_name, render_mode='rgb_array', **env_kwargs)
    env = TransformWrapper(env)

    wandb_logger = WandbLogger(log_model="all")

    vae_checkpoint_reference = "team-good-models/model-registry/WorldModelVAE:v0"
    mdnrnn_checkpoint_reference = "team-good-models/model-registry/WorldModelMDNRNN:v0"
    policy_checkpoint_reference = "team-good-models/model-registry/WorldModelActor:v0"

    vae_dir = wandb_logger.download_artifact(vae_checkpoint_reference, artifact_type="model")
    mdnrnn_dir = wandb_logger.download_artifact(mdnrnn_checkpoint_reference, artifact_type="model")
    policy_dir = wandb_logger.download_artifact(policy_checkpoint_reference, artifact_type="model")

    observation_space = gym.spaces.Box(low = np.zeros(32+64,),  high = np.ones(32+64,),dtype = np.float16)

    policy = ActorMLP(observation_space=observation_space,
                      action_space=env.action_space,
                      device='cuda')
    policy.load(Path(policy_dir) / "actorPolicy.pt")

    world_model_agent = WorldModelAgent(vae_dir = Path(vae_dir) / "model.ckpt",
                                mdnrnn_dir = Path(mdnrnn_dir) / "model.ckpt",
                                policy = policy,
                                action_space=env.action_space)

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_file_path = os.path.join(output_folder, 'details_simulation.csv')
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Step', 'Action', 'Reward', 'Done', 'Truncated', 'ImagePath', 'NextImagePath'])


    observation, infos = env.reset()

    episode = 0
    for step_count in tqdm(range(total_steps)):
        observation, _ = env.reset()
        world_model_agent.reset()
        
        for eps_length in range(max_eps_length):
            # Render the environment to a numpy array and save as an image
            frame = env.render()
            img_path = os.path.join(output_folder, f'episode_{episode}_step_{eps_length}.png')
            Image.fromarray(frame).save(img_path)

            action = world_model_agent(observation)  # Replace this with your action selection mechanism
            observation, reward, done, truncated, info = env.step(action)
        
            truncated |= eps_length > max_eps_length                

            # Prepare next image path for CSV (None if simulation ends)
            next_img_path = os.path.join(output_folder, f'episode_{episode}_step_{eps_length+1}.png') if not done else None

            # Write details to CSV
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, eps_length, action, reward, done, truncated, img_path, next_img_path])

            if done or truncated:
                observation, _ = env.reset()
                world_model_agent.reset()


        episode += 1

    env.close()


# Example usage
rollout_and_save_images_with_csv('CarRacing-v2', './data/pretrained_carracing-v2', total_steps=200, max_eps_length = 1000, env_kwargs={'continuous': False})

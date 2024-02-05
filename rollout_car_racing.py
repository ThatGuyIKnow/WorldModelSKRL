import os
import gymnasium as gym
import numpy as np
from PIL import Image
import csv

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

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_file_path = os.path.join(output_folder, 'details_simulation.csv')
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Step', 'Action', 'Reward', 'Done', 'Truncated', 'ImagePath', 'NextImagePath'])

    step_count = 0
    episode = 0
    while step_count < total_steps:
        observation = env.reset()
        eps_length = 0
        last_img_path = None
        for step in range(total_steps):
            # Render the environment to a numpy array and save as an image
            frame = env.render()
            img_path = os.path.join(output_folder, f'episode_{episode}_step_{step}.png')
            Image.fromarray(frame).save(img_path)

            action = env.action_space.sample()  # Replace this with your action selection mechanism
            observation, reward, done, truncated, info = env.step(action)
        
            truncated |= eps_length > max_eps_length

            # Prepare next image path for CSV (None if simulation ends)
            next_img_path = os.path.join(output_folder, f'episode_{episode}_step_{step+1}.png') if not done else None

            # Write details to CSV
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, step, action, reward, done, truncated, img_path, next_img_path])

            last_img_path = img_path
            step_count += 1
            eps_length += 1
            if done or step_count >= total_steps or eps_length > max_eps_length:
                break

        episode += 1

    env.close()

# Example usage
rollout_and_save_images_with_csv('CarRacing-v2', './data/carracing-v2', total_steps=1000, max_eps_length = 50, env_kwargs={'continuous': False})

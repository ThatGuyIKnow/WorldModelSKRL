import time
import gymnasium as gym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from Models.WorldModelController import WorldModelController
from Agents.WorldModelDQN import WorldModelDQN


# seed for reproducibility
set_seed(123)  # e.g. `set_seed(42)` for fixed seed

# load and wrap the environment
env_name = 'ALE/Breakout-v5'
env = gym.make(env_name, render_mode='rgb_array')
env = gym.wrappers.AtariPreprocessing(env, frame_skip=1, screen_size=64)
#env = gym.wrappers.RecordVideo(env, f'videos/{env_name}_{time.time()}/', episode_trigger=lambda x: x % 50 == 0, video_length=500)
env = wrap_env(env)

device = env.device

# instantiate a memory as experience replay
memory = RandomMemory(memory_size=15000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's models (function approximators).
# DQN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#models
models = {}
models["world_model_controller"] = WorldModelController(env.observation_space, env.action_space, device)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#configuration-and-hyperparameters
cfg = DQN_DEFAULT_CONFIG.copy()
cfg["learning_starts"] = 1e6
cfg["update_interval"] = 4
cfg["target_update_interval"] = 4
cfg["polyak"] = 0.05
cfg["learning_rate"] = 1e-4
cfg["exploration"]["initial_epsilon"] = 1.0
cfg["exploration"]["final_epsilon"] = 0.01
cfg["exploration"]["timesteps"] = 100000
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = f"runs/torch/{env_name}"

agent = WorldModelDQN(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()

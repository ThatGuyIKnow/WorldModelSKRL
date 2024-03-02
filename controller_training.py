# import the agent and its default configuration
from typing import Callable
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

import numpy as np
import gymnasium as gym
from skrl.memories.torch import RandomMemory
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path
import torch

from Models.AgentModel import ActorMLP, CriticMLP

from Models.WorldModel import WorldModel
from Models.VAE import VAE
from Models.MDNRNN import MDNRNN
from Trainers.WorldModelSequentialTrainer import WorldModelSequentialTrainer
from Utils.TransformerWrapper import TransformWrapper

from stable_baselines3.common.monitor import Monitor



device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

env = gym.make("CarRacing-v2", render_mode='rgb_array')
env = Monitor(env)
env = gym.wrappers.RecordVideo(env, './videos/CarRacing', episode_trigger=lambda x: x % 10 == 0)
env = TransformWrapper(env)

wandb_logger = WandbLogger(log_model="all")

# latent z shape + hidden rnn shape
observation_space = gym.spaces.Box(low = np.zeros(32+64,),  high = np.ones(32+64,),dtype = np.float16)

# instantiate the model (assumes there is a wrapped environment: env)
critic = CriticMLP(observation_space=observation_space,
             action_space=env.action_space,
             device=device,
             clip_actions=False)
policy = ActorMLP(observation_space=observation_space,
             action_space=env.action_space,
             device=device)


vae_checkpoint_reference = "team-good-models/model-registry/WorldModelVAE:latest"
mdnrnn_checkpoint_reference = "team-good-models/model-registry/WorldModelMDNRNN:latest"

vae_dir = wandb_logger.download_artifact(vae_checkpoint_reference, artifact_type="model")
mdnrnn_dir = wandb_logger.download_artifact(mdnrnn_checkpoint_reference, artifact_type="model")

vae = VAE.load_from_checkpoint(Path(vae_dir) / "model.ckpt")
mdnrnn = MDNRNN.load_from_checkpoint(Path(mdnrnn_dir) / "model.ckpt", strict=False)

world_model = WorldModel(observation_space=env.observation_space,
             action_space=env.action_space,
             vae = vae,
             mdnrnn = mdnrnn,
             device = device)

# instantiate the agent's models
models = {}
models["policy"] = policy
models["value"] = critic  # only required during training

# adjust some configuration if necessary
cfg_agent = PPO_DEFAULT_CONFIG.copy()
cfg_agent['learning_starts'] = 15000
cfg_agent['entropy_loss_scale'] = 1e-2
cfg_agent['learning_rate'] = 2.5e-4
cfg_agent['mini_batches'] = 4
cfg_agent['learning_epochs'] = 4
cfg_agent['vf_coef'] = 0.5
cfg_agent['rollouts'] = 64

cfg_agent['experiment']['wandb'] = True
cfg_agent['experiment']['wandb_kwargs'] = {'project': 'world_model', 'monitor_gym': True}


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=15000, num_envs=1, device=device, replacement=False)

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = PPO(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=observation_space,
            action_space=env.action_space,
            device=device)


# env = WandbRecordVideo(env, './videos/CarRacing', episode_trigger=lambda x: x % 10 == 0, agent=agent)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": True}
trainer = WorldModelSequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent, ], world_model = world_model, device=device)

# start training
trainer.train()

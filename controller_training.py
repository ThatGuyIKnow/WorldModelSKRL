import skrl
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed
import gymnasium as gym
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path
import torch

from Models.AgentModels import ActorMLP, CriticMLP
from Models.VAE import VAE
from Models.MDNRNN import MDNRNN
from Utils.ClipRewardWrapper import ClipRewardWrapper
from Utils.TransformerWrapper import TransformWrapper
from stable_baselines3.common.monitor import Monitor
from Utils.WorldModelWrapper import WorldModelWrapper

LATENT_SPACE = 32
HIDDEN_SPACE = 256

set_seed(42)

# Set device (CPU or GPU)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Initialize Weights and Biases (Wandb) logger
wandb_logger = WandbLogger(log_model="all")

# 1) References to the latest VAE and MDNRNN checkpoints on Wandb
# 2) Download VAE and MDNRNN checkpoints from Wandb
# 3) Load VAE and MDNRNN models from checkpoints
vae_checkpoint_reference = "team-good-models/model-registry/WorldModelVAE:latest"
mdnrnn_checkpoint_reference = "team-good-models/model-registry/WorldModelMDNRNN:latest"

vae_dir = wandb_logger.download_artifact(vae_checkpoint_reference, artifact_type="model")
mdnrnn_dir = wandb_logger.download_artifact(mdnrnn_checkpoint_reference, artifact_type="model")

vae = VAE.load_from_checkpoint(Path(vae_dir) / "model.ckpt")
mdnrnn = MDNRNN.load_from_checkpoint(Path(mdnrnn_dir) / "model.ckpt", strict=False)

vae.freeze()
mdnrnn.freeze()

# Create the environment
env = gym.make("CarRacing-v2", render_mode='rgb_array')
env = ClipRewardWrapper(env, -0.101, 1.)  
env = Monitor(env)  # Monitor the environment (Necessary for Wandb)
env = gym.wrappers.RecordVideo(env, './videos/CarRacing', episode_trigger=lambda x: x % 100 == 0)  # Record videos
env = TransformWrapper(env)  # Apply necessary visual transformations to the environment
env = WorldModelWrapper(env, vae, mdnrnn, output_dim=LATENT_SPACE+HIDDEN_SPACE, episode_trigger=lambda x: x % 100 == 0, use_wandb=True, device = device)  # Engange the WorldModel
env = wrap_env(env)  # Wrap environment with skrl wrapper to make it compatible

# Instantiate actor and critic models
critic = CriticMLP(observation_space=env.observation_space,
                   action_space=env.action_space,
                   device=device,
                   clip_actions=False)
policy = ActorMLP(observation_space=env.observation_space,
                  action_space=env.action_space,
                  device=device)

# Model dictionary
models = {"policy": policy, "value": critic}  # Models used by the agent during training

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# Configure agent's default parameters
cfg_agent = PPO_DEFAULT_CONFIG.copy()
cfg_agent['rollouts'] = 1024
cfg_agent['learning_starts'] = cfg_agent['rollouts']
cfg_agent['entropy_loss_scale'] = 1e-2
cfg_agent['learning_rate'] = 3e-4
cfg_agent['mini_batches'] = 4
cfg_agent['learning_epochs'] = 8
cfg_agent['vf_coef'] = 0.5
cfg_agent['experiment']['wandb'] = True
cfg_agent['experiment']['wandb_kwargs'] = {'project': 'world_model', 'monitor_gym': True}

# Instantiate experience memory for the agent
memory = RandomMemory(memory_size=cfg_agent['rollouts'], num_envs=1, device=device, replacement=False)

# Instantiate the PPO agent
agent = PPO(models=models,
            memory=memory,
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# Trainer configuration
cfg_trainer = {"timesteps": int(1e6), "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent, ])

# Start training
trainer.train()

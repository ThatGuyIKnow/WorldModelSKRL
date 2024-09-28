
# WorldModelSKRL

## Overview
This repository implements world models in reinforcement learning, leveraging powerful tools like Lightning PyTorch for modular training and SKRL for integrating various RL algorithms. It includes modules for training VAEs, MDN-RNNs, and SAC controllers.

## Features
- **Lightning PyTorch**: Simplifies multi-GPU training, logging, and model checkpointing.
- **SKRL**: Provides easy-to-use interfaces for reinforcement learning algorithms.
- **World Models**: VAE for state representation, MDN-RNN for dynamics, and SAC for control in simulated environments.

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/ThatGuyIKnow/WorldModelSKRL.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Train VAE: `vae_training.py`
- Train MDN-RNN: `mdnrnn_training.py`
- Train Controller (SAC): `controller_training_sac.py`

## License
MIT License.

---

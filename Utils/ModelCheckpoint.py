from pathlib import Path
import torch
import wandb

class ModelCheckpoint:
    def __init__(self, model, checkpoint_interval, dir, model_name, best_metric=None):
        self.model = model
        self.checkpoint_interval = checkpoint_interval
        self.best_metric = best_metric
        self.model_name = model_name
        self.checkpoint_folder = f"{dir}/checkpoints"

    def save_checkpoint(self, timestep, current_metric):
        # Save regular checkpoint
        if timestep % self.checkpoint_interval == 0:
            checkpoint_path = f"{self.checkpoint_folder}/{self.model_name}_{timestep}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)

        # Save best checkpoint
        if self.best_metric is None or current_metric < self.best_metric:
            best_checkpoint_path = f"{self.checkpoint_folder}/{self.model_name}_best.pt"
            torch.save(self.model.state_dict(), best_checkpoint_path)
            self.best_metric = current_metric


def retrieve_wandb_checkpoint(user : str, project : str, model_run : str, version='best') -> Path:
    # reference can be retrieved in artifacts panel
    # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
    checkpoint_reference = f"{user}/{project}/{model_run}:{version}"

    # download checkpoint locally (if not already cached)
    run = wandb.init(project=project)
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()

    # load checkpoint
    return Path(artifact_dir) / "model.ckpt"
"""
Experiment Tracking: TensorBoard and Weights & Biases integration.
"""

import logging
from typing import Dict, Optional, List
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class BaseTracker:
    """Base class for experiment trackers."""

    def __init__(self, project_name: str, run_name: str = ""):
        self.project_name = project_name
        self.run_name = run_name

    def log_metrics(self, metrics: Dict, step: int, prefix: str = ""):
        raise NotImplementedError

    def log_hyperparams(self, params: Dict):
        raise NotImplementedError

    def log_images(self, images: Dict, step: int):
        raise NotImplementedError

    def close(self):
        pass


class TensorBoardTracker(BaseTracker):
    """TensorBoard experiment tracker."""

    def __init__(self, project_name: str, run_name: str = "", log_dir: str = "./outputs/logs"):
        super().__init__(project_name, run_name)
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = str(Path(log_dir) / (run_name or "default"))
        self.writer = SummaryWriter(self.log_dir)
        logger.info(f"TensorBoard logging to {self.log_dir}")

    def log_metrics(self, metrics: Dict, step: int, prefix: str = ""):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}{key}", value, step)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"{prefix}{key}/{k}", v, step)

    def log_hyperparams(self, params: Dict):
        from torch.utils.tensorboard import SummaryWriter
        hp_str = "\n".join(f"{k}: {v}" for k, v in params.items())
        self.writer.add_text("hparams", hp_str)

    def log_images(self, images: Dict, step: int):
        """images: dict of {name: numpy_array (C,H,W) or (H,W,3)}."""
        try:
            import torch
            for name, img_array in images.items():
                img_tensor = torch.from_numpy(img_array).float()
                if img_tensor.dim() == 3:
                    if img_tensor.shape[2] == 3:  # H,W,C -> C,H,W
                        img_tensor = img_tensor.permute(2, 0, 1)
                self.writer.add_image(name, img_tensor, step)
        except Exception as e:
            logger.warning(f"Failed to log images: {e}")

    def close(self):
        self.writer.close()


class WandbTracker(BaseTracker):
    """Weights & Biases experiment tracker."""

    def __init__(self, project_name: str, run_name: str = "", entity: str = ""):
        super().__init__(project_name, run_name)
        try:
            import wandb
            self.wandb = wandb
            wandb.init(
                project=project_name,
                name=run_name or None,
                entity=entity or None,
                reinit=True,
            )
            logger.info(f"W&B run: {wandb.run.name}")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.wandb = None
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            self.wandb = None

    def log_metrics(self, metrics: Dict, step: int, prefix: str = ""):
        if self.wandb is None:
            return
        log_dict = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_dict[f"{prefix}{key}"] = value
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        log_dict[f"{prefix}{key}/{k}"] = v
        self.wandb.log(log_dict, step=step)

    def log_hyperparams(self, params: Dict):
        if self.wandb is None:
            return
        self.wandb.config.update(params)

    def log_images(self, images: Dict, step: int):
        if self.wandb is None:
            return
        try:
            wandb_images = {}
            for name, img_array in images.items():
                wandb_images[name] = self.wandb.Image(img_array)
            self.wandb.log(wandb_images, step=step)
        except Exception as e:
            logger.warning(f"W&B image log failed: {e}")

    def close(self):
        if self.wandb is not None:
            self.wandb.finish()


class DummyTracker(BaseTracker):
    """No-op tracker when tracking is disabled."""

    def log_metrics(self, metrics: Dict, step: int, prefix: str = ""):
        pass

    def log_hyperparams(self, params: Dict):
        pass

    def log_images(self, images: Dict, step: int):
        pass


class NullTracker:
    """No-op tracker when tracking is disabled (no __init__ args needed)."""

    def log_metrics(self, metrics, step=0, prefix=""):
        pass

    def log_hyperparams(self, params):
        pass

    def log_images(self, images, step=0):
        pass

    def close(self):
        pass


def build_tracker(cfg) -> BaseTracker:
    """Build experiment tracker from config."""
    track_cfg = cfg.get("tracking", {})
    backend = track_cfg.get("backend", "none")

    if backend == "tensorboard":
        log_dir = str(Path(cfg["output"]["dir"]) / "logs")
        return TensorBoardTracker(
            project_name=track_cfg.get("project_name", "chest-xray"),
            run_name=track_cfg.get("run_name", ""),
            log_dir=log_dir,
        )
    elif backend == "wandb":
        wandb_cfg = track_cfg.get("wandb", {})
        return WandbTracker(
            project_name=track_cfg.get("project_name", "chest-xray"),
            run_name=track_cfg.get("run_name", ""),
            entity=wandb_cfg.get("entity", ""),
        )
    else:
        logger.info("Experiment tracking disabled (backend=none)")
        return NullTracker()

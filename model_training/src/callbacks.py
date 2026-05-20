"""
Callbacks: Early Stopping, Checkpointing, LR Scheduling, Gradient Clipping.
"""

import os
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    ReduceLROnPlateau,
)

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Monitor a metric and stop training if no improvement."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, current_score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = current_score
            return False

        improved = False
        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:  # max
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered: {self.monitor} hasn't improved "
                    f"for {self.patience} epochs (best={self.best_score:.4f})"
                )
                return True

        return False

    def state_dict(self):
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state):
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.should_stop = state.get("should_stop", False)


class CheckpointManager:
    """Save best / last / top-k model checkpoints."""

    def __init__(
        self,
        output_dir: str,
        monitor: str = "val_auc",
        mode: str = "max",
        save_best: bool = True,
        save_last: bool = True,
        save_top_k: int = 3,
    ):
        self.output_dir = output_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.save_top_k = save_top_k

        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.top_k_scores = []
        os.makedirs(output_dir, exist_ok=True)

    def _is_better(self, new_score: float) -> bool:
        if self.mode == "max":
            return new_score > self.best_score
        return new_score < self.best_score

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict] = None,
        scheduler=None,
        early_stopping=None,
    ):
        """Save checkpoint with full training state."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "metrics": metrics or {},
            "monitor_score": metrics.get(self.monitor, 0) if metrics else 0,
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if early_stopping is not None:
            state["early_stopping"] = early_stopping.state_dict()

        # Save last checkpoint
        if self.save_last:
            path = os.path.join(self.output_dir, "last_checkpoint.pth")
            torch.save(state, path)

        # Save best checkpoint
        current_score = metrics.get(self.monitor, 0) if metrics else 0
        if self.save_best and self._is_better(current_score):
            self.best_score = current_score
            path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(state, path)
            logger.info(f"  -> Saved best model ({self.monitor}={current_score:.4f})")

        # Save top-k
        if self.save_top_k > 0:
            self.top_k_scores.append((current_score, epoch, state))
            if self.mode == "max":
                self.top_k_scores.sort(key=lambda x: x[0], reverse=True)
            else:
                self.top_k_scores.sort(key=lambda x: x[0])
            # Prune old
            for score, ep, st in self.top_k_scores[self.save_top_k:]:
                old_path = os.path.join(self.output_dir, f"top_{ep:04d}.pth")
                if os.path.exists(old_path):
                    os.remove(old_path)
            self.top_k_scores = self.top_k_scores[:self.save_top_k]
            for score, ep, st in self.top_k_scores:
                path = os.path.join(self.output_dir, f"top_{ep:04d}.pth")
                torch.save(st, path)


def build_scheduler(optimizer, cfg_scheduler: Dict, num_epochs: int = 30):
    """Build learning rate scheduler from config."""
    sched_type = cfg_scheduler.get("type", "cosine")

    if sched_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg_scheduler.get("T_max", num_epochs),
            eta_min=cfg_scheduler.get("eta_min", 1e-6),
        )
    elif sched_type == "cosine_warmup":
        # Warmup + cosine annealing via warm restarts
        warmup_epochs = cfg_scheduler.get("warmup_epochs", 3)
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_epochs - warmup_epochs,
            T_mult=1,
            eta_min=cfg_scheduler.get("eta_min", 1e-6),
        )
    elif sched_type == "step":
        return StepLR(
            optimizer,
            step_size=cfg_scheduler.get("step_size", 10),
            gamma=cfg_scheduler.get("step_gamma", 0.1),
        )
    elif sched_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=cfg_scheduler.get("patience", 3),
            factor=cfg_scheduler.get("factor", 0.5),
            min_lr=cfg_scheduler.get("eta_min", 1e-6),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


def clip_grad_norm_(model, max_norm: float = 1.0):
    """Gradient clipping. No-op if max_norm <= 0."""
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

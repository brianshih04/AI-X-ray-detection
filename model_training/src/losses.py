"""
Loss Functions: BCEWithLogitsLoss, FocalLoss, and ASL for multi-label classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .asymmetric_loss import AsymmetricLoss, AsymmetricLossOptimized


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced multi-label classification.

    FocalLoss = -alpha * (1 - p_t)^gamma * log(p_t)
    Reduces loss for well-classified examples, focusing on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_loss(cfg) -> nn.Module:
    """Build loss function from config.
    
    Supported: bce, focal, asl, asl_weighted
    """
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "bce")

    if loss_type == "focal":
        return FocalLoss(
            alpha=loss_cfg.get("focal_alpha", 0.25),
            gamma=loss_cfg.get("focal_gamma", 2.0),
        )
    elif loss_type == "asl":
        return AsymmetricLoss(
            gamma_neg=loss_cfg.get("asl_gamma_neg", 4.0),
            gamma_pos=loss_cfg.get("asl_gamma_pos", 0.0),
            clip=loss_cfg.get("asl_clip", 0.05),
        )
    elif loss_type == "asl_weighted":
        # ASL with pos_weight — set later in train.py
        return AsymmetricLossOptimized(
            gamma_neg=loss_cfg.get("asl_gamma_neg", 4.0),
            gamma_pos=loss_cfg.get("asl_gamma_pos", 1.0),
            clip=loss_cfg.get("asl_clip", 0.05),
            pos_weight=None,
        )
    elif loss_type == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_pos_weights(train_loader, num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute positive class weights from training data for WeightedBCE.

    pos_weight[i] = num_negatives_i / num_positives_i
    """
    pos_counts = torch.zeros(num_classes, device=device)
    total = 0

    for _, labels in train_loader:
        labels = labels.to(device)
        pos_counts += labels.sum(dim=0)
        total += labels.shape[0]

    neg_counts = total - pos_counts
    # Clamp to avoid division by zero
    pos_weights = neg_counts / (pos_counts + 1e-6)
    return pos_weights

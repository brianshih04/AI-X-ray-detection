"""
Asymmetric Loss (ASL) for multi-label classification.

Reference: "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al., 2021)
- asymmetric_γ: different focusing for positive/negative (γ_neg > γ_pos)
- probability shifting: margin-based threshold adjustment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """ASL: Asymmetric Loss for multi-label classification.
    
    Key idea: use different gamma values for positive and negative samples,
    effectively down-weighting easy negatives much more aggressively than 
    standard Focal Loss. This reduces false positive over-prediction.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Sigmoid probabilities
        probs = torch.sigmoid(inputs)
        
        # Probability shifting (clip predictions)
        if self.clip and self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1.0)
        else:
            probs_neg = probs

        # Loss calculation
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))

        # Focusing
        if self.gamma_pos > 0:
            loss_pos *= (1 - probs) ** self.gamma_pos
        if self.gamma_neg > 0:
            loss_neg *= probs_neg ** self.gamma_neg

        loss = -(loss_pos + loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLossOptimized(nn.Module):
    """Optimized ASL with per-class pos_weight support.
    
    Combines ASL with class-balanced weighting for extreme imbalance
    (e.g., Hernia 86 vs No Finding 9861).
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        pos_weight: torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.register_buffer("pos_weight", pos_weight)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        
        if self.clip and self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1.0)
        else:
            probs_neg = probs

        # Per-sample loss
        loss_pos = targets * F.logsigmoid(inputs)
        loss_neg = (1 - targets) * F.logsigmoid(-inputs)

        # Asymmetric focusing
        if self.gamma_pos > 0:
            loss_pos *= (1 - probs) ** self.gamma_pos
        if self.gamma_neg > 0:
            loss_neg *= probs_neg ** self.gamma_neg

        # Class weighting
        if self.pos_weight is not None:
            loss_pos *= self.pos_weight

        loss = -(loss_pos + loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

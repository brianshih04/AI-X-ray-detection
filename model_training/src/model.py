"""
Model Module: DenseNet-121 with transfer learning + multi-label classification head.

Supports: densenet121, efficientnet_b0, vit_b_16 backbones.
All use sigmoid (not softmax) for multi-label classification.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class ChestXrayClassifier(nn.Module):
    """Multi-label chest X-ray classifier.

    Architecture:
        ImageNet-pretrained backbone -> Dropout -> Linear(num_classes)
        Output: raw logits (apply sigmoid externally)
    """

    BACKBONE_REGISTRY = {
        "densenet121": (models.densenet121, "IMAGENET1K_V1"),
        "efficientnet_b0": (models.efficientnet_b0, "IMAGENET1K_V1"),
        "efficientnet_b3": (models.efficientnet_b3, "IMAGENET1K_V1"),
        "convnext_tiny": (models.convnext_tiny, "IMAGENET1K_V1"),
        "convnext_small": (models.convnext_small, "IMAGENET1K_V1"),
    }

    def __init__(
        self,
        backbone_name: str = "densenet121",
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        if backbone_name not in self.BACKBONE_REGISTRY:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "
                f"Choose from: {list(self.BACKBONE_REGISTRY.keys())}"
            )

        builder_fn, weights_key = self.BACKBONE_REGISTRY[backbone_name]
        weights = weights_key if pretrained else None

        self.backbone = builder_fn(weights=weights)

        # Extract features dimensionality and remove original classifier
        num_features = self._replace_classifier(backbone_name)

        # Multi-label classification head: Dropout + Linear
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes),
        )

        self._init_classifier()

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Model: {backbone_name} | params={total:,} | "
            f"trainable={trainable:,} | classes={num_classes}"
        )

    def _replace_classifier(self, backbone_name: str) -> int:
        """Remove original classifier head and return feature dim."""
        if backbone_name.startswith("densenet"):
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name.startswith("efficientnet"):
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name.startswith("convnext"):
            # ConvNeXt classifier: Sequential(LayerNorm2d, Flatten, Linear)
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
        return num_features

    def _init_classifier(self):
        """Xavier init for the classification head."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # Ensure features are 2D: (B, C, H, W) -> (B, C)
        if features.dim() > 2:
            features = features.flatten(2).mean(dim=2)  # Global average pooling
        return self.classifier(features)

    def freeze_backbone(self):
        """Freeze all backbone parameters (only train classifier)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only classifier is trainable")

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — all parameters trainable")

    def get_param_groups(self, lr_head: float, lr_backbone: float, weight_decay: float):
        """Return param groups with separate learning rates for backbone and head."""
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone},
            {"params": self.classifier.parameters(), "lr": lr_head},
        ]


def build_model(cfg) -> ChestXrayClassifier:
    """Build model from config dict."""
    model_cfg = cfg["model"]
    model = ChestXrayClassifier(
        backbone_name=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg.get("dropout", 0.3),
    )
    return model


def export_onnx(
    model: nn.Module,
    output_path: str,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    img_size: int = 224,
):
    """Export model to ONNX format."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    logger.info(f"ONNX exported to {output_path} (opset={opset_version})")


def export_torchscript(
    model: nn.Module,
    output_path: str,
    img_size: int = 224,
):
    """Export model to TorchScript via tracing."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    traced = torch.jit.trace(model, dummy)
    traced.save(output_path)
    logger.info(f"TorchScript exported to {output_path}")

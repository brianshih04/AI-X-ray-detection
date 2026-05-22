"""
Chest X-ray Multi-label Classification — Complete Training Pipeline

Usage:
  python train.py                          # use default config
  python train.py --config my_config.yaml  # custom config
  python train.py --data_dir /path/to/CheXpert  # override data dir

Requirements: see requirements.txt
"""

import os
import sys
import time
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import build_dataloaders
from src.model import build_model, export_onnx, export_torchscript
from src.metrics import compute_all_metrics, format_metrics_report, find_optimal_thresholds
from src.callbacks import EarlyStopping, CheckpointManager, build_scheduler, clip_grad_norm_
from src.tracker import build_tracker
from src.losses import build_loss, compute_pos_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Chest X-ray Multi-label Training")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override head learning rate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true",
                        help="Run evaluation only (no training)")
    parser.add_argument("--export_only", action="store_true",
                        help="Export model to ONNX/TorchScript and exit")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu/cuda/cuda:0)")
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model, train_loader, criterion, optimizer, device,
    scaler=None, accum_steps: int = 1, clip_grad_norm: float = 1.0,
    epoch: int = 0, tracker=None, log_interval: int = 10,
    use_amp: bool = True,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps
            loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer) if (use_amp and scaler) else None
                clip_grad_norm_(model, clip_grad_norm)

            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        num_batches += 1

        if tracker and (batch_idx + 1) % log_interval == 0:
            tracker.log_metrics(
                {"batch_loss": loss.item() * accum_steps, "lr": optimizer.param_groups[0]["lr"]},
                step=epoch * len(train_loader) + batch_idx,
                prefix="train/",
            )

        pbar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model, val_loader, criterion, device,
    class_names, epoch: int = 0, tracker=None,
):
    """Run validation and compute all metrics."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    avg_loss = total_loss / len(val_loader)

    # Compute full metrics
    metrics = compute_all_metrics(all_labels, all_probs, class_names)

    # Find optimal thresholds on this epoch
    optimal_thresh = find_optimal_thresholds(all_labels, all_probs, class_names)
    metrics["optimal_thresholds"] = optimal_thresh.tolist()

    return avg_loss, metrics, all_probs, all_labels


def main():
    args = parse_args()

    # ── Load Config ────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        # Try relative to this script
        cfg_path = Path(__file__).parent / args.config
    cfg = load_config(str(cfg_path))
    logger.info(f"Config loaded from {cfg_path}")

    # ── Override config from CLI args ──────────────────
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr_head"] = args.lr
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir
    if args.seed:
        cfg["output"]["seed"] = args.seed

    # ── Setup ──────────────────────────────────────────
    seed = cfg["output"]["seed"]
    set_seed(seed)
    output_dir = cfg["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    num_epochs = cfg["training"]["num_epochs"]
    label_columns = cfg.get("labels", [])
    img_size = cfg["data"]["img_size"]

    # ── Build Tracker ──────────────────────────────────
    tracker = build_tracker(cfg)
    tracker.log_hyperparams({k: v for k, v in cfg.items() if not isinstance(v, dict)})
    # Log top-level hyperparams
    for section in ["training", "model", "loss", "scheduler"]:
        if section in cfg:
            tracker.log_hyperparams({f"{section}/{k}": v for k, v in cfg[section].items() if not isinstance(v, dict)})

    # ── Build Data ─────────────────────────────────────
    logger.info("Building dataloaders...")
    train_loader, val_loader, label_columns = build_dataloaders(cfg)

    # ── Build Model ────────────────────────────────────
    logger.info("Building model...")
    model = build_model(cfg)
    model = model.to(device)

    # ── Build Loss ─────────────────────────────────────
    criterion = build_loss(cfg)
    loss_type = cfg.get("loss", {}).get("type", "bce")
    
    if cfg.get("loss", {}).get("use_pos_weight", False) or loss_type == "asl_weighted":
        logger.info("Computing positive weights from training data...")
        pos_weights = compute_pos_weights(train_loader, model.num_classes, device)
        logger.info(f"  pos_weight range: {pos_weights.min().item():.2f} - {pos_weights.max().item():.2f}")
        
        if loss_type == "asl_weighted":
            # ASL with class weighting
            criterion.pos_weight = pos_weights.to(device)
            logger.info("  Using ASL with pos_weight for class imbalance")
        else:
            # Standard BCE with pos_weight
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # ── Build Optimizer ───────────────────────────────
    train_cfg = cfg["training"]
    model.freeze_backbone()
    freeze_epochs = cfg["model"].get("freeze_backbone_epochs", 0)

    param_groups = model.get_param_groups(
        lr_head=train_cfg["lr_head"],
        lr_backbone=train_cfg["lr_backbone"],
        weight_decay=train_cfg["weight_decay"],
    )

    if train_cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        )
    elif train_cfg["optimizer"] == "sgd":
        optimizer = optim.SGD(
            param_groups,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg['optimizer']}")

    # ── Build Scheduler ────────────────────────────────
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), num_epochs)
    is_plateau = cfg.get("scheduler", {}).get("type") == "plateau"

    # ── Build Callbacks ────────────────────────────────
    cb_cfg = cfg.get("callbacks", {})
    es_cfg = cb_cfg.get("early_stopping", {})

    early_stopping = None
    if es_cfg.get("enabled", False):
        early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 7),
            min_delta=es_cfg.get("min_delta", 0.001),
            monitor=es_cfg.get("monitor", "val_loss"),
            mode=es_cfg.get("mode", "min"),
        )

    ckpt_cfg = cb_cfg.get("checkpoint", {})
    checkpoint_mgr = CheckpointManager(
        output_dir=output_dir,
        monitor=ckpt_cfg.get("monitor", "val_auc"),
        mode=ckpt_cfg.get("mode", "max"),
        save_best=ckpt_cfg.get("save_best", True),
        save_last=ckpt_cfg.get("save_last", True),
        save_top_k=ckpt_cfg.get("save_top_k", 3),
    )

    # ── AMP Scaler ─────────────────────────────────────
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    accum_steps = train_cfg.get("grad_accum_steps", 1)
    clip_grad = train_cfg.get("clip_grad_norm", 1.0)

    log_interval = cfg.get("tracking", {}).get("log_interval", 10)

    # ── Resume from checkpoint ─────────────────────────
    start_epoch = 0
    best_metrics = {}

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metrics = ckpt.get("metrics", {})
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("early_stopping"):
            early_stopping.load_state_dict(ckpt["early_stopping"])
        logger.info(f"Resumed from epoch {start_epoch}")

    if args.eval_only:
        logger.info("Running evaluation only...")
        val_loss, val_metrics, _, _ = validate(
            model, val_loader, criterion, device, label_columns
        )
        report = format_metrics_report(val_metrics, label_columns)
        print(report)
        tracker.log_metrics(val_metrics, step=0, prefix="eval/")
        tracker.close()
        return val_metrics

    # ── Training Loop ──────────────────────────────────
    logger.info(f"Starting training: {num_epochs} epochs")
    logger.info(f"  Batch size: {train_cfg['batch_size']} | AMP: {use_amp} | "
                f"Accum: {accum_steps} | Freeze: {freeze_epochs} epochs")

    training_history = []

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Progressive unfreezing
        if freeze_epochs > 0 and epoch == freeze_epochs:
            logger.info(f"Unfreezing backbone at epoch {epoch}")
            model.unfreeze_backbone()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, accum_steps=accum_steps, clip_grad_norm=clip_grad,
            epoch=epoch, tracker=tracker, log_interval=log_interval,
            use_amp=use_amp,
        )

        # Validate
        val_loss, val_metrics, val_probs, val_labels = validate(
            model, val_loader, criterion, device, label_columns,
            epoch=epoch, tracker=tracker,
        )

        # Step scheduler
        if is_plateau:
            monitor_val = val_metrics.get(
                es_cfg.get("monitor", "val_loss").replace("val_", ""), val_loss
            )
            scheduler.step(monitor_val)
        else:
            scheduler.step()

        # Log epoch metrics
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        epoch_metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_metrics["mean_auroc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "lr": lr,
            "epoch_time": epoch_time,
        }
        training_history.append(epoch_metrics)

        tracker.log_metrics(epoch_metrics, step=epoch + 1, prefix="epoch/")

        # Log per-class AUROC
        tracker.log_metrics(
            val_metrics["auroc_per_class"], step=epoch + 1, prefix="auroc/"
        )

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"AUC: {val_metrics['mean_auroc']:.4f} | "
            f"F1: {val_metrics['macro_f1']:.4f} | "
            f"LR: {lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Save checkpoint
        checkpoint_mgr.save(
            model, optimizer, epoch=epoch, metrics=epoch_metrics,
            scheduler=scheduler, early_stopping=early_stopping,
        )

        best_metrics = val_metrics

        # Early stopping
        if early_stopping is not None:
            monitor_val = val_metrics.get(
                es_cfg["monitor"].replace("val_", ""), val_loss
            )
            if early_stopping.step(monitor_val):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # ── Final Evaluation with Optimal Thresholds ───────
    logger.info("Running final evaluation with optimal thresholds...")
    val_loss, val_metrics, val_probs, val_labels = validate(
        model, val_loader, criterion, device, label_columns
    )

    optimal_thresh = np.array(val_metrics["optimal_thresholds"])
    val_preds_optimal = (val_probs >= optimal_thresh).astype(np.float32)

    from src.metrics import compute_f1_scores
    optimal_f1 = compute_f1_scores(val_labels, val_preds_optimal)
    val_metrics["optimal_threshold_f1"] = optimal_f1
    val_metrics["optimal_thresholds"] = optimal_thresh.tolist()

    report = format_metrics_report(val_metrics, label_columns)
    print("\n" + report)

    # ── Export ──────────────────────────────────────────
    export_cfg = cfg.get("export", {})

    # Load best model for export
    best_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(best_path):
        logger.info("Loading best model for export...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)

    if export_cfg.get("onnx", False):
        onnx_path = os.path.join(output_dir, "model.onnx")
        export_onnx(
            model, onnx_path,
            opset_version=export_cfg.get("onnx_opset", 17),
            dynamic_batch=export_cfg.get("dynamic_batch", True),
            img_size=img_size,
        )

    if export_cfg.get("torchscript", False):
        ts_path = os.path.join(output_dir, "model.torchscript")
        export_torchscript(model, ts_path, img_size=img_size)

    # ── Save Results ───────────────────────────────────
    results = {
        "config_path": str(cfg_path),
        "num_epochs_trained": epoch + 1 if 'epoch' in dir() else num_epochs,
        "final_metrics": {k: v for k, v in val_metrics.items() if not isinstance(v, dict)},
        "auroc_per_class": val_metrics.get("auroc_per_class", {}),
        "pr_auc_per_class": val_metrics.get("pr_auc_per_class", {}),
        "training_history": training_history,
        "optimal_thresholds": optimal_thresh.tolist(),
    }

    # Save JSON results
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save report text
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
        f.write(f"\n\nOptimal thresholds per class:\n")
        for name, thresh in zip(label_columns, optimal_thresh):
            f.write(f"  {name}: {thresh:.3f}\n")

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  - training_results.json")
    logger.info(f"  - evaluation_report.txt")
    logger.info(f"  - best_model.pth")

    tracker.log_metrics(
        {"final_auc": val_metrics["mean_auroc"], "final_macro_f1": val_metrics["macro_f1"]},
        step=num_epochs, prefix="final/",
    )
    tracker.close()

    return val_metrics


if __name__ == "__main__":
    main()

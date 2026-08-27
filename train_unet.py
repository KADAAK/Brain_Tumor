"""
================================================================================
  Brain MRI Tumor Segmentation – Training Script
  train_unet.py
================================================================================

OVERVIEW
--------
This script trains a U-Net model for binary brain tumor segmentation.

DATASET EXPECTED FORMAT
-----------------------
  dataset/
    Images/         <- MRI scans (PNG / JPG)
    Masks/          <- Binary masks with the SAME filename as their image

HOW TO RUN
----------
  # Basic run with all defaults:
  python train_unet.py

  # Custom dataset directory, 80 epochs, batch size 8:
  python train_unet.py --dataset-dir /path/to/dataset --epochs 80 --batch-size 8

  # Use only BCE loss, enable flips:
  python train_unet.py --loss bce --horizontal-flip --vertical-flip

  #python train_unet.py --dataset-dir D:\Brain_Tumor\archive --epochs 10 --batch-size 4 --loss bce_dice

OUTPUT
------
  ml/checkpoints/brain_tumor_unet_best.pth   <- Best model checkpoint
  ml/outputs/loss_graph.png                  <- Training & validation loss curves
  ml/outputs/dice_graph.png                  <- Dice coefficient curves
  ml/outputs/iou_graph.png                   <- IoU curves
  ml/outputs/dataset_validation.json         <- Dataset audit report
  ml/outputs/splits.json                     <- Train/val/test file split record
  ml/outputs/training_history.json           <- Full metrics history

SECTIONS
--------
  1. Imports & reproducibility seed
  2. CLI argument parsing
  3. Dataset validation (checks pairing, corruption, dimension match)
  4. Patient-aware train / validation / test splitting
  5. DataLoaders (augmentation only on training split)
  6. U-Net model, BCE+Dice loss, AdamW optimizer, LR scheduler
  7. Training loop with early stopping and model checkpointing
  8. Plot training curves
"""
from __future__ import annotations

# =============================================================================
# SECTION 1 - Imports & reproducibility
# =============================================================================
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")              # Use non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Internal ML pipeline modules (all inside ml/)
from ml.config import TrainingConfig
from ml.dataset import PairedMRIDataset, split_pairs, validate_dataset
from ml.losses import build_loss
from ml.metrics import segmentation_metrics
from ml.models.unet import UNet


def seed_everything(seed: int) -> None:
    """Fix all random seeds for reproducible results across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# SECTION 2 - CLI arguments
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Parse optional command-line overrides. All have sensible defaults."""
    parser = argparse.ArgumentParser(
        description="Train a U-Net for brain MRI tumor segmentation."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="Root dataset directory containing Images/ and Masks/ sub-folders."
    )
    parser.add_argument("--epochs",      type=int,   default=None, help="Number of training epochs (default: 50).")
    parser.add_argument("--batch-size",  type=int,   default=None, help="Batch size per step (default: 4).")
    parser.add_argument("--lr",          type=float, default=None, help="Initial learning rate (default: 0.001).")
    parser.add_argument("--base-ch",     type=int,   default=None, help="U-Net base channel count (default: 32).")
    parser.add_argument(
        "--loss", type=str, choices=["dice", "bce", "bce_dice"], default=None,
        help="Loss function. 'bce_dice' is the combined default."
    )
    parser.add_argument("--horizontal-flip", action="store_true",
                        help="Allow random horizontal flips during augmentation.")
    parser.add_argument("--vertical-flip",   action="store_true",
                        help="Allow random vertical flips during augmentation.")
    parser.add_argument("--no-augment",  action="store_true",
                        help="Disable all data augmentation.")
    return parser.parse_args()


# =============================================================================
# SECTION 7 - Training / validation epoch helper
# =============================================================================
def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Run one full pass over 'loader'.
    - If optimizer is provided  => training mode (gradients enabled).
    - If optimizer is None      => evaluation mode (no gradients).

    Returns averaged loss + Dice/IoU/Precision/Recall for this epoch.
    """
    training = optimizer is not None
    model.train(training)

    totals: dict[str, float] = {
        "loss": 0.0, "dice": 0.0, "iou": 0.0,
        "precision": 0.0, "recall": 0.0,
    }
    count = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            # Move image and mask tensors to the selected device (CPU / GPU)
            x = batch["image"].to(device)    # shape: (B, 1, H, W)
            y = batch["mask"].to(device)     # shape: (B, 1, H, W), float32 0/1

            # Forward pass: model outputs raw logits (NOT probabilities yet)
            logits = model(x)

            # Compute combined BCE + Dice loss
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics = segmentation_metrics(logits, y, threshold)
            totals["loss"] += loss.item()
            count += 1
            for k, v in metrics.items():
                totals[k] += v

    return {k: v / max(count, 1) for k, v in totals.items()}


# =============================================================================
# SECTION 8 - Plot helper
# =============================================================================
def save_graph(history: dict, output_path: Path, key: str, label: str) -> None:
    """Save a train vs. validation curve for the given metric to a PNG file."""
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history[f"train_{key}"], label="Train",      marker="o", markersize=3)
    plt.plot(history["epoch"], history[f"val_{key}"],   label="Validation", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.title(f"{label} - Training vs. Validation")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [Plot] Saved {output_path.name}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main() -> None:
    args = parse_args()

    # --------------------------------------------------------------------------
    # SECTION 2 (continued) - Build config, applying any CLI overrides
    # --------------------------------------------------------------------------
    config = TrainingConfig()

    # Dataset directory: default falls back to dataset/ at project root
    if args.dataset_dir is not None:
        config.dataset_dir = args.dataset_dir
    else:
        config.dataset_dir = Path(__file__).resolve().parent / "dataset"

    if args.epochs      is not None: config.epochs        = args.epochs
    if args.batch_size  is not None: config.batch_size    = args.batch_size
    if args.lr          is not None: config.learning_rate = args.lr
    if args.base_ch     is not None: config.base_channels = args.base_ch
    if args.loss        is not None: config.loss_name     = args.loss

    if args.no_augment:       config.augment               = False
    if args.horizontal_flip:  config.allow_horizontal_flip = True
    if args.vertical_flip:    config.allow_vertical_flip   = True

    # Reproducibility
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Brain MRI U-Net Training")
    print("=" * 70)
    print(f"  Dataset dir   : {config.dataset_dir}")
    print(f"  Image size    : {config.image_size}")
    print(f"  Epochs        : {config.epochs}  (patience: {config.early_stopping_patience})")
    print(f"  Batch size    : {config.batch_size}")
    print(f"  Loss          : {config.loss_name}")
    print(f"  Learning rate : {config.learning_rate}")
    print(f"  Augmentation  : {'ON' if config.augment else 'OFF'}")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # SECTION 3 - Dataset validation
    #
    # validate_dataset() verifies:
    #   - Images/ and Masks/ directories exist
    #   - Every image has a corresponding mask with the SAME filename stem
    #   - Masks without images are also reported
    #   - Every file can be opened (not corrupted)
    #   - Each image/mask pair has matching pixel dimensions
    #   - Unique pixel values in masks (expects {0} or {0, 255} or {0, 1})
    #   - Patient IDs extracted from filenames to support patient-level splitting
    #   - Writes ml/outputs/dataset_validation.json with a full audit report
    #
    # IMPORTANT: No images are silently discarded. If a problem is found,
    # it raises ValueError listing ALL issues so you can fix the dataset.
    # --------------------------------------------------------------------------
    print("\n[1/5] Validating dataset ...")
    try:
        pairs, report = validate_dataset(config, write_report=True)
    except ValueError as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    print(f"  OK  {report['valid_pairs']} paired images found")
    print(f"  OK  {report['unique_patients']} unique patients detected")
    print(f"  OK  Mask pixel values: {report['mask_values']}")
    if report["errors"]:
        print(f"  WARNING: {len(report['errors'])} issue(s) logged to dataset_validation.json")

    # --------------------------------------------------------------------------
    # SECTION 4 - Patient-aware splitting
    #
    # Why split by patient?
    #   If slices from the same patient appear in both train and test sets,
    #   the model can "memorise" that patient's anatomy - this is data leakage.
    #   Patient-aware splitting GUARANTEES no patient appears in more than one
    #   split, giving an honest estimate of generalisation.
    #
    # How patient IDs are extracted:
    #   - "patient001_slice043.png" -> patient_id = "patient001"
    #   - "image_001.png"           -> patient_id = full stem (one "patient")
    #
    # Split fractions: 70% train / 15% validation / 15% test (all configurable)
    # The split is reproducible (seeded with config.seed).
    # --------------------------------------------------------------------------
    print("\n[2/5] Splitting by patient (train / val / test) ...")
    try:
        splits = split_pairs(pairs, config)
    except ValueError as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    for name, subset in splits.items():
        print(f"  {name:>14}: {len(subset):4d} image-mask pairs")
    print(f"  Split record saved to: {config.output_dir / 'splits.json'}")

    # --------------------------------------------------------------------------
    # SECTION 5 - DataLoaders
    #
    # PairedMRIDataset per sample does:
    #   1. Load image as grayscale float32 numpy array
    #   2. Load mask and binarise: (mask > 0) -> {0, 1}
    #      This handles masks stored as {0,255} or {0,1} or multi-label
    #      WITHOUT modifying any file on disk.
    #   3. Resize image  to (256, 256) with bilinear interpolation
    #   4. Resize mask   to (256, 256) with NEAREST-NEIGHBOR interpolation
    #      (nearest-neighbor preserves binary 0/1 values; bilinear would create
    #       fractional values at edges, corrupting the binary mask)
    #   5. Normalise image: (x - mean) / std  per slice
    #
    # Augmentation (training split ONLY):
    #   - Random rotation +-10 degrees
    #   - Random zoom 0.95 to 1.05
    #   - Optional horizontal / vertical flip (off by default, MRI-safe)
    #   - Random brightness perturbation (30% chance, small magnitude)
    #   CRITICALLY: image and mask receive IDENTICAL transforms so spatial
    #   alignment is preserved. The mask is never augmented independently.
    #
    # Validation and test splits use training=False (no augmentation at all).
    # --------------------------------------------------------------------------
    print("\n[3/5] Building DataLoaders ...")
    train_ds = PairedMRIDataset(splits["train"],      config, training=True)
    val_ds   = PairedMRIDataset(splits["validation"], config, training=False)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True,  num_workers=config.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers)

    print(f"  Train loader : {len(train_loader)} batches x {config.batch_size}")
    print(f"  Val   loader : {len(val_loader)}   batches x {config.batch_size}")

    # --------------------------------------------------------------------------
    # SECTION 6 - Model, loss, optimiser, LR scheduler
    #
    # U-Net architecture:
    #   Encoder (contracting path):
    #     4 x DoubleConv blocks (Conv2d-BN-ReLU-Conv2d-BN-ReLU) + MaxPool2d
    #     Channel widths: 32 -> 64 -> 128 -> 256
    #   Bottleneck:
    #     DoubleConv at 512 channels (deepest representation)
    #   Decoder (expanding path):
    #     4 x ConvTranspose2d (upsample 2x) + skip-connection concat + DoubleConv
    #     Channel widths: 256 -> 128 -> 64 -> 32
    #   Output head:
    #     1x1 Conv -> 1 logit channel (sigmoid applied at inference time)
    #
    # Loss - Combined BCE + Dice (bce_dice):
    #   total_loss = 0.5 * BCE(logits, mask) + 0.5 * Dice(logits, mask)
    #   BCE  gives pixel-level supervision, stabilises early training
    #   Dice optimises region overlap directly, handles class imbalance better
    #
    # Optimiser: AdamW (Adam + weight decay for L2 regularisation)
    # Scheduler: ReduceLROnPlateau - halves LR if val Dice stagnates 3 epochs
    # --------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[4/5] Building model ...")
    print(f"  Device : {device}")

    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    criterion = build_loss(config.loss_name)
    optimizer = AdamW(model.parameters(),
                      lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3,
                                  factor=0.5, verbose=True)

    # --------------------------------------------------------------------------
    # SECTION 7 - Training loop with early stopping and model checkpointing
    #
    # Per epoch:
    #   1. TRAIN pass  : forward -> loss -> backward -> weight update
    #   2. VAL   pass  : forward -> loss and metrics ONLY (no gradient update)
    #   3. Scheduler step based on val Dice
    #   4. If val Dice improved -> save checkpoint (model checkpointing)
    #   5. If val Dice did not improve -> increment stale counter
    #   6. If stale >= patience -> early stopping
    #
    # The VALIDATION and TEST splits are NEVER used during back-propagation.
    # Their masks are loaded only to compute loss/metrics for monitoring.
    # The TEST split is not touched at all during training - it is used only
    # in evaluate.py after training is complete.
    # --------------------------------------------------------------------------
    print(f"\n[5/5] Training for up to {config.epochs} epochs ...\n")

    best_val_dice = -1.0
    stale_epochs  = 0
    best_path     = config.checkpoint_dir / "brain_tumor_unet_best.pth"

    history: dict[str, list] = {
        "epoch": [],
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou":  [], "val_iou":  [],
    }

    for epoch in range(1, config.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, device,
                            optimizer=optimizer, threshold=config.threshold)
        val_m   = run_epoch(model, val_loader,   criterion, device,
                            optimizer=None,      threshold=config.threshold)

        scheduler.step(val_m["dice"])

        history["epoch"].append(epoch)
        for key in ("loss", "dice", "iou"):
            history[f"train_{key}"].append(train_m[key])
            history[f"val_{key}"].append(val_m[key])

        print(
            f"Epoch {epoch:3d}/{config.epochs}  |  "
            f"Train  Loss={train_m['loss']:.4f}  Dice={train_m['dice']:.4f}  |  "
            f"Val  Loss={val_m['loss']:.4f}  Dice={val_m['dice']:.4f}  "
            f"IoU={val_m['iou']:.4f}  Prec={val_m['precision']:.4f}  Rec={val_m['recall']:.4f}"
        )

        # Model checkpointing
        if val_m["dice"] > best_val_dice:
            best_val_dice = val_m["dice"]
            stale_epochs  = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config":           config.as_dict(),
                    "epoch":            epoch,
                    "val_metrics":      val_m,
                },
                best_path,
            )
            print(f"  >> Best model saved (Val Dice={best_val_dice:.4f}) -> {best_path}")
        else:
            stale_epochs += 1
            print(f"     No improvement ({stale_epochs}/{config.early_stopping_patience})")
            if stale_epochs >= config.early_stopping_patience:
                print(f"\n  Early stopping after {epoch} epochs.")
                break

    # --------------------------------------------------------------------------
    # SECTION 8 - Save history and plot training curves
    # --------------------------------------------------------------------------
    (config.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (config.output_dir / "training_config.json").write_text(
        json.dumps(config.as_dict(), indent=2), encoding="utf-8"
    )

    print("\n  Saving training curve plots ...")
    save_graph(history, config.output_dir / "loss_graph.png",  "loss", "Loss")
    save_graph(history, config.output_dir / "dice_graph.png",  "dice", "Dice Coefficient")
    save_graph(history, config.output_dir / "iou_graph.png",   "iou",  "IoU (Jaccard Index)")

    print(f"\n{'=' * 70}")
    print(f"  Training complete.  Best val Dice: {best_val_dice:.4f}")
    print(f"  Checkpoint : {best_path}")
    print(f"  Plots      : {config.output_dir}/")
    print(f"\n  Next steps:")
    print(f"    python evaluate.py          <- evaluate on held-out test set")
    print(f"    python predict.py --image <path/to/image.png>")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

"""
================================================================================
  Brain MRI Tumor Segmentation - Evaluate on Test Set
  evaluate.py
================================================================================

OVERVIEW
--------
Loads the best saved checkpoint and evaluates it on the HELD-OUT TEST split.
The test split was never seen during training or validation.

For each test image this script:
  - Generates the predicted binary mask
  - Saves predicted masks as PNG files to predictions/
  - Creates a 4-panel comparison figure:
      [Original MRI | Ground-Truth Mask | Predicted Mask | MRI + Overlay]
  - Computes and prints: Dice, IoU, Precision, Recall
  - Writes ml/outputs/test_metrics.json

HOW TO RUN
----------
  # Default: uses the checkpoint saved by train_unet.py
  python evaluate.py

  # Specify a different checkpoint:
  python evaluate.py --checkpoint ml/checkpoints/brain_tumor_unet_best.pth

  # Specify a different dataset directory:
  python evaluate.py --dataset-dir /path/to/dataset
"""
from __future__ import annotations

# =============================================================================
# SECTION 1 - Imports
# =============================================================================
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ml.config import TrainingConfig
from ml.dataset import PairedMRIDataset, split_pairs, validate_dataset
from ml.losses import build_loss
from ml.metrics import segmentation_metrics
from ml.models.unet import UNet


# =============================================================================
# SECTION 2 - CLI arguments
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the best U-Net checkpoint on the held-out test set."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to the .pth checkpoint. Defaults to ml/checkpoints/brain_tumor_unet_best.pth"
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="Root dataset directory (with Images/ and Masks/ subdirs)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to save comparison plots and metrics."
    )
    parser.add_argument(
        "--pred-dir", type=Path, default=None,
        help="Where to save predicted mask PNG files."
    )
    parser.add_argument(
        "--max-vis", type=int, default=20,
        help="Maximum number of comparison images to save (default: 20)."
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args = parse_args()

    # --------------------------------------------------------------------------
    # SECTION 3 - Load checkpoint and restore config
    #
    # The checkpoint stores the model weights AND the TrainingConfig used during
    # training, so image_size, base_channels, threshold etc. are all restored
    # automatically. This ensures the evaluation uses the exact same settings
    # as training.
    # --------------------------------------------------------------------------
    config = TrainingConfig()

    if args.dataset_dir is not None:
        config.dataset_dir = args.dataset_dir
    else:
        config.dataset_dir = Path(__file__).resolve().parent / "dataset"

    checkpoint_path = args.checkpoint or (config.checkpoint_dir / "brain_tumor_unet_best.pth")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("  Run  python train_unet.py  first.")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Restore hyperparameters that affect model architecture / inference
    saved_cfg = checkpoint.get("config", {})
    for name in ("image_size", "in_channels", "out_channels", "base_channels",
                 "threshold", "loss_name"):
        if name in saved_cfg:
            value = saved_cfg[name]
            setattr(config, name, tuple(value) if name == "image_size" else value)

    trained_epoch = checkpoint.get("epoch", "?")
    saved_val     = checkpoint.get("val_metrics", {})
    print(f"  Checkpoint from epoch {trained_epoch},  "
          f"val Dice={saved_val.get('dice', '?'):.4f}  "
          f"val IoU={saved_val.get('iou', '?'):.4f}")

    output_dir = args.output_dir or config.output_dir
    pred_dir   = args.pred_dir   or (output_dir / "predictions")
    vis_dir    = output_dir / "test_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # SECTION 4 - Dataset validation and test split
    #
    # We re-run validate_dataset + split_pairs using the SAME seed as training
    # so the test split is IDENTICAL to the one held out during training.
    # No test images were ever used during back-propagation.
    # --------------------------------------------------------------------------
    print("\nValidating dataset and extracting test split ...")
    try:
        pairs, _ = validate_dataset(config, write_report=False)
        splits   = split_pairs(pairs, config)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    test_pairs = splits["test"]
    print(f"  Test split : {len(test_pairs)} images")

    if not test_pairs:
        print("ERROR: Test split is empty.")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # SECTION 5 - Build model and load weights
    #
    # Loads the architecture with the same in/out/base channel counts as training
    # then restores the saved weights. model.eval() disables dropout/batchnorm
    # training-mode behaviour.
    # --------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nBuilding model on {device} ...")
    model = UNet(config.in_channels, config.out_channels, config.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --------------------------------------------------------------------------
    # SECTION 6 - DataLoader for test split (NO augmentation, shuffle=False)
    # --------------------------------------------------------------------------
    test_ds     = PairedMRIDataset(test_pairs, config, training=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=config.num_workers)

    criterion = build_loss(config.loss_name)

    # --------------------------------------------------------------------------
    # SECTION 7 - Evaluation loop
    #
    # For each test image:
    #   1. Forward pass through the model to get logits
    #   2. Apply sigmoid + threshold to produce the binary mask
    #   3. Compute Dice, IoU, Precision, Recall
    #   4. Save predicted mask as PNG to predictions/
    #   5. Create a 4-panel comparison figure and save to test_comparisons/
    #
    # SECTION 19/20/21 requirements are all fulfilled here.
    # --------------------------------------------------------------------------
    print("\nRunning evaluation on test set ...\n")
    totals = {k: 0.0 for k in ("loss", "dice", "iou", "precision", "recall")}

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            x    = batch["image"].to(device)   # (1, 1, H, W)
            y    = batch["mask"].to(device)    # (1, 1, H, W)
            name = Path(batch["name"][0]).stem

            logits  = model(x)
            totals["loss"] += criterion(logits, y).item()

            m = segmentation_metrics(logits, y, config.threshold)
            for k, v in m.items():
                totals[k] += v

            # ---- Predicted binary mask (numpy, 2D) ----
            pred_prob = torch.sigmoid(logits)[0, 0].cpu().numpy()   # float [0,1]
            pred_mask = (pred_prob >= config.threshold)             # bool

            # ---- Save predicted mask as PNG ----
            # Multiply by 255 so the file is a standard 8-bit binary mask image
            Image.fromarray((pred_mask.astype(np.uint8) * 255)).save(
                pred_dir / f"{name}_pred_mask.png"
            )

            # ---- 4-panel comparison (requirements 21) ----
            if idx < args.max_vis:
                image_np  = x[0, 0].cpu().numpy()       # normalised float
                truth_np  = y[0, 0].cpu().numpy()       # binary 0/1 float

                # Overlay: MRI (grey) with predicted tumor region coloured red
                img_norm  = (image_np - image_np.min()) / max(image_np.max() - image_np.min(), 1e-6)
                overlay   = np.stack([img_norm, img_norm, img_norm], axis=-1)
                overlay_c = overlay.copy()
                overlay_c[pred_mask] = [1.0, 0.0, 0.0]   # red = predicted tumor

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                panels = [
                    (image_np,   "gray", "Original MRI"),
                    (truth_np,   "gray", "Ground-Truth Mask"),
                    (pred_mask,  "gray", "Predicted Mask"),
                    (overlay_c,  None,   "MRI + Predicted Overlay"),
                ]
                for ax, (data, cmap, title) in zip(axes, panels):
                    ax.imshow(data, cmap=cmap, vmin=0, vmax=1 if data.ndim == 2 else None)
                    ax.set_title(
                        f"{title}\nDice={m['dice']:.3f}  IoU={m['iou']:.3f}",
                        fontsize=9
                    )
                    ax.axis("off")

                fig.tight_layout()
                fig.savefig(vis_dir / f"{name}_comparison.png", dpi=150)
                plt.close(fig)

            print(
                f"  [{idx + 1:3d}/{len(test_loader)}] {name:<30}  "
                f"Dice={m['dice']:.4f}  IoU={m['iou']:.4f}  "
                f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}"
            )

    # --------------------------------------------------------------------------
    # SECTION 8 - Aggregate results
    # --------------------------------------------------------------------------
    n = max(len(test_loader), 1)
    results = {k: v / n for k, v in totals.items()}

    print(f"\n{'=' * 60}")
    print(f"  TEST SET RESULTS  ({len(test_loader)} images)")
    print(f"{'=' * 60}")
    print(f"  Loss      : {results['loss']:.4f}")
    print(f"  Dice      : {results['dice']:.4f}")
    print(f"  IoU       : {results['iou']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"{'=' * 60}")
    print(f"  Predicted masks  -> {pred_dir}/")
    print(f"  Comparison plots -> {vis_dir}/")

    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Metrics JSON     -> {metrics_path}")


if __name__ == "__main__":
    main()

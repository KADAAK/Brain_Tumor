"""
================================================================================
  Brain MRI Tumor Segmentation - Single Image Prediction
  predict.py
================================================================================

OVERVIEW
--------
Run the trained U-Net on one or more images and save:
  - Predicted binary mask (PNG)
  - MRI + predicted tumor overlay (PNG)
  - A JSON analysis file with tumor count, area, etc.

HOW TO RUN
----------
  # Predict on a single image:
  python predict.py --image path/to/mri_scan.png

  # Predict on all images in a directory:
  python predict.py --image path/to/Images/

  # Use a specific checkpoint and custom output directory:
  python predict.py --image scan.png --checkpoint ml/checkpoints/best.pth --output-dir predictions/

OUTPUT per image (e.g. for mri_001.png):
  predictions/mri_001_pred_mask.png       <- binary mask (0=background, 255=tumor)
  predictions/mri_001_overlay.png         <- RGB overlay with red tumor contour
  predictions/mri_001_visualization.png   <- 3-panel figure (original / mask / overlay)
  predictions/mri_001_analysis.json       <- tumor count, area in pixels, etc.
"""
from __future__ import annotations

# =============================================================================
# SECTION 1 - Imports
# =============================================================================
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ml.config import TrainingConfig
from ml.inference.predictor import UNetPredictor
from ml.postprocessing import analyze_mask


# =============================================================================
# SECTION 2 - CLI arguments
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run U-Net inference on a single MRI image or a directory of images."
    )
    parser.add_argument(
        "--image", type=Path, required=True,
        help="Path to an MRI image file (PNG/JPG) OR a directory of images."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to the .pth checkpoint (defaults to ml/checkpoints/brain_tumor_unet_best.pth)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to save outputs (default: predictions/)."
    )
    return parser.parse_args()


# =============================================================================
# SECTION 3 - Predict on a single image
# =============================================================================
def predict_single(
    image_path: Path,
    predictor: UNetPredictor,
    output_dir: Path,
) -> dict:
    """
    Run inference on one MRI image.

    Steps:
      1. Load image as grayscale (to match training input format)
      2. Run UNetPredictor.predict():
           - Converts to float32
           - Resizes to the training image_size with bilinear interpolation
           - Z-score normalises per-image: (x - mean) / std
           - Runs through U-Net, applies sigmoid
           - Thresholds at config.threshold (default 0.5)
           - Resizes mask back to original image resolution with NEAREST-NEIGHBOR
      3. Save:
           - Binary mask PNG (0 / 255)
           - Contour overlay RGB PNG
           - 3-panel visualization PNG
           - JSON analysis (tumor count, area, bounding boxes, etc.)
    """
    # Load original image as numpy array
    image_np = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    stem     = image_path.stem

    # --------------------------------------------------------------------------
    # Run inference
    # The predict() method handles all preprocessing internally and returns
    # a boolean numpy array of shape (H, W) at the original image resolution.
    # --------------------------------------------------------------------------
    pred_mask = predictor.predict(image_np)   # bool ndarray, shape (H, W)

    # --------------------------------------------------------------------------
    # Save binary mask PNG
    # Convention: 255 = tumor, 0 = background (standard 8-bit grayscale mask)
    # --------------------------------------------------------------------------
    mask_uint8 = (pred_mask.astype(np.uint8) * 255)
    Image.fromarray(mask_uint8).save(output_dir / f"{stem}_pred_mask.png")

    # --------------------------------------------------------------------------
    # Create overlay: draw red contours of predicted tumor on the original MRI
    # --------------------------------------------------------------------------
    base_rgb  = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(
        pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    overlay_rgb = base_rgb.copy()
    cv2.drawContours(overlay_rgb, contours, -1, (255, 0, 0), 2)
    Image.fromarray(overlay_rgb).save(output_dir / f"{stem}_overlay.png")

    # --------------------------------------------------------------------------
    # 3-panel visualization: Original | Predicted Mask | Overlay
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np, cmap="gray");    axes[0].set_title("Original MRI");      axes[0].axis("off")
    axes[1].imshow(mask_uint8, cmap="gray");  axes[1].set_title("Predicted Mask");    axes[1].axis("off")
    axes[2].imshow(overlay_rgb);              axes[2].set_title("MRI + Tumor Overlay"); axes[2].axis("off")
    fig.suptitle(f"Prediction: {image_path.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_visualization.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # JSON analysis: connected components, tumor count, pixel areas, bounding boxes
    # --------------------------------------------------------------------------
    analysis = analyze_mask(pred_mask, spacing=None)
    analysis_path = output_dir / f"{stem}_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    return {
        "image":        str(image_path),
        "mask":         str(output_dir / f"{stem}_pred_mask.png"),
        "overlay":      str(output_dir / f"{stem}_overlay.png"),
        "visualization": str(output_dir / f"{stem}_visualization.png"),
        "tumor_count":  analysis["tumor_count"],
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main() -> None:
    args = parse_args()

    # Default checkpoint path
    default_ckpt = TrainingConfig().checkpoint_dir / "brain_tumor_unet_best.pth"
    checkpoint   = args.checkpoint or default_ckpt

    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        print("  Run  python train_unet.py  first to produce a checkpoint.")
        sys.exit(1)

    # Output directory
    output_dir = args.output_dir or (Path(__file__).resolve().parent / "predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # Load trained model
    # UNetPredictor reads the checkpoint, restores the TrainingConfig
    # (image_size, threshold, etc.) and loads model weights.
    # --------------------------------------------------------------------------
    print(f"Loading model from: {checkpoint}")
    predictor = UNetPredictor(checkpoint)

    # --------------------------------------------------------------------------
    # Collect images to process
    # Supports a single file OR a directory of images.
    # --------------------------------------------------------------------------
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if args.image.is_dir():
        image_paths = sorted(
            p for p in args.image.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            print(f"ERROR: No supported image files found in {args.image}")
            sys.exit(1)
        print(f"Found {len(image_paths)} images in {args.image}")
    elif args.image.is_file():
        image_paths = [args.image]
    else:
        print(f"ERROR: --image path does not exist: {args.image}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Run inference on all collected images
    # --------------------------------------------------------------------------
    results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {img_path.name} ...", end=" ")
        try:
            info = predict_single(img_path, predictor, output_dir)
            print(f"Tumors detected: {info['tumor_count']}")
            results.append(info)
        except Exception as exc:
            print(f"FAILED: {exc}")

    # Save a combined summary JSON
    summary_path = output_dir / "predictions_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n  Predictions saved to: {output_dir}/")
    print(f"  Summary            : {summary_path}")
    print(f"  Total images       : {len(results)}")


if __name__ == "__main__":
    main()

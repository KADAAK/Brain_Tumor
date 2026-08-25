from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from backend.utils.image_utils import normalize_uint8


class VisualizationService:
    def generate(self, study_id: str, image, mask, components, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        base = normalize_uint8(image)
        original = output_dir / f"{study_id}_original.png"
        Image.fromarray(base).save(original)

        segmentation = output_dir / f"{study_id}_mask.png"
        Image.fromarray((mask.astype(np.uint8) * 255)).save(segmentation)

        # 1. Base RGB canvas
        rgb = np.dstack([base, base, base])
        overlay_mask = (mask > 0).astype(np.uint8)

        # 2. Semi-transparent red highlight fill inside tumor (matches Reference Image 2)
        color_fill = np.zeros_like(rgb)
        color_fill[:] = [230, 45, 45]  # Red tint

        # Blend: 62% original MRI + 38% red tint over the tumor
        alpha = 0.38
        blended = np.where(
            overlay_mask[:, :, None] > 0,
            np.clip((1.0 - alpha) * rgb + alpha * color_fill, 0, 255).astype(np.uint8),
            rgb,
        )

        # 3. Outer boundary contours with bright crisp red outline
        contours, _ = cv2.findContours(overlay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (245, 50, 50), 2)

        annotated = Image.fromarray(blended)
        draw = ImageDraw.Draw(annotated)

        # Draw clean ID tags for single slices / isolated lesions
        is_multi_panel = len(components) >= 4
        crops = []
        for item in components:
            if not is_multi_panel:
                y, x = item.centroid[-2:]
                draw.text(
                    (x + 4, y + 4),
                    item.tumor_id,
                    fill=(255, 255, 255),
                    stroke_width=2,
                    stroke_fill=(200, 20, 20),
                )
            minr, minc, maxr, maxc = item.bbox[-4:]
            crop = annotated.crop(
                (
                    max(0, minc - 8),
                    max(0, minr - 8),
                    min(base.shape[1], maxc + 8),
                    min(base.shape[0], maxr + 8),
                )
            )
            cp = output_dir / f"{study_id}_{item.tumor_id}.png"
            crop.save(cp)
            crops.append(cp)

        annotated_path = output_dir / f"{study_id}_annotated.png"
        annotated.save(annotated_path)
        return {
            "original": original,
            "segmentation": segmentation,
            "annotated": annotated_path,
            "crops": crops,
        }

import cv2
import numpy as np


def mock_segment(image: np.ndarray) -> np.ndarray:
    """Deterministic demo mask based on unusually bright regions; not clinical inference."""
    source = np.asarray(image)
    if source.ndim == 3:
        source = source[source.shape[0] // 2]
    threshold = max(float(np.percentile(source, 92)), float(source.mean() + source.std()))
    mask = (source >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask

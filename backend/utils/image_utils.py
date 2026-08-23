import numpy as np
from PIL import Image


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) * 255 / (high - low), 0, 255).astype(np.uint8)


def save_grayscale(image: np.ndarray, path) -> None:
    Image.fromarray(normalize_uint8(image)).save(path)

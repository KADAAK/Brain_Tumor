from dataclasses import dataclass
import numpy as np
from backend.utils.image_utils import normalize_uint8


@dataclass
class PreprocessedImage:
    array: np.ndarray
    spacing: tuple[float, ...] | None = None
    is_3d: bool = False


def preprocess(image: np.ndarray, spacing: tuple[float, ...] | None = None) -> PreprocessedImage:
    """Placeholder pipeline: finite-value cleanup and display-scale normalization."""
    array = np.nan_to_num(np.asarray(image, dtype=np.float32))
    return PreprocessedImage(array=normalize_uint8(array), spacing=spacing, is_3d=array.ndim == 3)

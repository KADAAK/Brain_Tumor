from dataclasses import dataclass
import numpy as np
from skimage.measure import label, regionprops


@dataclass
class TumorComponent:
    tumor_id: str
    mask: np.ndarray
    bbox: tuple[int, ...]
    centroid: tuple[float, ...]


def detect_components(mask: np.ndarray, min_size: int = 20) -> list[TumorComponent]:
    labels = label(mask.astype(bool), connectivity=mask.ndim)
    components = []
    for region in regionprops(labels):
        if region.area < min_size:
            continue
        components.append(TumorComponent(f"T{len(components) + 1}", labels == region.label,
                                         tuple(int(v) for v in region.bbox), tuple(float(v) for v in region.centroid)))
    return components

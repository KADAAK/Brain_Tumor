import math
import numpy as np
from skimage.measure import regionprops


def measure_component(mask: np.ndarray, spacing: tuple[float, ...] | None = None) -> dict:
    prop = regionprops(mask.astype(np.uint8))[0]
    bbox = tuple(int(x) for x in prop.bbox)
    if mask.ndim == 2:
        minr, minc, maxr, maxc = bbox
        height, width, depth = maxr-minr, maxc-minc, 1
    else:
        mind, minr, minc, maxd, maxr, maxc = bbox
        depth, height, width = maxd-mind, maxr-minr, maxc-minc
    area = int(mask.sum())
    values = {"area_pixels": area, "volume_voxels": area, "width_pixels": width, "height_pixels": height,
              "depth_voxels": depth, "max_diameter_pixels": float(prop.feret_diameter_max if mask.ndim == 2 else max(width, height, depth)),
              "equivalent_radius_pixels": float(math.sqrt(area / math.pi)), "bbox": list(bbox), "centroid": list(map(float, prop.centroid))}
    values["physical_available"] = spacing is not None and len(spacing) >= mask.ndim
    if values["physical_available"]:
        sp = spacing[-mask.ndim:]
        if mask.ndim == 2:
            values["area_mm2"] = area * sp[0] * sp[1]
            values["dimensions_mm"] = [height * sp[0], width * sp[1]]
            values["max_diameter_mm"] = values["max_diameter_pixels"] * max(sp)
        else:
            values["volume_mm3"] = area * float(np.prod(sp))
            values["dimensions_mm"] = [depth * sp[0], height * sp[1], width * sp[2]]
            values["max_diameter_mm"] = values["max_diameter_pixels"] * max(sp)
    return values

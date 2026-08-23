import itertools
import numpy as np
from scipy.spatial.distance import cdist


def relative_position(a: tuple[float, ...], b: tuple[float, ...]) -> str:
    labels = ("superior/inferior", "anterior/posterior", "left/right") if len(a) == 3 else ("superior/inferior", "left/right")
    parts = []
    for i, label in enumerate(labels):
        if abs(b[i] - a[i]) > 1:
            halves = label.split("/")
            parts.append(halves[1] if b[i] > a[i] else halves[0])
    return ", ".join(parts) or "overlapping centroids"


def pairwise_analysis(components, spacing: tuple[float, ...] | None = None) -> list[dict]:
    output = []
    for a, b in itertools.combinations(components, 2):
        ca, cb = np.array(a.centroid), np.array(b.centroid)
        centroid_px = float(np.linalg.norm(ca-cb))
        points_a, points_b = np.argwhere(a.mask), np.argwhere(b.mask)
        boundary_px = float(cdist(points_a, points_b).min())
        item = {"tumor_a": a.tumor_id, "tumor_b": b.tumor_id, "centroid_distance_pixels": centroid_px,
                "boundary_distance_pixels": boundary_px, "relative_position": relative_position(a.centroid, b.centroid)}
        if spacing and len(spacing) >= a.mask.ndim:
            scale = np.asarray(spacing[-a.mask.ndim:])
            item["centroid_distance_mm"] = float(np.linalg.norm((ca-cb)*scale))
            # Uses physically scaled coordinates for anisotropic images.
            item["boundary_distance_mm"] = float(cdist(points_a*scale, points_b*scale).min())
        output.append(item)
    return output

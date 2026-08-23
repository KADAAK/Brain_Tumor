import numpy as np
from processing.connected_components import detect_components
from processing.spatial_analysis import pairwise_analysis


def test_connected_components_and_pairwise_distance():
    mask=np.zeros((30,30),bool); mask[2:7,2:7]=True; mask[18:23,18:23]=True
    parts=detect_components(mask,min_size=1)
    assert [p.tumor_id for p in parts] == ["T1","T2"]
    pairs=pairwise_analysis(parts,(1.0,1.0))
    assert len(pairs)==1
    assert pairs[0]["boundary_distance_pixels"] > 0
    assert pairs[0]["centroid_distance_mm"] > 0

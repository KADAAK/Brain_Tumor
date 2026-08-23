"""Region and inter-tumoral-gap measurements from a predicted binary mask."""
from __future__ import annotations
import numpy as np
from processing.connected_components import detect_components
from processing.measurements import measure_component
from processing.spatial_analysis import pairwise_analysis
def analyze_mask(mask:np.ndarray,spacing:tuple[float,...]|None=None,min_size:int=20)->dict:
    components=detect_components(mask,min_size); tumors=[]
    for component in components:
        item=measure_component(component.mask,spacing); item["tumor_id"]=component.tumor_id; item["physical_measurements_note"]=None if item["physical_available"] else "Physical measurements are unavailable without pixel/voxel spacing; values are reported in pixels."
        tumors.append(item)
    pairs=pairwise_analysis(components,spacing)
    for pair in pairs: pair["inter_tumoral_gap_pixels"]=pair["boundary_distance_pixels"]; pair["inter_tumoral_gap_mm"]=pair.get("boundary_distance_mm")
    return {"tumor_count":len(tumors),"tumors":tumors,"inter_tumoral_separation":pairs}

from backend.models.schemas import PairwiseResult, PhysicalMeasurement
from processing.spatial_analysis import pairwise_analysis


class DistanceService:
    def analyze(self, components, spacing=None) -> list[PairwiseResult]:
        rows=[]
        for value in pairwise_analysis(components, spacing):
            physical="centroid_distance_mm" in value
            note=None if physical else "Physical distance unavailable: image spacing metadata was not supplied."
            rows.append(PairwiseResult(**{k:v for k,v in value.items() if not k.endswith("_mm")},
                centroid_distance=PhysicalMeasurement(value=value.get("centroid_distance_mm"), unit="mm" if physical else None, available=physical, note=note),
                boundary_distance=PhysicalMeasurement(value=value.get("boundary_distance_mm"), unit="mm" if physical else None, available=physical, note=note)))
        return rows

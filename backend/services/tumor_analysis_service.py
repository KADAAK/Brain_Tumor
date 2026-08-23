from backend.models.schemas import PhysicalMeasurement, TumorResult
from processing.measurements import measure_component


def unavailable(note="Physical measurement unavailable: image spacing metadata was not supplied."):
    return PhysicalMeasurement(available=False, note=note)


class TumorAnalysisService:
    def analyze(self, components, spacing=None) -> list[TumorResult]:
        tumors=[]
        for component in components:
            m=measure_component(component.mask, spacing)
            physical=m["physical_available"]
            dims = m.get("dimensions_mm", [])
            tumors.append(TumorResult(tumor_id=component.tumor_id, bbox=m["bbox"], centroid=m["centroid"],
                area_pixels=m["area_pixels"], volume_voxels=m["volume_voxels"], width_pixels=m["width_pixels"], height_pixels=m["height_pixels"], depth_voxels=m["depth_voxels"], max_diameter_pixels=m["max_diameter_pixels"], equivalent_radius_pixels=m["equivalent_radius_pixels"], physical_measurements_available=physical,
                area=PhysicalMeasurement(value=m.get("area_mm2"), unit="mm²" if physical else None, available=physical, note=None if physical else unavailable().note),
                volume=PhysicalMeasurement(value=m.get("volume_mm3"), unit="mm³" if "volume_mm3" in m else None, available="volume_mm3" in m, note=None if "volume_mm3" in m else unavailable().note),
                dimensions={"height": PhysicalMeasurement(value=dims[-2] if physical else None, unit="mm" if physical else None, available=physical, note=None if physical else unavailable().note), "width": PhysicalMeasurement(value=dims[-1] if physical else None, unit="mm" if physical else None, available=physical, note=None if physical else unavailable().note), "depth": PhysicalMeasurement(value=dims[0] if len(dims)==3 else None, unit="mm" if len(dims)==3 else None, available=len(dims)==3, note=None if len(dims)==3 else unavailable().note)},
                maximum_diameter=PhysicalMeasurement(value=m.get("max_diameter_mm"), unit="mm" if physical else None, available=physical, note=None if physical else unavailable().note)))
        return tumors

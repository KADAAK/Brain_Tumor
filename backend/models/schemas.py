from typing import Any, Literal
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    study_id: str
    filename: str
    content_type: str | None = None
    message: str


class PhysicalMeasurement(BaseModel):
    value: float | None = None
    unit: str | None = None
    available: bool
    note: str | None = None


class TumorResult(BaseModel):
    tumor_id: str
    bbox: list[int]
    centroid: list[float]
    area_pixels: int
    volume_voxels: int
    width_pixels: int
    height_pixels: int
    depth_voxels: int = 1
    max_diameter_pixels: float
    equivalent_radius_pixels: float
    physical_measurements_available: bool
    area: PhysicalMeasurement
    volume: PhysicalMeasurement
    dimensions: dict[str, PhysicalMeasurement]
    maximum_diameter: PhysicalMeasurement
    crop_url: str | None = None


class PairwiseResult(BaseModel):
    tumor_a: str
    tumor_b: str
    centroid_distance_pixels: float
    boundary_distance_pixels: float
    centroid_distance: PhysicalMeasurement
    boundary_distance: PhysicalMeasurement
    relative_position: str


class ModelInfo(BaseModel):
    name: str = "mock"
    version: str = "0.1"
    status: Literal["mock", "loaded", "unavailable"] = "mock"


class AnalysisResult(BaseModel):
    study_id: str
    tumor_count: int
    tumors: list[TumorResult] = Field(default_factory=list)
    pairwise_analysis: list[PairwiseResult] = Field(default_factory=list)
    model: ModelInfo = Field(default_factory=ModelInfo)
    image_shape: list[int]
    voxel_spacing: list[float] | None = None
    original_image_url: str | None = None
    segmentation_url: str | None = None
    annotated_image_url: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReportResponse(BaseModel):
    study_id: str
    format: Literal["pdf", "docx"]
    download_url: str

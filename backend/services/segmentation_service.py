import numpy as np
from ml.inference.predictor import MockSegmentationModel
from ml.models.unet import BaseSegmentationModel


class SegmentationService:
    """Single replacement point for mock, U-Net, Attention U-Net, or U-Net++ models."""
    def __init__(self, model: BaseSegmentationModel | None = None):
        self.model = model or MockSegmentationModel()
    def predict(self, image: np.ndarray) -> np.ndarray:
        return self.model.predict(image).astype(bool)
    @property
    def model_info(self) -> dict:
        return {"name": "mock", "version": "0.1", "status": "mock"}

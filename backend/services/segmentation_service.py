from pathlib import Path
import numpy as np
from backend.config import settings
from ml.inference.predictor import MockSegmentationModel, UNetPredictor
from ml.models.unet import BaseSegmentationModel


class SegmentationService:
    """Single replacement point for mock, U-Net, Attention U-Net, or U-Net++ models."""

    def __init__(self, model: BaseSegmentationModel | None = None):
        if model is not None:
            self.model = model
        elif settings.model_path and Path(settings.model_path).is_file():
            self.model = UNetPredictor(settings.model_path)
        else:
            self.model = MockSegmentationModel()

    def predict(self, image: np.ndarray) -> np.ndarray:
        return self.model.predict(image).astype(bool)

    @property
    def model_info(self) -> dict:
        if isinstance(self.model, UNetPredictor):
            return {"name": "unet", "version": "1.0", "status": "loaded"}
        return {"name": getattr(self.model, "name", "mock"), "version": getattr(self.model, "version", "0.1"), "status": getattr(self.model, "status", "mock")}

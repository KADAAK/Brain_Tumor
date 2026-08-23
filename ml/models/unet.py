from abc import ABC, abstractmethod
import numpy as np


class BaseSegmentationModel(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray: ...


class UNetModel(BaseSegmentationModel):
    """Integration placeholder. Load trained weights here when available."""
    def __init__(self, weights_path: str): self.weights_path = weights_path
    def predict(self, image: np.ndarray) -> np.ndarray:
        raise RuntimeError("A trained U-Net is not configured. Use the mock model until weights are integrated.")

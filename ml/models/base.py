"""Abstract base contract for all segmentation models — torch-free."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseSegmentationModel(ABC):
    """Small contract used by the web application; implementations return a boolean mask."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray: ...


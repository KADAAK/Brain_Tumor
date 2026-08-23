import numpy as np
from ml.models.unet import BaseSegmentationModel
from processing.segmentation import mock_segment


class MockSegmentationModel(BaseSegmentationModel):
    def predict(self, image: np.ndarray) -> np.ndarray:
        return mock_segment(image)

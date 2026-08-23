from ml.models.unet import BaseSegmentationModel


class AttentionUNetModel(BaseSegmentationModel):
    """Future Attention U-Net adapter."""

    def predict(self, image):
        raise NotImplementedError("Attention U-Net predictor requires model weights checkpoint.")

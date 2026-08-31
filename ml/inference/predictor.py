"""Reusable U-Net inference interface for CLI and FastAPI."""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np
from ml.models.unet import BaseSegmentationModel
from processing.segmentation import mock_segment
class MockSegmentationModel(BaseSegmentationModel):
    """Existing non-clinical fallback used until an explicit checkpoint is configured."""
    name = "mock"; version = "0.1"; status = "mock"
    def predict(self, image: np.ndarray) -> np.ndarray: return mock_segment(image)

class UNetPredictor(BaseSegmentationModel):
    def __init__(self, checkpoint_path: str|Path, device: str|None=None):
        import torch  # lazy import — avoids OOM crash on Render free tier at module load
        from ml.config import TrainingConfig
        from ml.models.unet import UNet
        self.path=Path(checkpoint_path); self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not self.path.is_file(): raise FileNotFoundError(f"U-Net checkpoint not found: {self.path}")
        checkpoint=torch.load(self.path,map_location=self.device,weights_only=False); saved=checkpoint.get("config",{}); self.config=TrainingConfig()
        for name in ("image_size","in_channels","out_channels","base_channels","threshold"):
            if name in saved: setattr(self.config,name,tuple(saved[name]) if name=="image_size" else saved[name])
        self.model=UNet(self.config.in_channels,self.config.out_channels,self.config.base_channels).to(self.device); self.model.load_state_dict(checkpoint["model_state_dict"]); self.model.eval(); self.metadata={"epoch":checkpoint.get("epoch"),"val_metrics":checkpoint.get("val_metrics",{})}
    def predict(self,image:np.ndarray)->np.ndarray:
        import torch  # lazy import
        source=np.asarray(image); source=source.mean(axis=-1) if source.ndim==3 else source; original_shape=source.shape
        resized=cv2.resize(source.astype(np.float32),(self.config.image_size[1],self.config.image_size[0]),interpolation=cv2.INTER_LINEAR); normalized=(resized-resized.mean())/max(float(resized.std()),1e-6)
        tensor=torch.from_numpy(normalized[None,None]).float().to(self.device)
        with torch.no_grad(): mask=(torch.sigmoid(self.model(tensor))[0,0].cpu().numpy()>=self.config.threshold).astype(np.uint8)
        return cv2.resize(mask,(original_shape[1],original_shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)

def predict(image:np.ndarray,checkpoint_path:str|Path)->np.ndarray: return UNetPredictor(checkpoint_path).predict(image)

from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image, UnidentifiedImageError
from backend.utils.file_utils import suffix_for


class ImageLoadError(ValueError): pass


class ImageService:
    def load(self, path: Path) -> tuple[np.ndarray, tuple[float, ...] | None]:
        suffix = suffix_for(path.name)
        try:
            if suffix in {".nii", ".nii.gz"}:
                nii = nib.load(str(path))
                volume = nii.get_fdata()
                # A full 3D inference adapter can consume volume later. The mock displays middle slice.
                image = volume[..., volume.shape[-1] // 2] if volume.ndim == 3 else volume
                return np.asarray(image), tuple(float(v) for v in nii.header.get_zooms()[:image.ndim])
            with Image.open(path) as img:
                return np.asarray(img.convert("L")), None
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageLoadError("The image could not be read. It may be corrupted or unsupported.") from exc

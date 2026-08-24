"""Strict paired-image validation, patient-aware splits, preprocessing and augmentation."""
from __future__ import annotations
import json, random, re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from ml.config import TrainingConfig

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
@dataclass(frozen=True)
class ImageMaskPair: image_path: Path; mask_path: Path; patient_id: str

def resolve_dataset_paths(config: TrainingConfig) -> tuple[Path, Path]:
    images, masks = config.dataset_dir/config.image_dir_name, config.dataset_dir/config.mask_dir_name
    # Compatibility only: canonical lowercase folders take priority.
    canonical_has_files = images.is_dir() and masks.is_dir() and any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in images.iterdir()) and any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in masks.iterdir())
    return (images, masks) if canonical_has_files else (config.dataset_dir / "Images", config.dataset_dir / "Masks")

def validate_dataset(config: TrainingConfig, write_report: bool = True) -> tuple[list[ImageMaskPair], dict]:
    images_dir, masks_dir = resolve_dataset_paths(config); errors: list[str] = []
    if not images_dir.is_dir(): errors.append(f"Images directory does not exist: {images_dir}")
    if not masks_dir.is_dir(): errors.append(f"Masks directory does not exist: {masks_dir}")
    if errors: raise ValueError("\n".join(errors))
    images = {p.stem:p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}; masks = {p.stem:p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    if images.keys()-masks.keys(): errors.append("Missing masks for: " + ", ".join(sorted(images.keys()-masks.keys())))
    if masks.keys()-images.keys(): errors.append("Masks without images: " + ", ".join(sorted(masks.keys()-images.keys())))
    pairs=[]; dimensions=Counter(); values=set()
    for stem in sorted(images.keys() & masks.keys()):
        try:
            with Image.open(images[stem]) as im, Image.open(masks[stem]) as ma: im.verify(); ma.verify()
            with Image.open(images[stem]) as im, Image.open(masks[stem]) as ma:
                if im.size != ma.size: errors.append(f"Dimension mismatch for {stem}: image={im.size}, mask={ma.size}"); continue
                dimensions[f"{im.size[0]}x{im.size[1]}"] += 1; values.update(map(int, np.unique(np.asarray(ma.convert('L')))))
        except (UnidentifiedImageError, OSError, ValueError) as exc: errors.append(f"Corrupted/unreadable pair '{stem}': {exc}"); continue
        match = re.match(config.patient_id_pattern, stem, re.I); patient_id = match.group(1).lower() if match else stem.lower()
        pairs.append(ImageMaskPair(images[stem], masks[stem], patient_id))
    report={"images_directory":str(images_dir),"masks_directory":str(masks_dir),"images_found":len(images),"masks_found":len(masks),"valid_pairs":len(pairs),"unique_patients":len({p.patient_id for p in pairs}),"dimensions":dict(dimensions),"mask_values":sorted(values),"mask_binarization":"Masks are converted in memory with mask > 0 to {0,1}; source labels/files are not modified.","errors":errors}
    if write_report: config.output_dir.mkdir(parents=True,exist_ok=True); (config.output_dir/"dataset_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if errors or not pairs: raise ValueError("Dataset validation failed; no files were silently discarded.\n"+"\n".join(errors or ["No paired images found."]))
    return pairs, report

def split_pairs(pairs: list[ImageMaskPair], config: TrainingConfig) -> dict[str,list[ImageMaskPair]]:
    if not np.isclose(config.train_fraction+config.validation_fraction+config.test_fraction,1): raise ValueError("Split fractions must sum to 1.")
    groups={}; [groups.setdefault(p.patient_id,[]).append(p) for p in pairs]; patients=list(groups); random.Random(config.seed).shuffle(patients)
    if len(patients)<3: raise ValueError("At least three patients are required for train/validation/test splitting.")
    n_train=max(1,round(len(patients)*config.train_fraction)); n_val=max(1,round(len(patients)*config.validation_fraction))
    if n_train+n_val>=len(patients): n_train,n_val=len(patients)-2,1
    psets={"train":patients[:n_train],"validation":patients[n_train:n_train+n_val],"test":patients[n_train+n_val:]}; splits={k:[p for pid in v for p in groups[pid]] for k,v in psets.items()}
    config.output_dir.mkdir(parents=True,exist_ok=True); (config.output_dir/"splits.json").write_text(json.dumps({k:[p.image_path.name for p in v] for k,v in splits.items()},indent=2),encoding="utf-8")
    return splits

class PairedMRIDataset(Dataset):
    def __init__(self,pairs:list[ImageMaskPair],config:TrainingConfig,training:bool=False): self.pairs,self.config,self.training=pairs,config,training
    def __len__(self): return len(self.pairs)
    def __getitem__(self,index):
        pair=self.pairs[index]; image=np.asarray(Image.open(pair.image_path).convert("L"),dtype=np.float32); mask=(np.asarray(Image.open(pair.mask_path).convert("L"))>0).astype(np.float32)
        size=(self.config.image_size[1],self.config.image_size[0]); image=cv2.resize(image,size,interpolation=cv2.INTER_LINEAR); mask=cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
        if self.training and self.config.augment:
            h,w=image.shape; matrix=cv2.getRotationMatrix2D((w/2,h/2),random.uniform(-10,10),random.uniform(.95,1.05)); image=cv2.warpAffine(image,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101); mask=cv2.warpAffine(mask,matrix,(w,h),flags=cv2.INTER_NEAREST)
            if self.config.allow_horizontal_flip and random.random()<.5: image,mask=np.fliplr(image).copy(),np.fliplr(mask).copy()
            if self.config.allow_vertical_flip and random.random()<.5: image,mask=np.flipud(image).copy(),np.flipud(mask).copy()
            if random.random()<.3: image=image*random.uniform(.9,1.1)+random.uniform(-10,10)
        image=(image-image.mean())/max(float(image.std()),1e-6)
        return {"image":torch.from_numpy(image[None]),"mask":torch.from_numpy((mask>0)[None].astype(np.float32)),"name":pair.image_path.name}

# U-Net Brain MRI Segmentation (Research Prototype)

This isolated module trains and runs a **standard 2D U-Net** for binary tumor-mask segmentation. It is not a clinical diagnostic or treatment tool and does not classify tumor type.

## 1. Dataset preparation

Use paired, identically named MRI images and ground-truth masks:

```text
dataset/images/patient001.png
dataset/masks/patient001.png
```

Masks are mandatory. The validator will fail, rather than pretend training is possible, if a pair is missing. The inspected `dataset/Images` and `dataset/Marks` layout is supported temporarily for this project.

## 2. Installation

```powershell
pip install -r ml/requirements.txt
```

Install a CPU or CUDA-compatible PyTorch build first if required by your machine.

## 3. Dataset validation

```powershell
python -m ml.validate
```

Expected: `DATASET VALIDATION PASSED`. Fix every named missing, corrupt, or dimension-mismatched pair; none are silently dropped.

## 4. U-Net implementation

`models/unet.py` provides encoder, bottleneck, decoder, and skip connections. `dataset.py` resizes image/mask separately, normalizes MRI only, and performs training-only augmentation.

## 5. Training

```powershell
python -m ml.train --epochs 50 --loss bce_dice
```

Expected output each epoch: Train Loss, Val Loss, Val Dice, and Val IoU. Use a lower `batch_size` in `config.py` for out-of-memory errors.

## 6. Evaluation

```powershell
python -m ml.test --checkpoint ml/checkpoints/brain_tumor_unet_best.pth
```

This evaluates only the held-out test split and saves metrics and visual comparisons.

## 7. Best model and artefacts

The best validation Dice model is saved as `checkpoints/brain_tumor_unet_best.pth`. History/configuration/splits and graphs go to `outputs/`.

## 8. New MRI prediction

```powershell
python -m ml.predict --image path/to/image.png
```

It writes a mask, boundary overlay, individual tumor measurements, and inter-tumoral separation. Without image spacing, measurements are pixels—not cm/mm.

## 9. Backend connection

Set `MODEL_PATH=ml/checkpoints/brain_tumor_unet_best.pth` in the project `.env`. FastAPI will call `inference/predictor.py`; with no path, it retains mock mode.

## 10. Full application

```powershell
python -m pytest
python run.py
```

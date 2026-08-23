# Brain Tumor MRI Analysis and Automated Reporting

A modular, runnable framework for **AI-assisted image analysis** of brain MRI images. It accepts PNG/JPG and NIfTI files, runs a deliberately non-clinical mock segmentation, identifies connected regions, calculates pixel/voxel and spacing-aware measurements, produces visualizations, and creates PDF/DOCX reports. It does **not** train or bundle a medical model, and it must not be used for diagnosis.

## Architecture

`frontend` is a responsive static dashboard. `backend/api` exposes FastAPI endpoints; `backend/services` coordinates loading, segmentation, region analysis, visualization, and reports. `processing` contains model-independent image mathematics. `ml` holds replaceable model adapters. `reports` contains document generators.

## Setup and run

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
python run.py
```

Open `http://127.0.0.1:8000`. API docs are at `http://127.0.0.1:8000/docs`.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Service health/status |
| POST | `/api/upload` | Upload PNG/JPG/JPEG/NIfTI MRI |
| POST | `/api/analysis/{study_id}` | Run preprocessing, mock segmentation and analysis |
| GET | `/api/analysis/{study_id}` | Retrieve saved result JSON |
| POST | `/api/reports/{study_id}/pdf` | Generate a PDF report |
| POST | `/api/reports/{study_id}/docx` | Generate a DOCX report |

## Mock inference

`processing/segmentation.py` uses a threshold-based deterministic demo mask so the app can operate without weights. It is only scaffolding and does not provide clinical segmentation quality. Physical measurements are intentionally marked unavailable for ordinary PNG/JPG uploads because no pixel spacing exists. NIfTI spacing is read when present.

## U-Net research training pipeline

The separate `ml/` package adds only a standard 2D U-Net for binary segmentation. It is an AI-assisted research/prototype tool, not a diagnosis, pathology confirmation, treatment recommendation, tumor-weight calculation, or replacement for a radiologist/neurosurgeon. No tumor-type classification is included.

1. **Dataset preparation** — use matching pairs in `ml/dataset/images/patient001.png` and `ml/dataset/masks/patient001.png`. Masks are mandatory; training cannot run without them. The inspected legacy `Images/` + `Marks/` folders remain supported during migration.
2. **Installation** — activate the environment and run `pip install -r requirements.txt` then `pip install -r ml/requirements.txt`. Install the appropriate CUDA PyTorch wheel first if needed.
3. **Dataset validation** — run `python -m ml.validate`. It reports all missing/orphan/corrupt/mismatched files, mask values, and counts in `ml/outputs/dataset_validation.json`; it never silently discards data. Expected final line: `DATASET VALIDATION PASSED`.
4. **U-Net implementation** — `ml/models/unet.py` contains the encoder, bottleneck, decoder, skip connections and configurable channels. `ml/dataset.py` resizes images/masks separately, normalizes images only, uses nearest-neighbour masks, and applies training-only rotation/scaling/intensity augmentation. Flips are disabled by default for anatomical safety.
5. **Training** — run `python -m ml.train --epochs 50 --loss bce_dice`. It uses reproducible patient-aware 70/15/15 splits, CUDA when available (CPU otherwise), scheduler, early stopping, loss choice (`dice`, `bce`, `bce_dice`), and prints Train Loss / Val Loss / Val Dice / Val IoU each epoch. Lower `batch_size` in `ml/config.py` if memory fails.
6. **Evaluation** — run `python -m ml.test --checkpoint ml/checkpoints/brain_tumor_unet_best.pth`. It uses only the held-out test split; saves Dice/IoU/precision/recall and MRI/ground-truth/prediction/overlay comparisons in `ml/outputs/`.
7. **Best model** — training saves `ml/checkpoints/brain_tumor_unet_best.pth`, configuration, splits, history, and loss/Dice/IoU graphs.
8. **New MRI prediction** — run `python -m ml.predict --image path/to/image.png`. It saves a mask, boundary overlay, and connected-component region analysis. Measurements are pixels unless trustworthy pixel/voxel spacing is supplied; this initial U-Net is 2D, not a 3D volume network.
9. **Backend connection** — set `MODEL_PATH=ml/checkpoints/brain_tumor_unet_best.pth` in `.env`. `SegmentationService` then calls `ml.inference.predictor.UNetPredictor`; no value leaves the existing mock safely active. Existing FastAPI, visualization, measurement, PDF, and DOCX layers remain intact.
10. **Complete app** — run `python -m pytest` then `python run.py`. If a checkpoint fails to load, check its path and provenance; clear `MODEL_PATH` to return to mock mode.

## Integrating a trained model later

1. Implement loading and `predict(image) -> mask` in `ml/models/unet.py` (or an Attention U-Net adapter).
2. Make it inherit `BaseSegmentationModel` and have `SegmentationService` instantiate it when `MODEL_PATH` is set.
3. Preserve the output contract: Boolean 2D/3D segmentation mask aligned with the preprocessed input.
4. Validate the model externally, add version/metrics to `model_info`, and replace the mock only after clinical governance and expert review.

## Notes and limitations

DICOM is deliberately not implemented yet; add a reader adapter in `ImageService` when the dependency and workflow are selected. The report includes fixed safety wording, metric placeholders, and supportive/specialist information placeholders—no diagnosis or treatment advice.

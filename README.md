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

## Integrating a trained model later

1. Implement loading and `predict(image) -> mask` in `ml/models/unet.py` (or an Attention U-Net adapter).
2. Make it inherit `BaseSegmentationModel` and have `SegmentationService` instantiate it when `MODEL_PATH` is set.
3. Preserve the output contract: Boolean 2D/3D segmentation mask aligned with the preprocessed input.
4. Validate the model externally, add version/metrics to `model_info`, and replace the mock only after clinical governance and expert review.

## Notes and limitations

DICOM is deliberately not implemented yet; add a reader adapter in `ImageService` when the dependency and workflow are selected. The report includes fixed safety wording, metric placeholders, and supportive/specialist information placeholders—no diagnosis or treatment advice.

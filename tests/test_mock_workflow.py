"""End-to-end validation for the non-clinical mock workflow."""
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


def synthetic_mri() -> bytes:
    """Create a small image with two bright regions for deterministic mock testing."""
    image = np.full((160, 160), 20, dtype=np.uint8)
    image[25:55, 25:55] = 245
    image[100:135, 105:140] = 245
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def test_complete_mock_analysis_and_reporting_workflow():
    with TestClient(app) as client:
        upload = client.post(
            "/api/upload",
            files={"file": ("synthetic_mri.png", synthetic_mri(), "image/png")},
        )
        assert upload.status_code == 200
        study_id = upload.json()["study_id"]

        analysis = client.post(f"/api/analysis/{study_id}")
        assert analysis.status_code == 200
        result = analysis.json()
        assert result["model"]["name"] == "mock"
        assert result["tumor_count"] == 2
        assert len(result["tumors"]) == 2
        assert len(result["pairwise_analysis"]) == 1
        assert result["tumors"][0]["area_pixels"] > 0
        assert result["tumors"][0]["physical_measurements_available"] is False

        annotated = client.get(result["annotated_image_url"])
        assert annotated.status_code == 200
        assert annotated.headers["content-type"] == "image/png"

        for format in ("pdf", "docx"):
            report = client.post(f"/api/reports/{study_id}/{format}")
            assert report.status_code == 200
            download = client.get(report.json()["download_url"])
            assert download.status_code == 200
            assert len(download.content) > 100

from fastapi.testclient import TestClient
from backend.main import app


def test_health():
    with TestClient(app) as client:
        response=client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_reject_invalid_upload():
    with TestClient(app) as client:
        response=client.post("/api/upload",files={"file":("not-mri.txt",b"no", "text/plain")})
    assert response.status_code == 415


def test_accept_nii_gz_upload_validation():
    with TestClient(app) as client:
        response=client.post("/api/upload",files={"file":("test.nii.gz",b"fake nii content", "application/gzip")})
    assert response.status_code == 200

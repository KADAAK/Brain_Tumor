from pathlib import Path
from fastapi import HTTPException, UploadFile
from backend.config import settings


def suffix_for(filename: str | None) -> str:
    name = filename or ""
    return ".nii.gz" if name.lower().endswith(".nii.gz") else Path(name).suffix.lower()


def validate_upload(file: UploadFile, size: int) -> None:
    if suffix_for(file.filename) not in settings.allowed_extensions:
        raise HTTPException(415, "Unsupported file type. Use PNG, JPG, JPEG, NIfTI (.nii/.nii.gz).")
    if size == 0:
        raise HTTPException(400, "Uploaded file is empty.")
    if size > settings.max_upload_size:
        raise HTTPException(413, f"Upload exceeds the {settings.max_upload_size // (1024 * 1024)} MB limit.")

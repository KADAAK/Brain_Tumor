from uuid import uuid4
from fastapi import APIRouter, File, UploadFile
from backend.config import settings
from backend.models.database import save_study
from backend.models.schemas import UploadResponse
from backend.utils.file_utils import suffix_for, validate_upload

router=APIRouter(prefix="/api",tags=["upload"])

@router.post("/upload",response_model=UploadResponse)
async def upload_mri(file: UploadFile = File(...)):
    content=await file.read(); validate_upload(file,len(content)); study_id=uuid4().hex
    filename=f"{study_id}{suffix_for(file.filename)}"; (settings.upload_dir/filename).write_bytes(content)
    save_study(study_id,filename)
    return UploadResponse(study_id=study_id,filename=file.filename or filename,content_type=file.content_type,message="Upload stored. Ready for AI-assisted mock analysis.")

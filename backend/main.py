from contextlib import asynccontextmanager
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.config import settings, ROOT_DIR
from backend.models.database import initialize_database
from backend.api import upload, analysis, reports

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories(); initialize_database(); yield

app=FastAPI(title="Brain Tumor AI",version="0.1.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
app.include_router(upload.router); app.include_router(analysis.router); app.include_router(reports.router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Always return JSON on unhandled errors — prevents <!DOCTYPE html> reaching the frontend."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )

@app.get("/api/health",tags=["system"])
def health():
    from backend.services.segmentation_service import SegmentationService
    return {"status":"ok","service":"brain-tumor-ai","model":SegmentationService().model_info}

app.mount("/files",StaticFiles(directory=settings.processed_dir),name="files")
app.mount("/reports",StaticFiles(directory=settings.report_dir),name="reports")
app.mount("/",StaticFiles(directory=ROOT_DIR/"frontend",html=True),name="frontend")


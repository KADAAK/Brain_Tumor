from pathlib import Path
from fastapi import APIRouter, HTTPException
from backend.config import settings
from backend.models.database import get_study
from backend.models.schemas import ReportResponse
from backend.services.report_service import ReportService

router=APIRouter(prefix="/api",tags=["reports"])

@router.post("/reports/{study_id}/{format}",response_model=ReportResponse)
def create_report(study_id:str,format:str):
    if format not in {"pdf","docx"}: raise HTTPException(400,"Report format must be pdf or docx.")
    study=get_study(study_id)
    if not study or not study["analysis"]: raise HTTPException(404,"Analysis not found. Run analysis before generating a report.")
    result=study["analysis"]; names=result.get("metadata",{}).get("assets",{})
    assets={k:[settings.processed_dir/n for n in v] if isinstance(v,list) else settings.processed_dir/v for k,v in names.items()}
    try: report=ReportService().generate(result,assets,settings.report_dir,format)
    except RuntimeError as exc: raise HTTPException(500,str(exc)) from exc
    return ReportResponse(study_id=study_id,format=format,download_url=f"/reports/{report.name}")

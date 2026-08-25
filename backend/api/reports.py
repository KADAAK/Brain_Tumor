from pathlib import Path
from fastapi import APIRouter, HTTPException
from backend.config import settings, ROOT_DIR
from backend.models.database import get_study
from backend.models.schemas import ReportResponse
from backend.services.report_service import ReportService
from reports.doctor_report_generator import generate_doctor_report

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


@router.post("/doctor-report/{study_id}", response_model=ReportResponse)
def create_doctor_report(study_id: str):
    """Generate the Doctor's Report PDF: original MRI → segmented MRI → radiology report → signature."""
    study = get_study(study_id)
    if not study or not study["analysis"]:
        raise HTTPException(404, "Analysis not found. Run analysis before generating a doctor report.")

    result = study["analysis"]
    names  = result.get("metadata", {}).get("assets", {})

    # Resolve image paths from stored asset names
    original_path   = settings.processed_dir / names["original"]   if "original"   in names else Path("/dev/null")
    segmented_path  = settings.processed_dir / names["annotated"]  if "annotated"  in names else Path("/dev/null")
    signature_path  = ROOT_DIR / "frontend" / "assets" / "signature.jpg"

    destination = settings.report_dir / f"{study_id}_doctor_report.pdf"

    try:
        report = generate_doctor_report(
            result        = result,
            original_image  = original_path,
            segmented_image = segmented_path,
            signature_image = signature_path,
            destination   = destination,
        )
    except Exception as exc:
        raise HTTPException(500, f"Doctor report generation failed: {exc}") from exc

    return ReportResponse(study_id=study_id, format="pdf", download_url=f"/reports/{report.name}")


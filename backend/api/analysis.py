from fastapi import APIRouter, HTTPException
from backend.config import settings
from backend.models.database import get_study, save_study
from backend.models.schemas import AnalysisResult
from backend.services.image_service import ImageLoadError, ImageService
from backend.services.segmentation_service import SegmentationService
from backend.services.tumor_analysis_service import TumorAnalysisService
from backend.services.distance_service import DistanceService
from backend.services.visualization_service import VisualizationService
from processing.preprocessing import preprocess
from processing.connected_components import detect_components

router=APIRouter(prefix="/api",tags=["analysis"])

@router.post("/analysis/{study_id}",response_model=AnalysisResult)
def analyze(study_id:str):
    study=get_study(study_id)
    if not study: raise HTTPException(404,"Study not found.")
    try: image,spacing=ImageService().load(settings.upload_dir/study["filename"])
    except ImageLoadError as exc: raise HTTPException(422,str(exc)) from exc
    segmentation=SegmentationService(); prepared=preprocess(image,spacing); mask=segmentation.predict(prepared.array); components=detect_components(mask)
    visual=VisualizationService().generate(study_id,prepared.array,mask,components,settings.processed_dir)
    tumors=TumorAnalysisService().analyze(components,prepared.spacing); pairs=DistanceService().analyze(components,prepared.spacing)
    base="/files/"
    model_info=segmentation.model_info
    warning="AI-assisted research/prototype output only; it is not a diagnosis and does not replace a radiologist or neurosurgeon." if model_info["status"]=="trained" else "Mock segmentation only; results are not diagnostic."
    result=AnalysisResult(study_id=study_id,tumor_count=len(tumors),tumors=tumors,pairwise_analysis=pairs,model=model_info,image_shape=list(prepared.array.shape),voxel_spacing=list(prepared.spacing) if prepared.spacing else None,original_image_url=base+visual["original"].name,segmentation_url=base+visual["segmentation"].name,annotated_image_url=base+visual["annotated"].name,warnings=[warning] + ([] if components else ["No segmented regions detected by the segmentation model."]))
    payload=result.model_dump(); payload["metadata"]["assets"]={k:[p.name for p in v] if isinstance(v,list) else v.name for k,v in visual.items()}; save_study(study_id,study["filename"],payload)
    return result

@router.get("/analysis/{study_id}",response_model=AnalysisResult)
def analysis_result(study_id:str):
    study=get_study(study_id)
    if not study or not study["analysis"]: raise HTTPException(404,"Analysis not found. Run analysis first.")
    return AnalysisResult.model_validate(study["analysis"])

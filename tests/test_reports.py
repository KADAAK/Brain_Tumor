from pathlib import Path
from PIL import Image
from reports.pdf_generator import generate_pdf
from reports.docx_generator import generate_docx


def sample_result():
    return {"study_id":"test", "tumor_count":1, "model":{"name":"mock","version":"0.1"}, "tumors":[{"tumor_id":"T1","area_pixels":10,"max_diameter_pixels":4.0,"centroid":[2,2],"bbox":[1,1,3,3],"equivalent_radius_pixels":1.8}],"pairwise_analysis":[]}


def test_generates_reports(tmp_path: Path):
    image=tmp_path/"image.png"; Image.new("L",(20,20)).save(image)
    assets={"original":image,"annotated":image,"crops":[image]}
    assert generate_pdf(sample_result(),assets,tmp_path/"report.pdf").exists()
    assert generate_docx(sample_result(),assets,tmp_path/"report.docx").exists()

from pathlib import Path
from reports.pdf_generator import generate_pdf
from reports.docx_generator import generate_docx


class ReportService:
    def generate(self, result: dict, assets: dict, report_dir: Path, format: str) -> Path:
        report_dir.mkdir(parents=True, exist_ok=True)
        target=report_dir / f"{result['study_id']}_report.{format}"
        try:
            return generate_pdf(result, assets, target) if format == "pdf" else generate_docx(result, assets, target)
        except Exception as exc:
            raise RuntimeError("Report generation failed. Please retry or inspect the source image.") from exc

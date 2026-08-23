from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Brain Tumor AI"
    debug: bool = False
    upload_dir: Path = ROOT_DIR / "data" / "uploads"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    prediction_dir: Path = ROOT_DIR / "data" / "predictions"
    report_dir: Path = ROOT_DIR / "data" / "reports"
    database_path: Path = ROOT_DIR / "data" / "brain_tumor_ai.db"
    model_path: str = ""
    max_upload_size: int = 25 * 1024 * 1024
    allowed_extensions: set[str] = {".png", ".jpg", ".jpeg", ".nii", ".nii.gz"}
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    def ensure_directories(self) -> None:
        for folder in (self.upload_dir, self.processed_dir, self.prediction_dir, self.report_dir):
            folder.mkdir(parents=True, exist_ok=True)


settings = Settings()

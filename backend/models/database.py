import json
import sqlite3
from pathlib import Path
from backend.config import settings


def initialize_database() -> None:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.database_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS studies (
            study_id TEXT PRIMARY KEY, filename TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            analysis_json TEXT
        )""")


def save_study(study_id: str, filename: str, analysis: dict | None = None) -> None:
    with sqlite3.connect(settings.database_path) as conn:
        conn.execute("INSERT INTO studies(study_id, filename, analysis_json) VALUES (?, ?, ?) "
                     "ON CONFLICT(study_id) DO UPDATE SET filename=excluded.filename, analysis_json=excluded.analysis_json",
                     (study_id, filename, json.dumps(analysis) if analysis else None))


def get_study(study_id: str) -> dict | None:
    with sqlite3.connect(settings.database_path) as conn:
        row = conn.execute("SELECT filename, analysis_json FROM studies WHERE study_id=?", (study_id,)).fetchone()
    if not row:
        return None
    return {"filename": row[0], "analysis": json.loads(row[1]) if row[1] else None}

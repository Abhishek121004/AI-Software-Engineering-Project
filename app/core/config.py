import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Application-wide settings managed through environment variables."""

    def __init__(self) -> None:
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
        self.chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chromadb")
        self.default_repository_id: str = os.getenv("DEFAULT_REPOSITORY_ID", "repository")

        self.base_dir: Path = Path(__file__).resolve().parent.parent.parent
        self.data_dir: Path = self.base_dir / "data"
        self.interview_notes_dir: Path = self.base_dir / "docs" / "interview-notes"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.interview_notes_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()

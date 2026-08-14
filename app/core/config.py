import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application-wide settings managed through environment variables."""
    
    def __init__(self):
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chromadb")
        
        # Absolute path references
        self.base_dir: Path = Path(__file__).resolve().parent.parent.parent
        self.data_dir: Path = self.base_dir / "data"
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    @property
    def has_gemini_key(self) -> bool:
        """Returns True if GEMINI_API_KEY is configured."""
        return bool(self.gemini_api_key)

settings = Settings()

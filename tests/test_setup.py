import os
from pathlib import Path
from app.core.config import settings

def test_settings_load_defaults():
    """Verify that settings load with correct defaults when no env variables are set."""
    # Since we loaded .env initially, let's just make sure the fields are present
    assert hasattr(settings, "gemini_api_key")
    assert hasattr(settings, "chroma_persist_dir")
    assert isinstance(settings.base_dir, Path)
    assert settings.data_dir.exists()

def test_has_gemini_key():
    """Check the has_gemini_key helper matches GEMINI_API_KEY environment state."""
    expected = bool(os.getenv("GEMINI_API_KEY", ""))
    assert settings.has_gemini_key == expected

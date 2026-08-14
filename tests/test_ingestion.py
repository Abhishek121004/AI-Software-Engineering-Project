import pytest
from pathlib import Path
from app.ingestion.reader import RepositoryReader

def write_file(path: Path, content: str | bytes) -> None:
    """Helper to write test files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        with open(path, "wb") as f:
            f.write(content)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

def test_repository_reader_scan(tmp_path):
    """Test standard file inclusion and exclusions."""
    # Write normal source files
    write_file(tmp_path / "app" / "main.py", "print('hello')")
    write_file(tmp_path / "static" / "index.js", "console.log('test')")
    write_file(tmp_path / "README.md", "# Project Title")
    
    # Write standard directories/files that should be ignored
    write_file(tmp_path / ".git" / "config", "some git configurations")
    write_file(tmp_path / "node_modules" / "react" / "index.js", "react-source-code")
    write_file(tmp_path / "venv" / "bin" / "python", "python executable binary mockup")
    write_file(tmp_path / "app" / "__pycache__" / "main.cpython-39.pyc", "pycache binary Mock")
    
    # Write excluded files
    write_file(tmp_path / "package-lock.json", "{}")
    
    # Write files with unsupported extensions
    write_file(tmp_path / "notes.txt", "Some random notes text")
    write_file(tmp_path / "image.png", "image binary data mockup")
    
    # Write a file that exceeds the max size
    write_file(tmp_path / "huge_file.py", "x" * 2000)  # Max size in test will be set to 1000 bytes
    
    # Write binary file containing invalid UTF-8
    write_file(tmp_path / "invalid_binary.py", b"\xff\xfe\x00\x00")

    # Instantiate Reader
    reader = RepositoryReader(root_dir=tmp_path, max_file_size_bytes=1000)
    files = reader.scan()

    # Convert to dictionary keyed by file_path for assertions
    files_dict = {f["file_path"]: f for f in files}

    # Verify inclusions
    assert "app/main.py" in files_dict
    assert files_dict["app/main.py"]["content"] == "print('hello')"
    assert files_dict["app/main.py"]["language"] == "python"

    assert "static/index.js" in files_dict
    assert files_dict["static/index.js"]["content"] == "console.log('test')"
    assert files_dict["static/index.js"]["language"] == "javascript"

    assert "README.md" in files_dict
    assert files_dict["README.md"]["content"] == "# Project Title"
    assert files_dict["README.md"]["language"] == "markdown"

    # Verify exclusions
    assert "notes.txt" not in files_dict
    assert "image.png" not in files_dict
    assert ".git/config" not in files_dict
    assert "node_modules/react/index.js" not in files_dict
    assert "venv/bin/python" not in files_dict
    assert "app/__pycache__/main.cpython-39.pyc" not in files_dict
    assert "package-lock.json" not in files_dict
    assert "huge_file.py" not in files_dict
    assert "invalid_binary.py" not in files_dict

    # Total included files should be exactly 3
    assert len(files) == 3

def test_repository_reader_invalid_dir():
    """Verify that reader raises exception on invalid directories."""
    with pytest.raises(FileNotFoundError):
        RepositoryReader("c:/invalid/path/that/does/not/exist/at/all/12345")

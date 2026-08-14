import os
from pathlib import Path
from typing import Dict, List, Set, Optional

class RepositoryReader:
    """Discovers and reads text files within a codebase repository directory."""
    
    # Supported file extension to language mappings
    SUPPORTED_EXTENSIONS: Dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "jsx",
        ".tsx": "tsx",
        ".java": "java",
        ".cpp": "cpp",
        ".h": "cpp",
        ".cc": "cpp",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    
    # Standard ignore directories
    IGNORE_DIRS: Set[str] = {
        ".git",
        "node_modules",
        "venv",
        ".venv",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".idea",
        ".vscode",
    }
    
    # Files to ignore (e.g. lock files, binary files, package locks)
    IGNORE_FILES: Set[str] = {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
    }

    def __init__(self, root_dir: str | Path, max_file_size_bytes: int = 512 * 1024):
        """
        Initialize the reader.
        
        Args:
            root_dir: The path of the repository folder to read.
            max_file_size_bytes: The maximum size of a file to read (default 512 KB).
        """
        self.root_dir = Path(root_dir).resolve()
        self.max_file_size_bytes = max_file_size_bytes
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Repository directory does not exist: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Provided path is not a directory: {self.root_dir}")

    def is_ignored(self, path: Path) -> bool:
        """Determines if a file or directory path should be ignored."""
        # Resolve to check components
        parts = path.relative_to(self.root_dir).parts
        
        # Check if any parent directory is in the IGNORE_DIRS list
        for part in parts[:-1]:
            if part in self.IGNORE_DIRS:
                return True
                
        # Check if the path itself is a directory in the IGNORE_DIRS list
        if path.is_dir() and path.name in self.IGNORE_DIRS:
            return True
            
        # Check if it is an ignored file
        if path.is_file() and path.name in self.IGNORE_FILES:
            return True

        # Skip obvious hidden/generated directories and files outside the supported set
        if any(part.startswith(".") and part not in {".", ".."} for part in parts[:-1]):
            return True
            
        return False

    def get_language(self, path: Path) -> Optional[str]:
        """Returns the mapped programming language name or None if not supported."""
        return self.SUPPORTED_EXTENSIONS.get(path.suffix.lower())

    def read_file_safe(self, path: Path) -> Optional[str]:
        """
        Safely read a file, verifying size limits and UTF-8 encoding.
        
        Returns:
            The string content of the file, or None if the file is ignored or unreadable.
        """
        try:
            # Check file size
            if path.stat().st_size > self.max_file_size_bytes:
                return None
                
            # Read content using UTF-8 decoding, catch errors if binary
            raw = path.read_bytes()
            if b"\x00" in raw:
                return None
            return raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, PermissionError, OSError):
            # Suppress errors for binary files or lock files we shouldn't open
            return None

    def scan(self) -> List[Dict[str, str]]:
        """
        Scans the directory structure and returns files list.
        
        Returns:
            A list of dicts: [
                {
                    "file_path": "relative/path/to/file.py",
                    "content": "file code here...",
                    "language": "python"
                }
            ]
        """
        discovered_files = []
        
        for root, dirs, files in os.walk(self.root_dir):
            current_dir_path = Path(root)
            
            # Prune directories in-place to prevent os.walk from entering them
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
            
            for file_name in files:
                file_path = current_dir_path / file_name
                
                # Check ignores
                if self.is_ignored(file_path):
                    continue
                    
                # Match extension
                language = self.get_language(file_path)
                if not language:
                    continue
                    
                content = self.read_file_safe(file_path)
                if content is None:
                    continue
                    
                # Relative path representation
                relative_path = file_path.relative_to(self.root_dir).as_posix()
                
                discovered_files.append({
                    "file_path": relative_path,
                    "content": content,
                    "language": language
                })
                
        return discovered_files

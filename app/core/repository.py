from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from git import Repo

from app.core.config import settings
from app.core.models import CodeChunk, RepositoryCorpus, RepositoryFile
from app.ingestion.chunker import CodeAwareChunker
from app.ingestion.reader import RepositoryReader


@dataclass(slots=True)
class RepositoryIndexOptions:
    root_dir: Path
    repository_id: Optional[str] = None
    max_file_size_bytes: int = 512 * 1024
    chunk_size_lines: int = 80
    chunk_overlap_lines: int = 12


class RepositoryIndexer:
    """Builds an inspectable repository corpus from files and chunks."""

    def __init__(self, chunker: Optional[CodeAwareChunker] = None, clone_dir: Optional[Path] = None) -> None:
        self.chunker = chunker or CodeAwareChunker()
        self.clone_dir = Path(clone_dir or settings.data_dir / "repositories").resolve()
        self.clone_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _repo_slug_from_url(repo_url: str) -> str:
        parsed = urlparse(repo_url)
        repo_name = Path(parsed.path).name
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        if not repo_name:
            raise ValueError("Could not determine repository name from URL.")
        return repo_name

    def clone_github_repository(self, repo_url: str) -> Path:
        repo_name = self._repo_slug_from_url(repo_url)
        repo_hash = hashlib.sha1(repo_url.encode("utf-8")).hexdigest()[:12]
        target_dir = self.clone_dir / f"{repo_name}-{repo_hash}"
        if target_dir.exists() and (target_dir / ".git").exists():
            return target_dir
        if target_dir.exists():
            raise FileExistsError(f"Clone target already exists and is not a git repository: {target_dir}")
        Repo.clone_from(repo_url, target_dir, depth=1)
        return target_dir

    def build_corpus(self, options: RepositoryIndexOptions) -> RepositoryCorpus:
        root_dir = Path(options.root_dir).resolve()
        reader = RepositoryReader(root_dir=root_dir, max_file_size_bytes=options.max_file_size_bytes)
        files = [RepositoryFile(**file_data) for file_data in reader.scan()]
        chunks = self.chunker.chunk_all(
            [
                {"file_path": file_data.file_path, "content": file_data.content, "language": file_data.language}
                for file_data in files
            ],
            repository_id=options.repository_id or root_dir.name,
        )
        return RepositoryCorpus(
            repository_id=options.repository_id or root_dir.name,
            root_dir=root_dir,
            files=files,
            chunks=[CodeChunk(**chunk) for chunk in chunks],
        )


def default_repository_id(root_dir: str | Path) -> str:
    return Path(root_dir).resolve().name or settings.default_repository_id

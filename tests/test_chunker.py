import pytest
from app.ingestion.chunker import CodeAwareChunker

def test_chunk_python_file():
    """Verify that python files are parsed via AST and symbols mapped accurately."""
    code_content = """# Module Docstring
import os

class MathService:
    def add(self, a, b):
        # A simple add function
        return a + b

    def subtract(self, a, b):
        return a - b

def health_check():
    return "ok"
"""
    file_data = {
        "file_path": "services/math.py",
        "content": code_content,
        "language": "python"
    }

    # Use smaller chunk size to trigger multiple chunks
    chunker = CodeAwareChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_file(file_data)

    assert len(chunks) > 0
    
    # Check that metadata properties are populated
    for chunk in chunks:
        meta = chunk["metadata"]
        assert meta["file_path"] == "services/math.py"
        assert meta["language"] == "python"
        assert "start_line" in meta
        assert "end_line" in meta
        assert "chunk_index" in meta
        assert "symbol" in meta
        assert "chunk_type" in meta
        
    # Find chunks overlapping specific functions
    add_chunks = [c for c in chunks if c["metadata"]["symbol"] == "add"]
    sub_chunks = [c for c in chunks if c["metadata"]["symbol"] == "subtract"]
    hc_chunks = [c for c in chunks if c["metadata"]["symbol"] == "health_check"]

    assert len(add_chunks) > 0
    assert add_chunks[0]["metadata"]["chunk_type"] == "function"
    
    assert len(sub_chunks) > 0
    assert sub_chunks[0]["metadata"]["chunk_type"] == "function"
    
    assert len(hc_chunks) > 0
    assert hc_chunks[0]["metadata"]["chunk_type"] == "function"

def test_chunk_javascript_file():
    """Verify standard language chunking maps lines correctly without AST crashes."""
    js_content = """// Express server setup
const express = require('express');
const app = express();

app.get('/api/users', (req, res) => {
    res.json([{ id: 1, name: 'Alice' }]);
});

app.listen(3000, () => {
    console.log('Server is running');
});
"""
    file_data = {
        "file_path": "server.js",
        "content": js_content,
        "language": "javascript"
    }

    chunker = CodeAwareChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_file(file_data)

    assert len(chunks) > 0
    for chunk in chunks:
        meta = chunk["metadata"]
        assert meta["file_path"] == "server.js"
        assert meta["language"] == "javascript"
        assert meta["symbol"] == "general"
        assert meta["chunk_type"] == "general"
        assert meta["start_line"] > 0
        assert meta["end_line"] >= meta["start_line"]

import ast
from typing import Dict, Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

class CodeAwareChunker:
    """Chunks code files using language-specific splitters while preserving metadata and symbol tracking."""
    
    # Map reader language string to LangChain Language enum
    LANG_MAP = {
        "python": Language.PYTHON,
        "javascript": Language.JS,
        "typescript": Language.TS,
        "jsx": Language.JS,
        "tsx": Language.TS,
        "java": Language.JAVA,
        "cpp": Language.CPP,
        "markdown": Language.MARKDOWN,
    }

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_splitter_for_language(self, language: str) -> RecursiveCharacterTextSplitter:
        """Returns LangChain splitter matching language or a fallback text splitter."""
        lang_enum = self.LANG_MAP.get(language)
        if lang_enum:
            return RecursiveCharacterTextSplitter.from_language(
                language=lang_enum,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

    def extract_python_symbols(self, content: str) -> List[Dict[str, Any]]:
        """Parses Python AST to extract functions and classes with their start/end lines."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []
            
        symbols = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_line = getattr(node, "end_lineno", node.lineno)
                symbols.append({
                    "name": node.name,
                    "type": "function",
                    "start_line": node.lineno,
                    "end_line": end_line
                })
            elif isinstance(node, ast.ClassDef):
                end_line = getattr(node, "end_lineno", node.lineno)
                symbols.append({
                    "name": node.name,
                    "type": "class",
                    "start_line": node.lineno,
                    "end_line": end_line
                })
                
        return symbols

    def find_matching_symbol(self, start_line: int, end_line: int, symbols: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Finds the most specific (smallest span) symbol that overlaps with the chunk lines."""
        matching_symbol = {}
        min_span = float("inf")
        
        for symbol in symbols:
            # Overlap conditions
            # Chunk: [start_line, end_line]
            # Symbol: [sym_start, sym_end]
            sym_start = symbol["start_line"]
            sym_end = symbol["end_line"]
            
            if not (end_line < sym_start or start_line > sym_end):
                span = sym_end - sym_start
                if span < min_span:
                    min_span = span
                    matching_symbol = symbol
                    
        return matching_symbol

    def chunk_file(self, file_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Splits a single file's content into chunks and aggregates metadata.
        
        Args:
            file_data: Dict containing 'file_path', 'content', and 'language'.
            
        Returns:
            List of dicts representing chunks with metadata.
        """
        file_path = file_data["file_path"]
        content = file_data["content"]
        language = file_data["language"]
        
        splitter = self.get_splitter_for_language(language)
        splits = splitter.split_text(content)
        
        python_symbols = []
        if language == "python":
            python_symbols = self.extract_python_symbols(content)
            
        chunks = []
        last_found_offset = 0
        
        for i, split in enumerate(splits):
            # Locate split substring in content to calculate line numbers
            # We search starting from last_found_offset to resolve duplicate substrings sequentially
            start_char = content.find(split, last_found_offset)
            if start_char == -1:
                # Fallback to general search if search-ahead fails
                start_char = content.find(split)
                
            if start_char != -1:
                last_found_offset = start_char + len(split)
                start_line = content[:start_char].count("\n") + 1
                end_line = start_line + split.count("\n")
            else:
                start_line = 1
                end_line = 1
                
            # Base metadata
            metadata = {
                "file_path": file_path,
                "language": language,
                "start_line": start_line,
                "end_line": end_line,
                "chunk_index": i,
                "symbol": "general",
                "chunk_type": "general",
            }
            
            # Map Python AST symbols
            if language == "python" and python_symbols:
                match = self.find_matching_symbol(start_line, end_line, python_symbols)
                if match:
                    metadata["symbol"] = match["name"]
                    metadata["chunk_type"] = match["type"]
                    
            chunks.append({
                "content": split,
                "metadata": metadata
            })
            
        return chunks

    def chunk_all(self, files_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Splits a list of files into a single flat list of chunks."""
        all_chunks = []
        for file_data in files_list:
            all_chunks.extend(self.chunk_file(file_data))
        return all_chunks

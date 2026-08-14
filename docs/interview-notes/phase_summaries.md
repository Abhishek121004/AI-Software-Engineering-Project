# Interview Notes: AI Software Engineering Copilot

This document outlines key technical decisions, designs, alternatives, tradeoffs, and mock interview questions for each phase of the development. Use this to prepare for your engineering interviews.

---

## Phase 1: Project Setup & Configuration

### Problem
- Application code needs a structured way to load API keys (e.g. Gemini API Key) and configuration paths (ChromaDB directory) securely, preventing secrets from leaking into Git repositories, and keeping the codebase modular and testable.

### Design
- Uses a central configuration file (`app/core/config.py`) that loads from `.env` using `python-dotenv`.
- Instantiates a singleton `settings` object containing paths and properties that other application modules import.
- Keeps configuration decoupled from execution.

### Alternatives
- **Hardcoded constants:** Unacceptable for security (API keys would leak to git) and makes switching between environment contexts (local dev vs. staging/prod) impossible.
- **Pydantic BaseSettings:** A robust alternative. We chose a simple class loading environment variables manually to avoid unnecessary dependencies and keep configuration highly transparent and easy to trace. We will stick to `python-dotenv` for local environment injection.

### Tradeoffs
- A singleton settings object is simple and clean but can sometimes make mock testing slightly more involved if we need to dynamically change settings in a test. However, it is the standard Python pattern for config management.

### Mock Interview Questions
1. **Q: Why do we use python-dotenv instead of hardcoding configurations?**
   * **A:** Hardcoding config parameters violates security guidelines (never check keys into version control) and 12-factor app principles. Using `.env` allows us to inject different parameters (e.g., local mock databases during tests, production databases during hosting) without modifying codebase logic.
2. **Q: How does a Python package structure imports? What is the purpose of `__init__.py`?**
   * **A:** `__init__.py` files designate directory folders as Python packages, enabling structured absolute and relative imports (e.g., `from app.core.config import settings`). Since Python 3.3, implicit namespace packages don't require them, but we write them explicitly to establish clean module boundaries and run setup scripts if needed.

---

## Phase 2: Repository Ingestion

### Problem
- The RAG system needs to locate, filter, and read source files inside code repositories, skipping binary images, dependency directories (like `node_modules`), virtual environments (`venv`), and build directories (`dist`, `build`).

### Design
- Implements `RepositoryReader` using `os.walk` to traverse directories.
- Prunes directories in-place inside `os.walk` (`dirs[:] = ...`) to prevent scanning ignored subdirectories entirely, improving performance.
- Validates file extensions (mapped to specific languages) and reads content using strict UTF-8 decoding to reject binary assets or compiled code.
- Includes file size checks (`max_file_size_bytes`) to avoid loading bloated dumps or lock files.

### Alternatives
- **Command line `find` or `ripgrep`**: Requires system dependencies, which complicates cross-platform execution (especially Windows workspace).
- **GitPython directory cloning**: Useful for remote cloning, but a custom file walker is needed for scanning the resulting folder regardless. Starting with a directory scanner covers both local repositories and remote repositories (cloned locally).

### Tradeoffs
- Parsing everything via UTF-8 can miss files with slightly corrupted encodings, but it is the most reliable way to prevent binary files from choking LLM embedding loops.
- Directory scanning is single-threaded; for massive enterprise codebases, we would eventually need a worker pool or indexed database caching.

### Mock Interview Questions
1. **Q: How did you design the scanner to ignore large directories like `node_modules` efficiently?**
   * **A:** I modified the directory list `dirs[:]` in-place inside the `os.walk` loop. Modifying `dirs` prevents `os.walk` from recursively stepping into those folders, saving massive CPU cycles and I/O reads.
2. **Q: How does the ingestion reader handle binary files?**
   * **A:** We check if the file size exceeds a threshold (e.g., 512KB). If it passes, we try to open it with UTF-8 decoding in strict mode. If a `UnicodeDecodeError` is caught, we classify it as non-text/binary and skip it.

---

## Phase 3: Code-Aware Chunking

### Problem
- General text splitters split documents by character lengths. In codebase retrieval, doing this cuts code definitions (e.g. splitting a function signature, cutting off imports, or splitting class variables) in half, resulting in poor retrieval search precision and unusable code generation prompts.

### Design
- Implements `CodeAwareChunker` that maps target languages to LangChain's syntax splitters (`RecursiveCharacterTextSplitter.from_language`). These splitters leverage language-specific code grammar boundaries (like class/method syntax and bracket closures) rather than naive character counts.
- Captures relative line ranges (`start_line` and `end_line`) for every generated split chunk by matching substring indices against the original document string.
- Integrates Python `ast` parser logic: traverses Python files' Abstract Syntax Trees to resolve functions and classes. If a chunk falls within a function or class range, we stamp it with the `symbol` name and `chunk_type` metadata.

### Alternatives
- **Fixed-size chunking**: Splitting strictly every N characters. Highly prone to cutting functions and classes in half. Very poor code semantic representations.
- **Recursive character chunking**: Better than fixed-size as it splits on paragraphs or line-breaks, but does not respect structural programming syntax.
- **Tree-sitter AST parsing**: Highly accurate for all languages. However, requires installing compiled packages that are highly platform-dependent, raising high setup overhead. A hybrid LangChain Language splitter + Python `ast` fallback is the most stable and portable choice for cross-platform applications.

### Tradeoffs
- Parsing ASTs is computationally heavier than plain text splitting. We mitigate this by checking the AST *only* for Python files (where the `ast` module is native and extremely fast).
- Substring offset line estimation can align incorrectly if identical files contain repeated, non-contiguous duplicate blocks. We solve this by starting character searches from the index of the previously completed chunk.

### Mock Interview Questions
1. **Q: Why is code-aware chunking superior to recursive text chunking for codebase RAG?**
   * **A:** Recursive character chunking splits on generic strings (like newlines or spaces). Code-aware chunking leverages programming syntax rules (like function signatures, brackets, class boundaries). This keeps semantic structures whole inside single chunks, preventing crucial contexts from being fragmented.
2. **Q: How did you compute the exact line range for each code chunk?**
   * **A:** When the splitter returns text chunks, we look up their indices in the original source string using `content.find(chunk)`. We count the occurrences of newlines (`\n`) prior to that index to find the start line. To avoid mapping duplicate strings incorrectly, we restrict each index search to start from the end position of the previously matched chunk.
3. **Q: How did you extract functions and classes to attach symbol names to Python chunks?**
   * **A:** I used Python's built-in `ast` module to build an Abstract Syntax Tree of the code. Walking this tree, we extract functions (`FunctionDef`) and classes (`ClassDef`), recording their `lineno` and `end_lineno`. For each chunk, we find the smallest overlapping symbol range and attach its name as metadata.

---

## Phase 4: Embeddings and ChromaDB

### Problem
- Chunks of code are unstructured text. To perform search queries against these chunks, we need to map them to continuous high-dimensional vectors (embeddings) representing their semantic meaning and store them in a persistent data repository supporting similarity lookups, repository filtering, and quick indexes deletion.

### Design
- Uses `langchain-huggingface`'s `HuggingFaceEmbeddings` with `"sentence-transformers/all-MiniLM-L6-v2"` to compute local 384-dimensional dense vectors on the CPU.
- Uses `ChromaDB` (via LangChain's `Chroma` wrapper) as the vector index store, persisted on disk in a customizable path (`data/chromadb`).
- Configures metadata filtering using a unique `repository_id` tag injected into every document chunk. When searching, we pass `filter={"repository_id": repository_id}`.
- Implements codebase clearance by executing native `.delete(where={"repository_id": repository_id})` on the underlying collection, making the vector store clean, isolated, and reproducible.

### Alternatives
- **Cloud Vector Stores (Pinecone, Weaviate)**: Require cloud setups, network latency, api credentials, and credit payments. Local ChromaDB is lightweight, runs completely in-memory or on local disk, has zero external latency, and is free, making it ideal for developers and resume showcases.
- **Gemini Embeddings API**: A strong remote API alternative. Using local embeddings (sentence-transformers) ensures that database ingestion operates for free without API rate limits or network issues, though it adds a local model startup memory overhead.

### Tradeoffs
- ChromaDB runs as an in-process library. While excellent for local apps, it does not scale horizontally like standalone services (e.g. Qdrant or Milvus).
- Local embeddings running on single-thread CPU can be slow for tens of thousands of files, but for standard target resume repositories (under 1000 files), it finishes indexing in seconds.

### Mock Interview Questions
1. **Q: How did you isolate different codebases inside the same vector database?**
   * **A:** I injected a unique metadata field called `repository_id` into every chunk during database loading. During semantic lookup, we pass a Chroma metadata query filter `{"repository_id": repository_id}` which narrows results strictly to that codebase.
2. **Q: Why did you choose local HuggingFace embeddings instead of an API-based embedding system?**
   * **A:** Local embeddings (via `sentence-transformers`) offer free, offline, rate-limit-free execution with complete reproducibility. It is ideal for local development and testing since it runs out-of-the-box on the CPU without requiring cloud tokens.
3. **Q: How does Chroma delete documents, and why is this important for reproducibility?**
   * **A:** We fetch the raw collection via the `_collection` attribute of the LangChain client and call the native `delete(where={"repository_id": repository_id})` API. This allows developers to re-index or wipe codebase listings without dropping the entire database or affecting other stored repositories.




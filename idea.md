# AI Software Engineering Copilot

## Role

You are my senior AI/ML engineering pair programmer.

I want to build a resume-quality **AI Software Engineering Copilot** that demonstrates practical understanding of LLM applications, RAG, LangChain, agents, tool/function calling, memory, context engineering, and LLM evaluation.

I will use **Antigravity** as my development environment.

Your job is to help me build the application incrementally, explain important decisions, write maintainable code, test the implementation, and never hide architectural decisions from me.

---

# 1. Project Objective

Build an AI Software Engineering Copilot that allows a developer to provide a GitHub repository or local codebase and ask questions about that repository.

The system should understand source code and documentation and provide grounded answers based on the actual repository.

Example questions:

* How does authentication work in this repository?
* Where is JWT generated?
* Where is JWT validated?
* Explain the login flow.
* Which files interact with the database?
* Where is `authenticateToken()` used?
* Explain the architecture of this repository.
* What happens when a user registers?
* Find the code responsible for sending emails.
* Explain this function.
* Generate unit tests for this function.
* Why could this API return a 401 response?

Answers should cite the relevant files and, where possible, line numbers or code sections.

The application must clearly distinguish between information retrieved from the repository and general knowledge.

If sufficient repository evidence is unavailable, the system should explicitly say that it does not have enough evidence rather than hallucinating.

---

# 2. Core Technology Stack

Use the following stack unless there is a strong technical reason to change something.

### Backend / AI

* Python
* LangChain
* Hugging Face embeddings
* ChromaDB
* Gemini API for production inference
* Ollama for local development/testing
* Pydantic
* GitPython or an appropriate GitHub/repository library

### UI

* Streamlit

### Deployment

* Hugging Face Spaces
* Docker only if required by the chosen Spaces configuration

### Development

* Git
* GitHub
* Environment variables through `.env`

Do NOT introduce LangGraph unless I explicitly request it.

Do NOT introduce unnecessary frameworks such as CrewAI unless I explicitly request them.

---

# 3. Important Development Philosophy

This is a learning and resume project.

Do NOT generate the entire application in one step.

Build it incrementally.

After every major phase:

1. Explain what we implemented.
2. Explain why we implemented it.
3. Show the important files changed.
4. Run tests.
5. Verify the feature.
6. Tell me how I should explain the feature in an interview.
7. Wait for my approval before moving to the next major phase.

I want to understand the system, not blindly accept generated code.

Avoid unnecessary abstraction.

Prefer simple, readable Python over overly complex architecture.

---

# 4. High-Level Architecture

The initial architecture should be:

User
↓
Streamlit UI
↓
Application Layer
↓
LangChain
↓
Query Processing
↓
Retriever / Agent
↓
Vector Store / Repository Tools
↓
Context Engineering
↓
LLM
↓
Grounded Answer
↓
Sources + Retrieval Information

The system should eventually support:

* RAG
* Semantic retrieval
* Code-aware chunking
* Metadata
* Repository search tools
* Function/tool calling
* LangChain agents
* Conversation memory
* Context engineering
* Citation/source tracking
* RAG evaluation
* Optional MCP integration later

Do not implement every feature immediately.

---

# 5. Development Phases

Implement the project in the following order.

## Phase 1 — Project Setup

Create a clean project structure.

Suggested structure:

ai-software-engineering-copilot/
│
├── app/
│   ├── ui/
│   ├── ingestion/
│   ├── retrieval/
│   ├── agents/
│   ├── tools/
│   ├── memory/
│   ├── evaluation/
│   ├── core/
│   └── main.py
│
├── tests/
│
├── data/
│
├── evaluation/
│
├── docs/
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── Dockerfile

Do not create unnecessary files.

---

# Phase 2 — Repository Ingestion

Implement repository ingestion.

The system should be able to accept a repository and discover relevant files.

Initially support:

* Python
* JavaScript
* TypeScript
* JSX/TSX
* Java
* C++
* Markdown
* JSON
* YAML

Ignore:

* `.git`
* `node_modules`
* `venv`
* `.venv`
* `__pycache__`
* `dist`
* `build`
* binary files
* large generated files
* lock files unless explicitly required

Create a reusable repository ingestion module.

Before implementing advanced parsing, make the basic ingestion pipeline work.

---

# Phase 3 — Code-Aware Chunking

Implement code-aware chunking rather than blindly splitting the repository into arbitrary character lengths.

Each chunk should preserve useful metadata.

Example metadata:

{
"repository": "...",
"file_path": "middleware/auth.js",
"language": "javascript",
"symbol": "authenticateToken",
"chunk_type": "function",
"start_line": 12,
"end_line": 37
}

If precise AST-based parsing is practical for a language, use it.

Otherwise use a reliable fallback strategy.

Explain the tradeoff between:

* fixed-size chunking
* recursive character chunking
* code-aware chunking

---

# Phase 4 — Embeddings and Chroma

Use Hugging Face embeddings.

Store embeddings in Chroma.

The vector database must preserve metadata.

Implement:

* document insertion
* collection management
* retrieval
* repository-specific filtering

Do not make the vector database a global uncontrolled singleton.

Make repository indexing reproducible.

---

# Phase 5 — Basic RAG

Build the first working RAG pipeline:

User Question
↓
Retriever
↓
Relevant Code/Documents
↓
Prompt
↓
LLM
↓
Answer

The answer must include source information.

Example:

Answer:
JWT authentication is implemented in the authentication middleware.

Sources:

* middleware/auth.js
* controllers/authController.js

At this stage, do NOT implement agents yet.

First make basic RAG reliable.

---

# Phase 6 — Retrieval Improvement

After basic RAG works, improve retrieval.

Investigate:

* top-k retrieval
* metadata filtering
* similarity thresholds
* exact identifier matching
* semantic search
* keyword search
* reranking if practical

For code, exact identifiers are important.

For example:

`authenticateToken`

should be retrievable even when the natural-language query is different.

Document the retrieval design.

Do not add complexity without measuring whether it improves results.

---

# Phase 7 — Repository Tools

Implement tools using LangChain.

Initial tools:

* search_code
* read_file
* find_function
* find_references
* get_repository_tree

Each tool should have:

* clear input schema
* validation
* useful output
* error handling
* tests

Tools should never allow unrestricted access outside the indexed repository.

---

# Phase 8 — LangChain Agent

After RAG and tools work independently, create a LangChain agent.

The agent should decide whether it needs:

* RAG retrieval
* code search
* file reading
* function lookup
* reference lookup
* repository structure

Example:

User:
"Where is authenticateToken used?"

Agent:

1. Determine that repository search is required.
2. Call search_code.
3. Inspect relevant files.
4. Produce a grounded response.

Do not make the agent unnecessarily autonomous.

Use explicit tools and clear system instructions.

---

# Phase 9 — Memory

Add conversational memory.

The system should support follow-up questions.

Example:

User:
"Explain the authentication system."

Assistant:
Provides explanation.

User:
"Where is the token generated?"

Assistant:
Understands that "the token" refers to the authentication discussion.

Keep memory implementation simple and inspectable.

Do not introduce long-term user profiling.

---

# Phase 10 — Context Engineering

Create a dedicated context-building layer.

Do not simply send all retrieved chunks to the LLM.

The context builder should consider:

* user query
* retrieved chunks
* file metadata
* relevant functions
* tool results
* repository structure
* conversation history

The goal is to construct compact, high-quality context.

Make the context visible during development so retrieval problems can be debugged.

---

# Phase 11 — Grounding and Citations

Every repository-based answer should provide sources.

Prefer:

* file path
* symbol/function
* line numbers where available

Example:

Source:
`backend/middleware/auth.py`
Function:
`authenticate_token`
Lines:
20–42

If the system cannot find sufficient evidence, respond appropriately instead of inventing an answer.

Add a source validation mechanism if practical.

---

# Phase 12 — Evaluation

Create a dedicated evaluation framework.

Create a dataset of repository questions.

Example:

{
"question": "Where is JWT generated?",
"expected_files": [
"controllers/authController.js"
]
}

Measure at least:

### Retrieval

* Recall@K
* Precision@K
* MRR

### Generation

* Faithfulness
* Answer relevance
* Context relevance

### System

* latency
* token usage where available

Create an evaluation report.

I want to be able to compare:

1. Basic vector retrieval
2. Improved retrieval
3. Retrieval + reranking
4. Agentic retrieval

Do not fabricate evaluation numbers.

Only display measurements produced by actual experiments.

---

# Phase 13 — Streamlit UI

Create a clean but simple Streamlit application.

The UI should include:

### Repository

* GitHub repository URL input
* Repository indexing button
* indexing status

### Chat

* conversation interface
* question input
* grounded answer
* source citations

### Retrieval Inspection

Show:

* retrieved chunks
* file paths
* metadata
* relevance scores where available

### Evaluation

Show:

* retrieval metrics
* generation metrics
* latency

### Configuration

Show relevant settings such as:

* LLM
* embedding model
* top-k
* retrieval strategy

Do not spend excessive time on visual design.

The AI functionality is more important.

---

# Phase 14 — Deployment

Prepare the project for deployment to Hugging Face Spaces using Streamlit.

The deployed version should preferably use an API-based LLM rather than depending on Ollama.

Use environment variables/secrets for API keys.

Never commit API keys.

Create:

* README
* `.env.example`
* deployment instructions
* requirements
* appropriate Hugging Face configuration

The application should work from a clean environment.

---

# Phase 15 — Optional MCP Extension

Only after the main system is stable.

Consider exposing repository operations through MCP:

* search_code
* read_file
* find_symbol
* find_references
* repository_tree

Do not make MCP necessary for the core application.

Document why MCP is useful and what problem it solves.

---

# Phase 16 — Optional Multi-Agent Experiment

Do NOT implement this initially.

Only experiment with multiple agents if there is a clear benefit.

Potential agents:

* Retrieval Agent
* Code Analysis Agent
* Documentation Agent

Compare single-agent and multi-agent approaches experimentally.

The project should not become complicated merely to include the multi-agent concept.

---

# 6. Security Requirements

Treat repository content as untrusted input.

Prevent:

* path traversal
* arbitrary filesystem access
* execution of repository code
* shell command execution based solely on LLM output
* exposure of environment variables
* accidental API key leakage

The agent must not execute arbitrary code from the repository.

Repository tools should operate only within the controlled repository workspace.

---

# 7. Testing Requirements

Every important module should have tests.

At minimum test:

* repository ingestion
* file filtering
* chunking
* metadata
* vector retrieval
* tools
* agent behavior
* citation/source generation
* error handling

Add integration tests for the complete RAG pipeline where practical.

---

# 8. Observability

During development, make it possible to inspect:

* user query
* retrieved documents
* retrieval scores
* tool calls
* final context
* model response
* latency

Do not expose secrets.

This observability will be important for debugging and evaluation.

---

# 9. Engineering Rules

Follow these rules:

1. Do not generate the entire project at once.
2. Do not introduce LangGraph.
3. Do not introduce unnecessary frameworks.
4. Do not use fake evaluation metrics.
5. Do not hide generated code from me.
6. Explain significant architectural decisions.
7. Prefer simple implementations.
8. Write tests before declaring a feature complete.
9. Keep modules small and understandable.
10. Use type hints where appropriate.
11. Handle errors explicitly.
12. Never hard-code API keys.
13. Keep README documentation synchronized with the implementation.
14. Do not execute arbitrary repository code.
15. Do not add features simply because they sound impressive.

---

# 10. Interview-First Development

For every major feature, maintain a short document in:

docs/interview-notes/

For each feature document:

### Problem

What problem does this component solve?

### Design

How does it work?

### Alternatives

What alternatives were considered?

### Why this approach?

Why was this implementation selected?

### Tradeoffs

What are its limitations?

### Interview Questions

List questions an interviewer could ask about this component.

Example:

For retrieval:

* Why embeddings?
* Why this embedding model?
* Why vector search?
* Why metadata?
* Why top-k?
* Why hybrid retrieval?
* How did you evaluate retrieval?
* What happens when retrieval returns irrelevant chunks?

This documentation is extremely important because the goal is to make me capable of explaining the project in interviews.

---

# 11. Git Workflow

Use meaningful commits.

Examples:

feat: add repository ingestion
feat: implement code-aware chunking
feat: add HuggingFace embeddings
feat: implement Chroma retrieval
feat: build basic RAG pipeline
feat: add repository search tools
feat: add LangChain agent
feat: add conversation memory
feat: add RAG evaluation
feat: add Streamlit interface
feat: prepare Hugging Face deployment

Do not make one giant commit containing the entire application.

---

# 12. First Task

Do NOT build the application yet.

First:

1. Analyze this specification.
2. Propose the final architecture.
3. Propose the project directory structure.
4. Identify the minimum dependencies for Phase 1.
5. Explain the development roadmap.
6. Identify potential technical risks.
7. Explain which parts I should understand deeply for interviews.

Then STOP and wait for my approval.

Do not write the implementation until I approve the architecture.
# AI Software Engineering Copilot

An AI-powered developer assistant that understands GitHub repositories,
performs code-aware search, answers repository questions, traces dependencies,
reviews code, and generates tests using hybrid retrieval and tool-calling agents.

## 🚀 Live Demo

[Try the Live Application]

## ✨ Key Features

- Repository-level Q&A
- Semantic + structural + symbol-based code search
- Function and reference lookup
- Dependency tracing
- Code review assistance
- Test generation
- Repository architecture exploration
- Grounded RAG to reduce hallucinations
- Tool-calling agent workflow
- Conversation memory
- Evaluation metrics

## 🏗️ Architecture

User
 ↓
Streamlit UI
 ↓
Agent / Router
 ↓
┌─────────────────────────────┐
│ Repository Analysis Tools   │
│ • Code Search               │
│ • Function Search           │
│ • Reference Search          │
│ • File Reader               │
│ • Dependency Tracing        │
└─────────────────────────────┘
 ↓
Hybrid Retrieval
 ↓
Vector Store + Code Metadata
 ↓
LLM
 ↓
Grounded Response

## 🔍 Retrieval Pipeline

1. Repository ingestion
2. File filtering
3. Code-aware chunking
4. Metadata extraction
5. Embedding generation
6. Hybrid retrieval
7. Context construction
8. LLM generation

## 🧠 Why Hybrid Retrieval?

Traditional semantic search can miss important relationships in source code.

This project combines:

- Semantic search
- File-level filtering
- Symbol/function search
- Structural relationships
- Reference lookup

to retrieve more relevant code context.

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain / LangGraph
- Google Gemini
- Vector Database
- HuggingFace Embeddings
- RAG
- Agentic Tool Calling

## 📂 Project Structure

...

## ⚙️ Installation

...

## 🧪 Evaluation

Explain:
- What was evaluated
- Retrieval accuracy
- Answer grounding
- Tool selection
- Failure cases

## 🎯 Use Cases

- Understand unfamiliar repositories
- Find implementations quickly
- Trace function dependencies
- Perform AI-assisted code review
- Generate tests
- Explore project architecture

## 🔮 Future Improvements

- Multi-language repository support
- Improved code graph analysis
- Automated PR review
- GitHub integration
- Repository-wide architecture visualization

## 👨‍💻 Author

Abhishek Kumar

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.repository_agent import RepositoryAgent
from app.core.config import settings
from app.core.repository import RepositoryIndexOptions, RepositoryIndexer
from app.memory.conversation import ConversationMemory
from app.retrieval.rag import RepositoryRAG
from app.retrieval.vector_store import CodeVectorStore
from app.tools.repository_tools import RepositoryTools


st.set_page_config(page_title="AI Software Engineering Copilot", layout="wide")
st.title("AI Software Engineering Copilot")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "corpus" not in st.session_state:
    st.session_state.corpus = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "repo_source" not in st.session_state:
    st.session_state.repo_source = ""

with st.sidebar:
    st.subheader("Repository")
    repo_url_text = st.text_input(
        "GitHub repository URL",
        value=st.session_state.repo_source,
        placeholder="https://github.com/owner/repository.git",
    )
    parsed_repo_name = Path(urlparse(repo_url_text).path).name if repo_url_text else ""
    if parsed_repo_name.endswith(".git"):
        parsed_repo_name = parsed_repo_name[:-4]
    repo_id_text = st.text_input("Repository id", value=parsed_repo_name or settings.default_repository_id)
    persist_dir = st.text_input("Index directory", value=settings.chroma_persist_dir)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)

    if st.button("Index repository"):
        try:
            indexer = RepositoryIndexer()
            if not repo_url_text.strip():
                raise ValueError("Enter a GitHub repository URL first.")
            cloned_path = indexer.clone_github_repository(repo_url_text.strip())
            corpus = indexer.build_corpus(
                RepositoryIndexOptions(
                    root_dir=cloned_path,
                    repository_id=repo_id_text,
                )
            )
            st.session_state.repo_source = repo_url_text.strip()
            vector_store = CodeVectorStore(persist_dir=persist_dir)
            vector_store.clear_repository(corpus.repository_id)
            vector_store.add_chunks([{"content": chunk.content, "metadata": chunk.metadata} for chunk in corpus.chunks], corpus.repository_id)
            tools = RepositoryTools(corpus=corpus, vector_store=vector_store)
            rag = RepositoryRAG(vector_store=vector_store)
            st.session_state.corpus = corpus
            st.session_state.vector_store = vector_store
            st.session_state.agent = RepositoryAgent(rag=rag, tools=tools, memory=st.session_state.memory)
            st.success(f"Cloned to {cloned_path} and indexed {len(corpus.files)} files and {len(corpus.chunks)} chunks.")
        except Exception as exc:  # pragma: no cover - UI guard
            st.error(str(exc))

st.subheader("Chat")
question = st.text_input("Ask a repository question", value="How does authentication work?")
run_question = st.button("Ask")

if run_question:
    if st.session_state.agent is None:
        st.warning("Index a repository first.")
    else:
        response = st.session_state.agent.answer(question, top_k=top_k)

        st.markdown("### Answer")
        st.write(response["answer"])

        if "sources" in response:
            st.markdown("### Sources")
            for source in response["sources"]:
                score = source.score if source.score is not None else 0.0
                st.write(
                    f"{source.file_path} | {source.symbol} | lines {source.start_line}-{source.end_line} | score {score:.3f}"
                )

        if "retrieved_chunks" in response:
            st.markdown("### Retrieved chunks")
            for chunk in response["retrieved_chunks"]:
                with st.expander(f"{chunk.metadata.get('file_path')} :: {chunk.metadata.get('symbol')} ({chunk.score:.3f})"):
                    st.code(chunk.content, language=str(chunk.metadata.get("language", "")))
                    st.json(chunk.metadata)

        if "context_text" in response:
            st.markdown("### Built context")
            st.code(response["context_text"])

        if "trace" in response:
            st.markdown("### Trace")
            st.json(response["trace"])

st.markdown("### Conversation memory")
st.code(st.session_state.memory.recent_context())

from __future__ import annotations

import json
from dataclasses import asdict
import sys
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.repository_agent import RepositoryAgent
from app.analysis.repository_intelligence import RepositoryIntelligence
from app.core.config import settings
from app.core.repository import RepositoryIndexOptions, RepositoryIndexer
from app.evaluation.benchmark import RepositoryBenchmarkRunner
from app.memory.conversation import ConversationMemory
from app.retrieval.hybrid import HybridRetriever
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
if "intelligence" not in st.session_state:
    st.session_state.intelligence = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "benchmark_runner" not in st.session_state:
    st.session_state.benchmark_runner = None
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
    capability = st.selectbox(
        "Capability",
        [
            "Codebase Q&A",
            "Intelligent code search",
            "Repository architecture analysis",
            "Dependency/module analysis",
            "Code review",
            "Documentation generation",
            "Unit-test generation",
            "Benchmark retrieval",
        ],
    )

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
            retriever = HybridRetriever(corpus=corpus, vector_store=vector_store)
            rag = RepositoryRAG(vector_store=vector_store, retriever=retriever)
            intelligence = RepositoryIntelligence(tools=tools, rag=rag, retriever=retriever)
            benchmark_runner = RepositoryBenchmarkRunner(corpus=corpus, vector_store=vector_store, hybrid_retriever=retriever)
            st.session_state.corpus = corpus
            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            st.session_state.intelligence = intelligence
            st.session_state.benchmark_runner = benchmark_runner
            st.session_state.agent = RepositoryAgent(rag=rag, tools=tools, memory=st.session_state.memory, intelligence=intelligence)
            st.success(f"Cloned to {cloned_path} and indexed {len(corpus.files)} files and {len(corpus.chunks)} chunks.")
        except Exception as exc:  # pragma: no cover - UI guard
            st.error(str(exc))

st.subheader(capability)

corpus = st.session_state.corpus
target_file = None
target_symbol = None
main_input_label = "Question"
main_input_value = "How does authentication work?"

if capability == "Intelligent code search":
    main_input_label = "Search query"
    main_input_value = "authentication"
elif capability == "Repository architecture analysis":
    main_input_label = "Architecture focus"
    main_input_value = "Explain the repository structure"
elif capability == "Dependency/module analysis":
    main_input_label = "Dependency focus"
    main_input_value = "Show internal imports and module boundaries"
elif capability == "Code review":
    main_input_label = "Review scope"
    main_input_value = "Review the repository for issues"
elif capability == "Documentation generation":
    main_input_label = "Documentation scope"
    main_input_value = "Generate documentation for the repository"
elif capability == "Unit-test generation":
    main_input_label = "Test scope"
    main_input_value = "Generate tests for the repository"
elif capability == "Benchmark retrieval":
    main_input_label = "Benchmark input"
    main_input_value = "Upload a JSON benchmark file or use the auto-generated cases"

if corpus is not None and capability in {"Code review", "Documentation generation", "Unit-test generation"}:
    file_options = [""] + [file.file_path for file in corpus.files]
    target_file = st.selectbox("Target file", file_options)
    target_symbol = st.text_input("Target symbol", value="")

benchmark_upload = None
if capability == "Benchmark retrieval":
    benchmark_upload = st.file_uploader("Benchmark cases JSON", type=["json"])

main_input = st.text_input(main_input_label, value=main_input_value)
run_action = st.button("Run")

if run_action:
    if st.session_state.agent is None or st.session_state.intelligence is None:
        st.warning("Index a repository first.")
    else:
        try:
            if capability == "Codebase Q&A":
                response = st.session_state.agent.answer(main_input, top_k=top_k)
                st.markdown("### Answer")
                st.write(response["answer"])
                if "sources" in response:
                    st.markdown("### Sources")
                    for source in response["sources"]:
                        if isinstance(source, dict):
                            score = float(source.get("score") or 0.0)
                            st.write(
                                f"{source.get('file_path')} | {source.get('symbol')} | lines "
                                f"{source.get('start_line')}-{source.get('end_line')} | score {score:.3f}"
                            )
                        else:
                            score = source.score if source.score is not None else 0.0
                            st.write(f"{source.file_path} | {source.symbol} | lines {source.start_line}-{source.end_line} | score {score:.3f}")
                if "retrieved_chunks" in response:
                    st.markdown("### Retrieved chunks")
                    for chunk in response["retrieved_chunks"]:
                        if isinstance(chunk, dict):
                            metadata = dict(chunk.get("metadata", {}))
                            score = float(chunk.get("score") or 0.0)
                            content = str(chunk.get("content", ""))
                        else:
                            metadata = dict(chunk.metadata)
                            score = chunk.score
                            content = chunk.content
                        with st.expander(f"{metadata.get('file_path')} :: {metadata.get('symbol')} ({score:.3f})"):
                            st.code(content, language=str(metadata.get("language", "")))
                            st.json(metadata)
                if "context_text" in response:
                    st.markdown("### Built context")
                    st.code(response["context_text"])
                if "trace" in response:
                    st.markdown("### Trace")
                    st.json(response["trace"])

            elif capability == "Intelligent code search":
                results = st.session_state.intelligence.code_search(main_input, top_k=top_k)
                st.write(f"{len(results)} result(s) found.")
                for item in results:
                    with st.expander(f"{item.metadata.get('file_path')} :: {item.metadata.get('symbol')} ({item.score:.3f})"):
                        st.code(item.content, language=str(item.metadata.get("language", "")))
                        st.json(item.metadata)

            elif capability == "Repository architecture analysis":
                report = st.session_state.intelligence.architecture_analysis()
                st.markdown("### Summary")
                st.write(report.summary)
                st.markdown("### Top-level directories")
                st.dataframe(report.top_level_directories, use_container_width=True)
                st.markdown("### Entry points")
                st.write(report.entrypoints or ["None detected"])
                st.markdown("### Languages")
                st.json(report.languages)
                st.markdown("### Dependency edges")
                st.dataframe(report.dependency_edges, use_container_width=True)
                st.markdown("### Notable symbols")
                st.dataframe(report.notable_symbols, use_container_width=True)

            elif capability == "Dependency/module analysis":
                report = st.session_state.intelligence.dependency_analysis()
                st.markdown("### Summary")
                st.write(report.summary)
                st.markdown("### Internal edges")
                st.dataframe(report.internal_edges, use_container_width=True)
                st.markdown("### External imports")
                st.write(report.external_imports or ["None detected"])
                st.markdown("### Fan-in")
                st.json(report.fan_in)
                st.markdown("### Fan-out")
                st.json(report.fan_out)

            elif capability == "Code review":
                report = st.session_state.intelligence.code_review(target_file=target_file or None)
                st.markdown("### Summary")
                st.write(report.summary)
                st.markdown("### Findings")
                st.dataframe([asdict(issue) for issue in report.issues], use_container_width=True)

            elif capability == "Documentation generation":
                artifact = st.session_state.intelligence.generate_documentation(
                    target_file=target_file or None,
                    symbol=target_symbol.strip() or None,
                )
                st.markdown("### Documentation")
                st.markdown(artifact.markdown)
                if artifact.sources:
                    st.markdown("### Sources")
                    st.json([asdict(source) for source in artifact.sources])

            elif capability == "Unit-test generation":
                artifact = st.session_state.intelligence.generate_unit_tests(
                    target_file=target_file or None,
                    symbol=target_symbol.strip() or None,
                )
                st.markdown("### Generated tests")
                if artifact.markdown:
                    st.markdown(artifact.markdown)
                if artifact.code:
                    st.code(artifact.code, language="python")
                else:
                    st.info("This generator currently emits Python/pytest skeletons only.")

            elif capability == "Benchmark retrieval":
                if st.session_state.benchmark_runner is None:
                    raise RuntimeError("Index a repository first.")
                if benchmark_upload is not None:
                    payload = json.loads(benchmark_upload.getvalue().decode("utf-8"))
                    cases = RepositoryBenchmarkRunner.load_cases_from_payload(payload)
                else:
                    cases = st.session_state.benchmark_runner.auto_cases(limit=5)
                report = st.session_state.benchmark_runner.run(cases, top_k=top_k)
                st.markdown("### Benchmark summary")
                st.json(report.summary())
                st.markdown("### Benchmark rows")
                st.dataframe(report.to_table(), use_container_width=True)

        except Exception as exc:  # pragma: no cover - UI guard
            st.error(str(exc))

st.markdown("### Conversation memory")
if st.session_state.memory.turns:
    st.code(st.session_state.memory.recent_context())
else:
    st.caption("No conversation history yet.")

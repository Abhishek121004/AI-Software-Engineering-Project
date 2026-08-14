# Interview Notes: Remaining Phases

## Phase 5-6: Basic RAG and Retrieval Improvement

### Problem
We need grounded answers backed by repository evidence, not generic LLM output.

### Design
The pipeline retrieves semantically similar chunks, boosts exact identifier matches, and builds a compact context block with file paths, symbols, and line numbers.

### Tradeoffs
The current backend is intentionally simple. It is deterministic and inspectable, but it does not yet do full reranking or learned hybrid search.

### Interview Questions
- Why do you combine semantic retrieval with exact identifier matching?
- How do you prevent the model from answering without evidence?

## Phase 7-8: Repository Tools and Agent Routing

### Problem
Some repository questions are better answered by reading a file or finding references than by retrieval alone.

### Design
We expose explicit tools for search, file reading, symbol lookup, reference lookup, and tree inspection. A lightweight agent routes questions to the right tool or to RAG.

### Tradeoffs
The router is heuristic-based rather than a full LangChain agent. That keeps it reliable and easy to inspect, but it is less flexible than a learned planner.

### Interview Questions
- When would you choose a tool over RAG?
- How do you prevent the tools from escaping the repository root?

## Phase 9-10: Memory and Context Engineering

### Problem
Follow-up questions need conversational continuity, but the system should not accumulate hidden long-term state.

### Design
A small conversation memory keeps recent turns. The context builder combines memory, retrieval results, and metadata into a visible prompt block.

### Tradeoffs
This is not a full conversational memory system. It is simple on purpose so the model context can be inspected and debugged.

## Phase 11-12: Grounding and Evaluation

### Problem
We need to know whether the system is answering from evidence and whether retrieval is improving over time.

### Design
Answers always expose sources. The evaluation helpers compute retrieval metrics like recall@k, precision@k, and MRR, plus simple generation and context overlap measures.

### Tradeoffs
The generation metrics are lightweight heuristics. They are useful for local development, but not a substitute for a full benchmark suite.


# Memory Agent — Multi-System Memory Orchestrator

A modular memory layer for AI agents that coordinates multiple memory types to provide useful context retrieval and storage.

## Memory Systems

- STM (Short-term): Recent conversation turns and working memory
- Semantic: Facts, preferences, and stable user knowledge
- Episodic: Events, experiences, and temporal context
- RAG: Document-based knowledge retrieval (chunking + embeddings)

## Features

- Deterministic routing with optional LLM arbiter fallback
- Multi-tenant isolation (tenant → user → conversation)
- Token budget management and reranking/merging
- Confidence/relevance scoring and thresholds
- Redis-backed STM with in-memory fallback
- Comprehensive CLI test suite with edge cases

## Requirements

- Python 3.10+
- Redis (optional, recommended for STM)
- See `memory-agent/requirements.txt` for Python packages

## Installation

```bash
git clone <this-repo>
cd <repo-root>
python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell
# or: source .venv/bin/activate  # macOS/Linux

pip install -r memory-agent/requirements.txt
```

## Configuration

Set environment variables (via `.env` or your environment):

```bash
# Core
OPENAI_API_KEY=your_openai_api_key_here        # For LLM arbiter (optional)

# Enable/disable systems
MEMORY_ENABLE_SEMANTIC=true
MEMORY_ENABLE_EPISODIC=true
MEMORY_ENABLE_RAG=true
MEMORY_ENABLE_STM=true
MEMORY_TOKEN_BUDGET=4000

# STM / Redis
MEMORY_USE_REDIS_STM=true
MEMORY_REDIS_HOST=127.0.0.1
MEMORY_REDIS_PORT=6379
MEMORY_REDIS_DB=0
MEMORY_REDIS_PASSWORD=
MEMORY_REDIS_SSL=false
MEMORY_REDIS_PREFIX=memory_agent

# Semantic fallback behavior
MEMORY_SEMANTIC_INCLUDE_LOWCONF=true
MEMORY_SEMANTIC_LOWCONF_K=2
```

Notes:
- If Redis is not reachable, STM falls back to in-memory automatically.
- SentenceTransformer embeddings are cached in-process to reduce latency.
- ChromaDB is optional; in-memory vector search is used by default.

## Usage

```python
from memory_agent import handle, post_write

# Persist a conversation turn
await post_write(
    user_text="I prefer Italian food over Chinese food",
    assistant_text="Got it — I'll remember that",
    tenant_id="org1",
    user_id="user123",
    agent_id="assistant",
    conversation_id="conv_456"
)

# Retrieve merged context for a new query
result = await handle(
    query="What are my food preferences?",
    tenant_id="org1",
    user_id="user123",
    agent_id="assistant",
    conversation_id="conv_456"
)
print(result["merged_context"])  # Sectioned STM/SEM/EPI/RAG context
```

## Testing

Run the comprehensive test suite:

```bash
python test_memory_agent.py
```

Or via the test runner:

```bash
python run_tests.py
```

## Project Structure

```
memory-agent/
  src/memory_agent/
    agent.py         # Orchestrator
    router.py        # Deterministic routing
    arbiter.py       # LLM fallback
    stm.py           # STM manager (Redis + memory)
    memory_store.py  # Semantic/Episodic manager
    rag.py           # RAG manager (ingest/query)
    services.py      # Reranker/merging
    config.py        # Configuration + env overrides
```

## Optional Backends

- Vector store: install `chromadb` to persist embeddings to DuckDB/Parquet.
- STM: connect Redis to persist conversation context across runs.

## Tips

- For offline runs, set `MEMORY_ENABLE_LLM_ARBITER=false`.
- Adjust token allocations in `config.py` to tune merged context budgets.
- Review `test_memory_agent.py` for end-to-end usage scenarios.

## License

MIT License — see LICENSE if provided.


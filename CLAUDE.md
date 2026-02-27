# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Use `uv` for all dependency management and running Python. Never use `pip` directly.

**Install dependencies:**
```bash
uv sync
```

**Add a dependency:**
```bash
uv add <package>
```

**Run the application:**
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app is served at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

**Environment setup:** Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## Architecture

This is a full-stack RAG (Retrieval-Augmented Generation) chatbot. The FastAPI backend serves both the API and the static frontend from a single port.

### Query Pipeline

User queries go through a two-pass Claude API call pattern:

1. **First call**: Claude receives the query + system prompt + conversation history + tool definitions. It either answers directly or requests a `search_course_content` tool call.
2. **Tool execution** (if needed): `CourseSearchTool` queries ChromaDB via `VectorStore.search()`, returning semantically matched course chunks.
3. **Second call**: Claude receives the original messages + tool results and synthesizes a final answer.

### Key Components

- **`backend/rag_system.py`** — Central orchestrator. Wires together all components and exposes `query()` and `add_course_folder()`.
- **`backend/ai_generator.py`** — Handles all Claude API interactions. Contains the system prompt and the two-pass tool execution logic in `_handle_tool_execution()`.
- **`backend/vector_store.py`** — ChromaDB wrapper. Stores both course metadata and chunked content in separate collections. Supports filtering by course name (fuzzy) and lesson number.
- **`backend/search_tools.py`** — Defines the `search_course_content` tool in Anthropic tool-calling format. `ToolManager` registers tools and tracks sources from the last search for display in the UI.
- **`backend/session_manager.py`** — In-memory conversation history, keyed by session ID. Keeps last `MAX_HISTORY` (default: 2) exchanges per session.
- **`backend/document_processor.py`** — Parses `.txt`/`.pdf`/`.docx` files from `docs/` into `Course` and `CourseChunk` objects for ingestion.

### Configuration (`backend/config.py`)

Key tunables via `Config` dataclass:
- `ANTHROPIC_MODEL` — Claude model used for generation
- `EMBEDDING_MODEL` — `all-MiniLM-L6-v2` via sentence-transformers
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — Controls document chunking (800 / 100 chars)
- `MAX_RESULTS` — Number of vector search results returned to Claude (default: 5)
- `MAX_HISTORY` — Conversation turns retained per session (default: 2)
- `CHROMA_PATH` — ChromaDB persistence directory (`./chroma_db` relative to `backend/`)

### Document Ingestion

On startup, `app.py` calls `rag_system.add_course_folder("../docs")`. It skips courses already present in ChromaDB by title, so re-runs are safe. Add new course files (`.txt`, `.pdf`, `.docx`) to `docs/` and restart to ingest them.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git

Always use single-line commit messages.

## Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Index Notion pages (fetches and embeds content)
python cli.py index

# Query indexed content
python cli.py ask "your question"

# List indexed pages
python cli.py sources

# Start API server (for Open WebUI)
python api.py
```

## Architecture

This is a local RAG system for querying Notion workspaces using Ollama.

**Data flow:**
1. `notion.py` fetches pages recursively from Notion API starting from ROOT_PAGES
2. `indexer.py` chunks text and generates embeddings via Ollama (nomic-embed-text)
3. `vectorstore.py` stores embeddings in numpy arrays with JSON metadata (no external DB)
4. `query.py` embeds the question, finds similar chunks via cosine similarity, and generates answers with Ollama LLM (mistral)

**Entry points:**
- `cli.py` - Command-line interface with index/ask/sources commands
- `api.py` - FastAPI server exposing OpenAI-compatible `/v1/chat/completions` endpoint for Open WebUI integration

**Key configuration (`src/config.py`):**
- `ROOT_PAGES` - Notion page URLs/IDs to index (includes children recursively)
- `EMBEDDING_MODEL` / `LLM_MODEL` - Ollama models for embedding and generation
- `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K_RESULTS` - RAG parameters

**Storage:**
- `db/embeddings.npy` - Numpy array of embedding vectors
- `db/metadata.json` - Document text and metadata (title, URL, page_id)

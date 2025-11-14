# Agentic RAG with LangChain + Weaviate (example)

This repository contains a minimal example showing how to build an agentic Retrieval-Augmented Generation (RAG) system using LangChain and Weaviate to store vectors for scraped website content.

Quick overview
- Start a local Weaviate instance (Docker Compose included).
- Ingest your scraped data (place files in `data/` directory) with `ingest.py`.
- Run `agent.py` to start a LangChain agent that can query Weaviate as a tool.

Prerequisites
- Docker (for running local Weaviate) or a Weaviate cloud instance.
- Python 3.9+ and a virtual environment.
- Optional: OpenAI API key if you choose OpenAI embeddings/LLM.

Setup (PowerShell)
```powershell
# 1. Start Weaviate (optional, local)
docker-compose up -d

# 2. Create venv and install deps
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

# 3. Copy `.env.example` to `.env` and fill values
Copy-Item .env.example .env
# Edit .env to add your OPENAI_API_KEY if using OpenAI embeddings/LLM

# 4. Prepare data
# Put scraped pages (HTML or .txt) into `data/` directory

# 5. Run ingestion
python ingest.py

# 6. Run the agent
python agent.py
```

Notes
- `ingest.py` supports both OpenAI embeddings and HuggingFace (sentence-transformers). Set `EMBEDDING_PROVIDER` in `.env`.
- This example embeds client-side and stores vectors in Weaviate; it does not require any Weaviate vectorizer module.

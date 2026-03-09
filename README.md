# 🎓 Ask Vibs — RAG-Powered NLP Professor Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that acts as
an NLP professor for BANA275. It answers questions from class lecture slides and
AI/ML reference materials using hybrid search and Google Gemini.

**Live demo:** `https://sarahjiwonlee-hue-ask-vibs.streamlit.app` ← update after deployment

---

## Features

| Feature | Implementation |
|---|---|
| Document Processing | PDF · HTML · TXT via `DocumentProcessor` |
| Chunking | RecursiveCharacterTextSplitter (1000 chars, 200 overlap) |
| Vector Database | ChromaDB persistent storage (3,385 chunks) |
| Embeddings | `all-MiniLM-L6-v2` (local HuggingFace, free) |
| LLM | Google Gemini 2.5 Flash (free tier) |
| Retrieval | **Hybrid Search** — BM25 (0.3) + Semantic (0.7) via EnsembleRetriever |
| Conversation Memory | Sliding window — last 7 exchanges |
| Source Citations | Expandable citation cards (title, file, chunk, snippet) |
| Multiple Chats | Create and switch between named conversation logs |
| Quiz Mode | 10-question MCQ on chosen topic, graded A–F with retake support |
| Export | Download any conversation as a `.txt` file |
| Deployment | Streamlit Community Cloud (free public URL) |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  Step 1: Contextualize              │
│  Rephrase follow-up Qs using        │
│  conversation memory (last 7 turns) │
└──────────────┬──────────────────────┘
               │ standalone question
               ▼
┌─────────────────────────────────────┐
│  Step 2: Hybrid Retrieve (top-K)    │
│  ┌──────────────┐ ┌───────────────┐ │
│  │ BM25         │ │ ChromaDB      │ │
│  │ (keyword)    │ │ (semantic)    │ │
│  │ weight: 0.3  │ │ weight: 0.7   │ │
│  └──────┬───────┘ └──────┬────────┘ │
│         └────────┬────────┘         │
│          Reciprocal Rank Fusion     │
└──────────────────┬──────────────────┘
               │ top-5 chunks
               ▼
┌─────────────────────────────────────┐
│  Step 3: Generate Answer            │
│  Gemini 2.5 Flash + context         │
│  → answer as "Vibs, NLP professor"  │
└──────────────────────────────────────┘
               │
               ▼
        Streamlit Chat UI
        + expandable citations
```

---

## Quick Start (Local)

### 1. Clone and install

```bash
git clone https://github.com/sarahjiwonlee-hue/ask-vibs.git
cd ask-vibs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your Google API key

```bash
cp .env.example .env
# Edit .env and set: GOOGLE_API_KEY=AIza...
```

Get a free key at https://aistudio.google.com/apikey

### 3. Run the app

```bash
streamlit run app/main.py
# Opens at http://localhost:8501
```

The knowledge base (`chroma_db/`) is pre-built and committed to the repo —
no re-ingestion needed on first run.

---

## Deployment (Streamlit Community Cloud)

1. Push repo to GitHub
2. Go to https://share.streamlit.io → **New app**
3. Select repo, branch `main`, main file `app/main.py`
4. **Advanced settings → Secrets**, paste:
   ```toml
   GOOGLE_API_KEY = "AIza..."
   LLM_MODEL = "gemini-2.5-flash"
   ```
5. Click **Deploy**

---

## Project Structure

```
ask-vibs/
├── app/
│   ├── main.py              # Streamlit UI (chat, quiz, conversation logs)
│   ├── rag_pipeline.py      # Query → contextualize → retrieve → generate
│   ├── document_processor.py # Multi-format loader + chunker
│   ├── vector_store.py      # ChromaDB persistent wrapper
│   ├── retriever.py         # Hybrid BM25 + semantic EnsembleRetriever
│   └── config.py            # All tunable settings
├── data/
│   └── raw/                 # 65 source documents (57 Wikipedia + 8 BANA275 slides)
├── chroma_db/               # Pre-built vector store (committed to repo)
├── scripts/
│   ├── ingest.py            # Re-ingestion pipeline
│   └── download_docs.py     # Downloads Wikipedia articles
├── tests/
│   └── test_pipeline.py     # Pytest smoke tests
├── .env.example             # Copy to .env and fill in API key
├── requirements.txt
└── README.md
```

---

## Configuration

All settings in `app/config.py` or via environment variables / Streamlit secrets:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | **Required** |
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `MAX_MEMORY_EXCHANGES` | `7` | Conversation turns to remember |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT License. Wikipedia content is CC BY-SA 4.0.

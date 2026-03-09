# 🤖 AI/ML Domain Expert Chatbot — RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers
questions about Artificial Intelligence and Machine Learning using a curated
knowledge base of 55 Wikipedia articles (~4,000 chunks).

**Live demo:** `https://YOUR-APP.streamlit.app` ← replace after deployment

---

## Features

| Feature | Implementation |
|---|---|
| Document Processing | PDF · HTML · TXT · MD via `DocumentProcessor` |
| Chunking | RecursiveCharacterTextSplitter (1000 chars, 200 overlap) |
| Vector Database | ChromaDB persistent storage |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Retrieval | **Hybrid Search** — BM25 (0.3) + Semantic (0.7) via EnsembleRetriever |
| Conversation Memory | Sliding window — last 7 exchanges |
| Source Citations | Inline `[Source N]` + expandable citation cards in UI |
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
│  GPT-4o-mini + context + question   │
│  → answer with [Source N] citations │
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
git clone https://github.com/YOUR-USERNAME/rag-chatbot.git
cd rag-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and set:  OPENAI_API_KEY=sk-...
```

Get a key at https://platform.openai.com/api-keys

### 3. Download source documents

```bash
python scripts/download_docs.py
# Downloads 55 Wikipedia articles → data/raw/*.txt
# Takes ~60 seconds
```

You can also add your own PDFs, HTML files, or text files to `data/raw/`.

### 4. Ingest documents into ChromaDB

```bash
python scripts/ingest.py
# Embeds all chunks and stores in chroma_db/
# Cost: ~$0.02 for 55 Wikipedia articles
# Takes ~2-3 minutes
```

### 5. Run the app

```bash
streamlit run app/main.py
# Opens at http://localhost:8501
```

---

## Deployment (Streamlit Community Cloud — free public URL)

### Prerequisites
- GitHub account
- OpenAI API key
- The `chroma_db/` directory populated (run `scripts/ingest.py` first)

### Steps

**1. Push to GitHub**
```bash
git init
git add .
git add -f chroma_db/    # must force-add (not ignored)
git commit -m "Initial RAG chatbot"
git remote add origin https://github.com/YOUR-USERNAME/rag-chatbot.git
git push -u origin main
```

**2. Deploy on Streamlit Cloud**
1. Go to https://share.streamlit.io
2. Click **New app**
3. Select your GitHub repo
4. Set **Main file path** to `app/main.py`
5. Click **Advanced settings** → **Secrets** and paste:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
6. Click **Deploy**

Your app will be live at `https://YOUR-USERNAME-rag-chatbot-appmain-XXXX.streamlit.app`

---

## Adding Your Own Documents

Drop files into `data/raw/` and re-run ingestion:

```bash
# Supported formats:
cp my_paper.pdf data/raw/
cp company_docs.html data/raw/
cp notes.txt data/raw/

python scripts/ingest.py --clear   # rebuild from scratch
# OR
python scripts/ingest.py           # append to existing
```

Minimum 50 documents required for the assignment. Use `--dry-run` to preview
chunk counts before embedding.

---

## Project Structure

```
rag-chatbot/
├── app/
│   ├── main.py              # Streamlit UI (chat + citations)
│   ├── rag_pipeline.py      # Query → contextualize → retrieve → generate
│   ├── document_processor.py # Multi-format loader + RecursiveCharacterTextSplitter
│   ├── vector_store.py      # ChromaDB persistent wrapper
│   ├── retriever.py         # Hybrid BM25 + semantic EnsembleRetriever
│   └── config.py            # All tunable settings
├── data/
│   └── raw/                 # Source documents (PDF, HTML, TXT, MD)
├── chroma_db/               # Persistent vector store (commit this!)
├── scripts/
│   ├── ingest.py            # One-time ingestion pipeline
│   └── download_docs.py     # Downloads 55 Wikipedia articles
├── tests/
│   └── test_pipeline.py     # Pytest smoke tests
├── .env.example             # Copy to .env and fill in API key
├── requirements.txt
├── packages.txt             # System deps for Streamlit Cloud
└── README.md
```

---

## Chunking Strategy

**Chosen:** `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=200`

**Why this over alternatives:**

| Strategy | Pros | Cons |
|---|---|---|
| Fixed-size | Fast, predictable | Cuts mid-sentence; incoherent chunks |
| Sentence-based | Preserves sentences | Fails on long technical paragraphs |
| Paragraph-based | Good semantics | Wildly variable chunk sizes |
| **Recursive (ours)** | **Hierarchical**: `\n\n → \n → . → space`; breaks at the largest natural boundary that fits | Slightly slower |

The 20% overlap (200 chars) ensures that sentences spanning chunk boundaries
appear in both adjacent chunks, preventing context loss at retrieval time.

---

## Advanced Feature: Hybrid Search

Implemented via LangChain's `EnsembleRetriever` combining:
- **BM25** (weight=0.3): Okapi BM25 sparse keyword retrieval via `rank-bm25`
- **Semantic** (weight=0.7): ChromaDB dense vector cosine similarity

Scores are merged using **Reciprocal Rank Fusion (RRF)**: `score = Σ w_i / (60 + rank_i)`

This outperforms pure semantic search on:
- Exact technical terms (`BERT`, `backpropagation`, `LSTM`)
- Rare words not well-represented in the embedding space
- Multi-word technical phrases

---

## Configuration

All settings in `app/config.py` or via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `MAX_MEMORY_EXCHANGES` | `7` | Conversation turns to remember |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: chunk ID assignment, metadata defaults, memory truncation,
hybrid deduplication, and mocked end-to-end query flow.

---

## Cost Estimates

| Operation | Cost |
|---|---|
| Ingestion (55 articles, ~4K chunks) | ~$0.02 |
| Per query (embedding + GPT-4o-mini) | ~$0.001 |
| 1,000 queries | ~$1.00 |

---

## License

MIT License. Wikipedia content is CC BY-SA 4.0.

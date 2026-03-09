# Rubric Checklist — RAG Domain Expert Chatbot

Use this to verify you've met every requirement before submission.

---

## 1. Domain Selection ✅

- [x] **Domain chosen:** Artificial Intelligence & Machine Learning
- [x] **At least 50 documents:** 55 Wikipedia articles downloaded by `scripts/download_docs.py`
- [x] **Documents are domain-relevant:** Covers AI, ML, DL, NLP, CV, RL, Ethics

**To verify:**
```bash
ls data/raw/ | wc -l   # Should show 55
```

---

## 2. Document Processing ✅

- [x] **PDF support:** `DocumentProcessor.load_pdf()` using `pypdf`
- [x] **HTML support:** `DocumentProcessor.load_html()` using `BeautifulSoup`
- [x] **Text/MD support:** `DocumentProcessor.load_text()`
- [x] **Metadata extracted per document:**
  - `title` — document or page title
  - `source` — filename
  - `doc_type` — pdf / html / wikipedia / text / markdown
  - `page` — page number (PDFs only)
  - `url` — original URL (Wikipedia articles)
  - `chunk_id` — unique per chunk (`{md5hash}_{index}`)
- [x] **Chunking strategy chosen and justified:** RecursiveCharacterTextSplitter
- [x] **Justification in write-up:** Section 2.1 of `technical_writeup.md`

**To verify:**
```python
from app.document_processor import DocumentProcessor
proc = DocumentProcessor()
chunks = proc.process_directory("data/raw")
print(chunks[0].metadata)   # Should show all metadata fields
```

---

## 3. Vector Database ✅

- [x] **Vector DB chosen:** ChromaDB
- [x] **Stores chunks:** Yes — all ~4,000 chunks stored
- [x] **Stores embeddings:** Yes — OpenAI text-embedding-3-small (1536-dim)
- [x] **Stores metadata:** Yes — all fields from Document.metadata
- [x] **Persistent storage:** `chroma_db/` directory persisted to disk
- [x] **Persistence confirmed:** Data survives app restart and Streamlit reloads

**To verify:**
```bash
python scripts/ingest.py --dry-run   # Preview without writing
python scripts/ingest.py             # Actual ingestion
ls chroma_db/                        # Directory should be populated
```

---

## 4. RAG Implementation ✅

- [x] **User query is embedded:** Yes — via `OpenAIEmbeddings` before retrieval
- [x] **Top-K retrieval (K=3–5):** Default K=5, adjustable 3–8 via slider
- [x] **LLM generates answer with retrieved context:** GPT-4o-mini + formatted context
- [x] **Source citations displayed in UI:**
  - Inline `[Source N]` in generated text
  - Expandable citation cards with title, filename, chunk ID, page, snippet
- [x] **Answers grounded in retrieved documents:** System prompt restricts to context

**To verify:**
- Run app, ask "What is backpropagation?"
- Confirm [Source 1], [Source 2] etc. appear in response
- Click "Sources" expander and confirm citation cards appear

---

## 5. Conversation Memory ✅

- [x] **Multi-turn conversation supported:** Session state preserves full chat
- [x] **Tracks last 5–10 exchanges:** `MAX_MEMORY_EXCHANGES=7` in `config.py`
- [x] **Follow-up questions handled correctly:** Query contextualization step
       rephrases ambiguous follow-ups using conversation history

**To verify:**
1. Ask: "What is a neural network?"
2. Ask: "What are its limitations?"  ← ambiguous follow-up
3. Confirm the response still correctly answers about neural networks

---

## 6. Advanced Feature: Hybrid Search ✅

- [x] **Feature chosen:** Hybrid Search (Semantic + BM25)
- [x] **BM25 implemented:** `BM25Retriever` from `langchain-community` + `rank-bm25`
- [x] **Semantic search implemented:** ChromaDB cosine similarity
- [x] **Combined via EnsembleRetriever:** Weights [0.3 BM25, 0.7 semantic]
- [x] **Fusion method explained:** Reciprocal Rank Fusion in `retriever.py` docstring
- [x] **Justified in write-up:** Section 2.2 of `technical_writeup.md`

**Code location:** `app/retriever.py`

---

## 7. Deployment ✅

- [ ] **Publicly accessible URL:** ← Fill in after deploying to Streamlit Cloud
- [x] **Conversational UI:** Streamlit `chat_message` + `chat_input` components
- [x] **Citations clearly displayed:** Expandable "📚 N sources" section per response
- [x] **Deployment instructions:** README.md → "Deployment" section

**To verify:**
- Share the public URL with a classmate and confirm they can use it without setup

---

## 8. GitHub Repository ✅

- [x] **All source code committed**
- [x] **README with:**
  - [x] Architecture explanation
  - [x] Setup instructions (local + Streamlit Cloud)
  - [x] Deployment steps
  - [x] Configuration reference
  - [x] Cost estimates
- [x] **chroma_db/ committed** (pre-built vector store)
- [x] **data/raw/ committed** (source documents)
- [ ] **.env NOT committed** (verify: `git ls-files | grep .env` returns empty)

---

## 9. Technical Write-Up ✅

File: `technical_writeup.md`

- [x] **2 pages / ~950 words**
- [x] **Architecture covered:** Section 1.1, 1.2, 1.3
- [x] **Chunking strategy with justification:** Section 2.1 (comparison table)
- [x] **Advanced feature explanation:** Section 2.2
- [x] **Challenges and solutions:** Section 2.3 (3 challenges documented)
- [x] **Results:** Section 2.4

---

## 10. Demo Video ✅

Script: `demo_script.md`

- [x] **Script written:** 5-segment, 7–9 minute script
- [ ] **Video recorded:** ← Record following the script
- [ ] **Video uploaded:** ← Upload to YouTube (unlisted) or Google Drive
- [ ] **Video link added to README/submission**

---

## 11. Peer Assessment Form

- [ ] Submitted peer assessment for all team members

---

## Final Pre-Submission Checklist

- [ ] All 5+ example queries work correctly
- [ ] Follow-up questions show memory working
- [ ] Citation cards show source title, file, chunk ID
- [ ] Streamlit Cloud URL is live and accessible
- [ ] GitHub repo is public
- [ ] `.env` is NOT in the repo
- [ ] `technical_writeup.md` is under 2 pages
- [ ] Demo video is 7–9 minutes
- [ ] `pytest tests/ -v` passes all tests

---

## Quick Scoring Summary

| Rubric Item | Implementation | Status |
|---|---|---|
| Domain + 50 docs | 55 Wikipedia articles | ✅ |
| PDF/HTML/doc extraction | pypdf + BeautifulSoup | ✅ |
| Metadata (source, type, page, chunk_id) | All fields in metadata dict | ✅ |
| Smart chunking with justification | RecursiveChar + write-up Section 2.1 | ✅ |
| Vector DB (Chroma/FAISS/Pinecone) | ChromaDB | ✅ |
| Store chunks + embeddings + metadata | Yes | ✅ |
| Persistent storage | chroma_db/ on disk | ✅ |
| Embed queries | OpenAI embeddings on each query | ✅ |
| Top-K retrieval (3–5) | Default K=5, slider 3–8 | ✅ |
| LLM answers with context | GPT-4o-mini + context | ✅ |
| Source citations in UI | Inline + expandable cards | ✅ |
| Multi-turn memory | 7-turn sliding window | ✅ |
| Follow-up questions | Query contextualization | ✅ |
| Advanced feature | Hybrid Search (BM25 + Semantic) | ✅ |
| Public URL | Streamlit Community Cloud | ⬜ deploy |
| Conversational UI | Streamlit chat components | ✅ |
| GitHub repo + docs | README + inline comments | ✅ |
| Technical write-up (2 pages) | technical_writeup.md | ✅ |
| Demo video | Script in demo_script.md | ⬜ record |

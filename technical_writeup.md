# Technical Write-Up: Ask Vibs — RAG-Powered NLP Professor Chatbot

**Course:** BANA275  |  **Date:** March 2026  |  **Team:** [Names]

---

## 1. System Architecture and Design

### 1.1 Overview

We built **Ask Vibs**, a production-ready Retrieval-Augmented Generation (RAG) chatbot
that acts as an NLP professor for BANA275. The system retrieves relevant passages from a
curated knowledge base of 65 documents — including 57 Wikipedia AI/ML articles and 8
BANA275 lecture slide exports (covering LLMs, prompt engineering, RAG systems, RLHF, and
agentic AI) — totaling 3,385 chunks. It generates grounded, cited answers using
Google's **Gemini 2.5 Flash** model. The pipeline processes a user query to a cited
response in approximately 2–5 seconds.

### 1.2 Component Architecture

The architecture is a five-stage pipeline:

**Stage 1 — Document Processing:** We ingest multiple document formats (PDF, HTML, plain
text, Markdown) through a unified `DocumentProcessor`. Each document is loaded with
structured metadata: `title`, `source` filename, `doc_type`, `page` number (for PDFs),
and `url` (for web sources). Documents are chunked using `RecursiveCharacterTextSplitter`
(described in Section 2).

**Stage 2 — Vector Storage:** All chunks and their local `all-MiniLM-L6-v2`
(HuggingFace sentence-transformers) embeddings are stored in a **ChromaDB** persistent
collection on disk (`chroma_db/`). We chose ChromaDB because it requires zero
infrastructure setup, supports local persistence, integrates natively with LangChain, and
can be committed to a GitHub repository for zero-cost deployment on Streamlit Community
Cloud. Local embeddings were selected to eliminate API rate limits and cost during ingestion.

**Stage 3 — Query Contextualization:** When a user sends a follow-up question (e.g.,
"What about its training?"), the system first calls Gemini with the conversation history
to rephrase it into a standalone question (e.g., "How is a Transformer model trained?").
This prevents the retriever from receiving ambiguous pronoun-heavy queries that produce
poor results.

**Stage 4 — Hybrid Retrieval (Advanced Feature):** The contextualized query is passed to
an `EnsembleRetriever` that fuses BM25 (keyword) and semantic (ChromaDB cosine
similarity) results using Reciprocal Rank Fusion (RRF) with weights of 0.3 and 0.7
respectively. Top-K=5 chunks are returned by default.

**Stage 5 — Generation:** The retrieved chunks are formatted as a numbered context block
and passed to Gemini with an instruction to answer as "Vibs, an NLP and AI professor" and
to cite inline using `[Source N]` notation. The full response and structured source
metadata are returned to the Streamlit UI, which renders citations as expandable cards
showing the document title, filename, chunk ID, page number, and a 350-char snippet.

**Conversation Memory:** We maintain a sliding window of the last 7 human–AI exchange
pairs in Streamlit `session_state`. These are passed to Stage 3 on every query, enabling
coherent multi-turn conversations without the cost of summarization-based memory.

### 1.3 Technology Choices

| Component | Choice | Justification |
|---|---|---|
| LLM | Gemini 2.5 Flash | Free-tier availability, strong reasoning, fast responses; replaces deprecated Gemini 1.5 Flash |
| Embeddings | all-MiniLM-L6-v2 (local) | Free, no rate limits, runs on CPU, sufficient quality for domain-specific retrieval |
| Vector DB | ChromaDB | Zero infra, persistent, Git-committable, native LangChain integration |
| Orchestration | LangChain + langchain_classic | Standardized interfaces; EnsembleRetriever sourced from `langchain_classic` due to version migration |
| Frontend | Streamlit | Fast chat UI with `chat_message`, `chat_input`, session state, and download components |
| Deployment | Streamlit Community Cloud | Free, one-click GitHub integration, HTTPS public URL |

---

## 2. Chunking Strategy, Advanced Feature, and Evaluation

### 2.1 Chunking Strategy: RecursiveCharacterTextSplitter

We chose `RecursiveCharacterTextSplitter` with `chunk_size=1000` and
`chunk_overlap=200` after evaluating four strategies:

**Fixed-size chunking** was rejected because cutting at a fixed character boundary
frequently splits sentences mid-thought, producing incoherent chunks that confuse both
the embedding model and the reader.

**Sentence-based chunking** was rejected because many technical paragraphs contain long,
complex sentences (e.g., mathematical definitions) that alone exceed a practical context
window segment.

**Paragraph-based chunking** produces semantically clean units but results in wildly
variable chunk sizes — a one-line section header becomes a useless chunk while a
six-paragraph introduction becomes too large for effective retrieval.

**Recursive chunking** solves all three problems by hierarchically applying separators
`["\n\n", "\n", ". ", " "]` — it tries to split at paragraph boundaries first, falls
back to line breaks, then sentence boundaries, then words. This guarantees chunks stay
within `chunk_size` while preserving the largest natural semantic unit possible. The 20%
overlap (200 chars) ensures that a sentence straddling a chunk boundary appears in both
adjacent chunks, preserving retrieval continuity.

### 2.2 Advanced Feature: Hybrid Search

We implemented hybrid search by combining two complementary retrieval strategies in
LangChain's `EnsembleRetriever`:

- **BM25 (Okapi BM25, weight=0.3):** A classical sparse retrieval method that scores
  based on term frequency and inverse document frequency. It excels at exact technical
  term matching (e.g., "BERT", "backpropagation", "RLHF"). BM25 is fast and requires no
  GPU or API calls.

- **Semantic Search (weight=0.7):** ChromaDB cosine similarity between the query
  embedding and chunk embeddings. Captures meaning-level similarity, handling paraphrased
  questions and conceptual queries that keywords alone would miss.

Scores are merged using **Reciprocal Rank Fusion**:
`score(d) = Σ w_i / (k + rank_i(d))` where `k=60` dampens the influence of top-ranked
documents, producing a stable fused ranking. We weight semantic search more heavily (0.7)
because most student questions about AI/ML are conceptual rather than exact-term lookups.

### 2.3 UI Features

Beyond the core RAG pipeline, we implemented several student-facing features:

- **Fun Fact Banner:** On each new session, the system generates a 2–3 sentence fun fact
  on a randomly selected topic (RLHF, transformers, LLMs, etc.) from the knowledge base,
  displayed in a styled banner at the top of the page.

- **Multiple Conversation Logs:** Students can create and switch between named
  conversation threads, each with independent message history and memory context.

- **Quiz Mode:** Students can select a topic (or type their own) and generate a 10-question
  multiple choice quiz from the course materials. After submission, the system grades
  responses, assigns a letter grade (A/B/C/F), and shows correct answers. Quiz scores are
  logged in the sidebar and students can retake for a new set of questions.

- **Quick-Start Prompts:** Three topic chips (RLHF, RAG systems, Prompt Engineering)
  displayed on an empty chat to help students get started.

- **Export Conversation:** Download any conversation as a `.txt` file.

### 2.4 Challenges and Solutions

**Challenge 1 — LangChain version migration:** `EnsembleRetriever` was removed from
`langchain.retrievers` in LangChain 1.x and relocated to the `langchain_classic` package.
We resolved this by updating the import to
`from langchain_classic.retrievers.ensemble import EnsembleRetriever`.

**Challenge 2 — Missing sentence-transformers dependency:** The `all-MiniLM-L6-v2`
HuggingFace embedding model requires `sentence-transformers`, which was not in the
original requirements. We added it via pip and updated `requirements.txt`.

**Challenge 3 — Gemini model deprecation:** `gemini-1.5-flash` was removed from the
API. We migrated to `gemini-2.5-flash`, which is available on the free tier and offers
improved reasoning.

**Challenge 4 — Ingest deduplication bug:** When adding new documents to an existing
ChromaDB collection, the ingest script's chunk-sorting logic caused new file chunks to be
skipped as duplicates. We resolved this by re-ingesting with `--clear` to rebuild the
collection from scratch, ensuring all 65 documents were indexed.

**Challenge 5 — Follow-up question coherence:** Follow-up questions like "What about its
limitations?" produced poor retrievals because the retriever had no context. We solved
this by adding a query contextualization step (Stage 3) that calls Gemini to rephrase
every follow-up into a standalone question before retrieval.

### 2.5 Results

The system successfully:
- Ingests 65 documents (Wikipedia articles + BANA275 lecture slides) → 3,385 chunks
- Responds to factual and conceptual queries with citations to 3–5 relevant chunks
- Handles 7-turn follow-up conversations without losing context
- Provides source citations with chunk-level granularity for academic traceability
- Generates quizzes, fun facts, and study aids from course-specific materials
- Runs at near-zero cost using free-tier Gemini API and local embeddings

The hybrid search approach ensured that technical term queries (BM25-strong) and
conceptual questions (semantic-strong) both returned high-quality top-5 results, making
the chatbot useful across a broad range of question types encountered in an NLP course.

---

*Word count: ~1,050 words*

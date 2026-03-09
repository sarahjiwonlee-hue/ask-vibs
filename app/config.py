"""
Central configuration for the RAG chatbot.
All tunable parameters live here so you only need to change one file.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
DATA_DIR = str(BASE_DIR / "data" / "raw")

# ── API Keys (loaded from .env locally, or Streamlit secrets on Cloud) ───
def _get_api_key() -> str:
    """Read from env first, then fall back to st.secrets (Streamlit Cloud)."""
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            pass
    return key

GOOGLE_API_KEY = _get_api_key()

# ── Models ────────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# ── ChromaDB ──────────────────────────────────────────────────────────────
COLLECTION_NAME = "domain_knowledge"

# ── Chunking ──────────────────────────────────────────────────────────────
# Strategy: RecursiveCharacterTextSplitter
# - chunk_size=1000 (~250 tokens) fits safely within context windows
# - overlap=200 (20%) ensures continuity across boundaries
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Retrieval ─────────────────────────────────────────────────────────────
# Advanced Feature: Hybrid Search
# BM25 weight=0.3 (keyword), Semantic weight=0.7 (meaning)
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
BM25_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.7

# ── Memory ────────────────────────────────────────────────────────────────
MAX_MEMORY_EXCHANGES = int(os.getenv("MAX_MEMORY_EXCHANGES", "7"))  # last 7 turns

# ── Domain ────────────────────────────────────────────────────────────────
DOMAIN_NAME = "Ask Vibs — NLP & AI Professor"
DOMAIN_DESCRIPTION = (
    "Course knowledge base for BANA275 covering NLP, large language models, "
    "prompt engineering, RAG systems, RLHF, agentic AI, and foundational ML/AI topics."
)

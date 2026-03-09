"""
ChromaDB vector store wrapper with persistent storage.

All embeddings, chunks, and metadata are stored in chroma_db/ (committed
to Git so Streamlit Cloud can load them without re-ingesting on every cold start).
"""

import os
import sys
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document

# ── SQLite compatibility fix for Streamlit Cloud ──────────────────────────
# Streamlit Cloud ships an older sqlite3; pysqlite3-binary provides a newer one.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # Local dev uses the system sqlite3, which is fine

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

from config import CHROMA_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages a persistent ChromaDB collection.

    Responsibilities:
      - Add documents (called by ingest.py)
      - Provide a LangChain retriever (used by HybridRetriever)
      - Return all stored documents for BM25 indexing
      - Expose stats for the Streamlit sidebar
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        os.makedirs(CHROMA_DIR, exist_ok=True)

        # PersistentClient stores data on disk at CHROMA_DIR
        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        # LangChain Chroma wrapper (handles embed-on-add automatically)
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    # ── Write ─────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """Embed and persist a list of Document chunks."""
        self.vectorstore.add_documents(documents)
        logger.info(f"Stored {len(documents)} chunks in ChromaDB")

    def clear(self) -> None:
        """Delete and recreate the collection (used by ingest --clear)."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            # Recreate the LangChain wrapper pointing to the fresh collection
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
            )
            logger.info("Collection cleared and recreated")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

    # ── Read ──────────────────────────────────────────────────────────────

    def get_retriever(self, k: int = 5):
        """Return a LangChain retriever for semantic (cosine) search."""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Direct semantic search (used as fallback)."""
        return self.vectorstore.similarity_search(query, k=k)

    def get_all_documents(self) -> List[Document]:
        """
        Return every stored chunk as a Document list.
        Used to build the BM25 index (must happen in-memory each session).
        """
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
            results = collection.get(include=["documents", "metadatas"])

            docs = []
            for text, meta in zip(results["documents"], results["metadatas"]):
                docs.append(Document(page_content=text, metadata=meta or {}))
            return docs
        except Exception as e:
            logger.error(f"Failed to fetch all documents: {e}")
            return []

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return chunk count and unique source document count."""
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
            count = collection.count()
            results = collection.get(include=["metadatas"])

            unique_sources = {
                meta.get("source", "")
                for meta in results["metadatas"]
                if meta
            }
            return {
                "chunk_count": count,
                "document_count": len(unique_sources),
            }
        except Exception:
            return {"chunk_count": 0, "document_count": 0}

    def is_populated(self) -> bool:
        """Return True if the collection has at least one chunk."""
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
            return collection.count() > 0
        except Exception:
            return False

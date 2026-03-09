"""
Advanced Feature: Hybrid Search Retriever
──────────────────────────────────────────
Combines two complementary retrieval strategies:

  1. BM25 (weight=0.3)  — keyword/lexical matching
     Best for: exact technical terms ("BERT", "backpropagation", "LSTM")
     Weakness: misses paraphrased or semantically similar queries

  2. ChromaDB Semantic (weight=0.7) — dense vector cosine similarity
     Best for: meaning-based queries ("how do transformers handle long sequences?")
     Weakness: can miss exact rare terms

Fusion method: Reciprocal Rank Fusion (RRF), built into LangChain's
EnsembleRetriever.  Each document's score = Σ weight_i / (k + rank_i)
where k=60 dampens the effect of high ranks.

The 70/30 semantic-heavy split works well for an AI/ML knowledge base
because most user questions are conceptual rather than keyword lookups.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid BM25 + semantic retriever using Reciprocal Rank Fusion."""

    def __init__(self, vsm: VectorStoreManager):
        self.vsm = vsm

    def _build_ensemble(self, k: int) -> EnsembleRetriever:
        """
        Build a fresh EnsembleRetriever.

        We rebuild each call because:
        - k can change per query (user slider)
        - BM25 is in-memory only, so no extra cost
        """
        all_docs = self.vsm.get_all_documents()

        if not all_docs:
            return None

        # BM25: tokenises on whitespace + lowercase
        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = k

        # Semantic: ChromaDB cosine similarity
        semantic = self.vsm.get_retriever(k=k)

        return EnsembleRetriever(
            retrievers=[bm25, semantic],
            weights=[0.3, 0.7],  # BM25=30%, Semantic=70%
        )

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve top-k chunks using hybrid search.

        Falls back to pure semantic search if BM25 index can't be built
        (e.g. empty database on first startup).
        """
        try:
            ensemble = self._build_ensemble(k)

            if ensemble is None:
                logger.warning("No documents in store; returning empty list")
                return []

            docs = ensemble.invoke(query)

            # Deduplicate by chunk_id and enforce exactly k results
            seen: set = set()
            unique: List[Document] = []
            for doc in docs:
                cid = doc.metadata.get("chunk_id", doc.page_content[:60])
                if cid not in seen:
                    seen.add(cid)
                    unique.append(doc)
                if len(unique) >= k:
                    break

            return unique

        except Exception as e:
            logger.error(f"Hybrid retrieval failed ({e}); falling back to semantic")
            return self.vsm.similarity_search(query, k=k)

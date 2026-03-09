"""
Smoke tests for the RAG pipeline.

Run with:  pytest tests/ -v
These tests use mocks to avoid real API calls (and charges).
"""

import sys
import os
from unittest.mock import MagicMock, patch

# Ensure app/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


# ── DocumentProcessor ─────────────────────────────────────────────────────

def test_chunk_documents_assigns_chunk_ids():
    from document_processor import DocumentProcessor
    from langchain.schema import Document

    proc = DocumentProcessor(chunk_size=200, chunk_overlap=20)
    docs = [
        Document(
            page_content="Neural networks are computational models. " * 20,
            metadata={"source": "test.txt", "title": "Test", "doc_type": "text"},
        )
    ]
    chunks = proc.chunk_documents(docs)

    assert len(chunks) > 1, "Long document should produce multiple chunks"
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "source" in chunk.metadata
        assert len(chunk.page_content) > 100


def test_chunk_documents_drops_short_chunks():
    from document_processor import DocumentProcessor
    from langchain.schema import Document

    proc = DocumentProcessor(chunk_size=200, chunk_overlap=20)
    docs = [
        Document(
            page_content="Short.",
            metadata={"source": "test.txt", "title": "Test", "doc_type": "text"},
        )
    ]
    chunks = proc.chunk_documents(docs)
    assert len(chunks) == 0, "Sub-100-char chunk should be filtered out"


def test_metadata_defaults_populated():
    from document_processor import DocumentProcessor
    from langchain.schema import Document

    proc = DocumentProcessor(chunk_size=200, chunk_overlap=20)
    docs = [
        Document(
            page_content="Deep learning uses multiple layers. " * 20,
            metadata={},  # empty metadata
        )
    ]
    chunks = proc.chunk_documents(docs)
    for chunk in chunks:
        assert chunk.metadata["title"] == "Unknown"
        assert chunk.metadata["source"] == "Unknown"
        assert chunk.metadata["doc_type"] == "unknown"


# ── RAGPipeline (mocked) ──────────────────────────────────────────────────

@patch("rag_pipeline.VectorStoreManager")
@patch("rag_pipeline.HybridRetriever")
@patch("rag_pipeline.ChatOpenAI")
@patch("rag_pipeline.OpenAIEmbeddings")
def test_query_returns_answer_and_sources(
    mock_emb, mock_llm_cls, mock_retriever_cls, mock_vsm_cls
):
    """Test that query() returns the expected keys and that memory is used."""
    os.environ["OPENAI_API_KEY"] = "sk-test-fake-key"

    from langchain.schema import Document
    from rag_pipeline import RAGPipeline

    # Mock LLM responses
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="Transformers use self-attention [Source 1]."
    )
    mock_llm_cls.return_value = mock_llm

    # Mock retriever returning a dummy document
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(
            page_content="The Transformer architecture uses self-attention mechanisms.",
            metadata={
                "title": "Transformer",
                "source": "Transformer.txt",
                "doc_type": "wikipedia",
                "chunk_id": "abc12345_00001",
                "page": None,
                "url": "https://en.wikipedia.org/wiki/Transformer",
            },
        )
    ]
    mock_retriever_cls.return_value = mock_retriever

    # Mock vector store
    mock_vsm = MagicMock()
    mock_vsm.is_populated.return_value = True
    mock_vsm_cls.return_value = mock_vsm

    pipeline = RAGPipeline()
    result = pipeline.query(
        question="How do transformers work?",
        chat_history=[("What is deep learning?", "Deep learning uses neural networks.")],
        top_k=3,
    )

    assert "answer" in result
    assert "sources" in result
    assert "standalone_question" in result
    assert len(result["sources"]) == 1
    assert result["sources"][0]["title"] == "Transformer"


def test_format_history_truncates_to_max_exchanges():
    os.environ["OPENAI_API_KEY"] = "sk-test-fake"

    with patch("rag_pipeline.VectorStoreManager"), \
         patch("rag_pipeline.HybridRetriever"), \
         patch("rag_pipeline.ChatOpenAI"), \
         patch("rag_pipeline.OpenAIEmbeddings"):

        from rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()

        # 10 exchanges, but MAX_MEMORY_EXCHANGES = 7
        history = [(f"Q{i}", f"A{i}") for i in range(10)]
        formatted = pipeline._format_history(history)

        # Should only contain the last 7
        assert "Q3" not in formatted  # Q0–Q2 should be dropped
        assert "Q9" in formatted


# ── HybridRetriever ───────────────────────────────────────────────────────

def test_hybrid_retriever_deduplicates():
    """Duplicate chunk_ids should not appear twice in results."""
    from langchain.schema import Document
    from retriever import HybridRetriever

    mock_vsm = MagicMock()

    # Both BM25 and semantic return the same document
    dup_doc = Document(
        page_content="Neural networks learn representations.",
        metadata={"chunk_id": "same_id", "title": "NN", "source": "nn.txt", "doc_type": "text"},
    )
    mock_vsm.get_all_documents.return_value = [dup_doc] * 3
    mock_vsm.get_retriever.return_value = MagicMock(invoke=MagicMock(return_value=[dup_doc]))
    mock_vsm.similarity_search.return_value = [dup_doc]

    retriever = HybridRetriever(mock_vsm)

    with patch("retriever.BM25Retriever") as mock_bm25_cls, \
         patch("retriever.EnsembleRetriever") as mock_ensemble_cls:

        mock_ensemble = MagicMock()
        mock_ensemble.invoke.return_value = [dup_doc, dup_doc, dup_doc]
        mock_ensemble_cls.return_value = mock_ensemble

        mock_bm25 = MagicMock()
        mock_bm25_cls.from_documents.return_value = mock_bm25

        results = retriever.retrieve("neural network", k=5)

    # Despite 3 copies returned, deduplication should yield 1
    assert len(results) == 1

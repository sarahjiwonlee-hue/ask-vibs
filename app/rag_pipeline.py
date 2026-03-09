"""
RAG Pipeline — Ask Vibs
─────────────────────────────────────────────────────────────────────────────
Query flow:
  1. Contextualize  — rephrase follow-up questions into standalone queries
                      using the last MAX_MEMORY_EXCHANGES conversation turns
  2. Retrieve       — HybridRetriever returns top-K chunks (BM25 + semantic)
  3. Generate       — Gemini answers with citations based on class materials
  4. Return         — answer text + structured source metadata for the UI

Conversation memory is managed externally by Streamlit session_state and
passed in as a list of (human, ai) string pairs on each call.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from vector_store import VectorStoreManager
from retriever import HybridRetriever
from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    MAX_MEMORY_EXCHANGES,
    DEFAULT_TOP_K,
)

logger = logging.getLogger(__name__)


# ── Prompt Templates ──────────────────────────────────────────────────────

# Step 1: turn "what about its training?" into "How is BERT trained?"
CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Given the conversation history below and a follow-up question, \
rewrite the follow-up as a self-contained question that can be understood \
without the history. If no rewriting is needed, return the question as-is.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question:"""
)

# Step 2: answer with the retrieved context
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are Vibs, an NLP and AI professor. Answer the student's question \
using ONLY the provided course materials below. Be clear, helpful, and pedagogical.

Rules:
- Do NOT include any inline citations or source references like [Source N] in your answer.
- If the context is insufficient, say so honestly rather than guessing.
- Use clear, well-structured paragraphs suitable for a student audience.

Context:
{context}

Question: {question}

Answer:"""
)


class RAGPipeline:
    """
    End-to-end RAG pipeline: query → contextualize → retrieve → generate.
    """

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. "
                "Add it to .env locally or to Streamlit Cloud secrets."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.1,           # Low temperature for factual answers
            google_api_key=GOOGLE_API_KEY,
        )

        # Free local embeddings — same model used during ingestion
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        self.vsm = VectorStoreManager(self.embeddings)
        self.hybrid_retriever = HybridRetriever(self.vsm)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _format_history(self, history: List[Tuple[str, str]]) -> str:
        """Format the last MAX_MEMORY_EXCHANGES turns for prompt injection."""
        if not history:
            return "No prior conversation."
        lines = []
        for human, ai in history[-MAX_MEMORY_EXCHANGES:]:
            lines.append(f"Human: {human}")
            # Truncate long AI turns to keep prompt size manageable
            lines.append(f"AI: {ai[:600]}{'…' if len(ai) > 600 else ''}")
        return "\n".join(lines)

    def _contextualize(self, question: str, history: List[Tuple[str, str]]) -> str:
        """Return a standalone version of the question."""
        if not history:
            return question
        prompt = CONDENSE_PROMPT.format_messages(
            chat_history=self._format_history(history),
            question=question,
        )
        return self.llm.invoke(prompt).content.strip()

    def _build_context(self, docs: List[Document]) -> Tuple[str, List[Dict]]:
        """
        Build the context string passed to the answer prompt AND
        the structured sources list returned to the UI for citation rendering.
        """
        context_parts = []
        sources = []

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            context_parts.append(
                f"[Source {i}]\n"
                f"Title: {meta.get('title', 'Unknown')}\n"
                f"File: {meta.get('source', 'Unknown')}\n"
                f"Content:\n{doc.page_content}"
            )
            sources.append(
                {
                    "index": i,
                    "title": meta.get("title", "Unknown"),
                    "source": meta.get("source", "Unknown"),
                    "doc_type": meta.get("doc_type", "unknown"),
                    "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
                    "page": meta.get("page"),
                    "url": meta.get("url"),
                    "snippet": doc.page_content[:400],
                }
            )

        return "\n\n" + ("─" * 40 + "\n\n").join(context_parts), sources

    # ── Main Entry Point ──────────────────────────────────────────────────

    def query(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """
        Process a user question and return an answer with citations.

        Args:
            question:     The raw user question.
            chat_history: List of (human, ai) pairs for memory context.
            top_k:        Number of chunks to retrieve (3–8 recommended).

        Returns:
            {
              "answer": str,
              "sources": List[Dict],
              "standalone_question": str,  # for debugging
            }
        """
        chat_history = chat_history or []

        try:
            # Step 1 — Contextualize
            standalone = self._contextualize(question, chat_history)
            logger.info(f"Standalone question: {standalone!r}")

            # Step 2 — Retrieve
            docs = self.hybrid_retriever.retrieve(standalone, k=top_k)

            if not docs:
                return {
                    "answer": (
                        "I couldn't find relevant information in the knowledge base. "
                        "Try rephrasing your question or asking about a different aspect."
                    ),
                    "sources": [],
                    "standalone_question": standalone,
                }

            # Step 3 — Build context + sources
            context, sources = self._build_context(docs)

            # Step 4 — Generate answer
            prompt = ANSWER_PROMPT.format_messages(
                context=context,
                question=question,  # Use the *original* question for naturalness
            )
            answer = self.llm.invoke(prompt).content

            return {
                "answer": answer,
                "sources": sources,
                "standalone_question": standalone,
            }

        except Exception as e:
            logger.exception("RAG pipeline error")
            return {
                "answer": f"⚠️ An error occurred: {e}",
                "sources": [],
                "standalone_question": question,
            }

    # ── Utility ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return self.vsm.get_stats()

    def is_ready(self) -> bool:
        return self.vsm.is_populated()

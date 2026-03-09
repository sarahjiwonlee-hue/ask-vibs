"""
Document processor: loads PDFs, HTML, TXT, MD and Wikipedia text files,
then splits them into overlapping chunks with rich metadata.

Chunking Strategy: RecursiveCharacterTextSplitter
──────────────────────────────────────────────────
Chosen over alternatives because:
  • Fixed-size: fast but ignores sentence/paragraph boundaries → incoherent chunks
  • Sentence-based: clean but fails on long technical paragraphs
  • Paragraph-based: good but inconsistent chunk sizes degrade retrieval
  • Recursive (ours): hierarchically tries [\n\n, \n, ". ", " "] so chunks
    break at the largest natural boundary that fits chunk_size, giving the
    best semantic coherence with predictable size.

Parameters: chunk_size=1000, chunk_overlap=200 (20% overlap)
"""

import os
import re
import hashlib
import logging
from typing import List
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Loads multiple document formats and chunks them for vector storage."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        # RecursiveCharacterTextSplitter: tries each separator in order,
        # falling back to the next if the chunk is still too large.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )

    # ── Loaders ──────────────────────────────────────────────────────────

    def load_pdf(self, file_path: str) -> List[Document]:
        """Extract text from each page of a PDF with page-level metadata."""
        from pypdf import PdfReader

        docs = []
        filename = Path(file_path).name
        title = filename.replace(".pdf", "").replace("_", " ").title()

        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "doc_type": "pdf",
                                "title": title,
                                "page": page_num,
                                "total_pages": len(reader.pages),
                                "file_path": file_path,
                            },
                        )
                    )
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")

        return docs

    def load_html(self, file_path: str) -> List[Document]:
        """Strip HTML tags and extract clean text with title metadata."""
        from bs4 import BeautifulSoup

        filename = Path(file_path).name
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "lxml")

            title_tag = soup.find("title")
            title = title_tag.get_text().strip() if title_tag else filename

            # Remove boilerplate elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)

            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "doc_type": "html",
                        "title": title,
                        "file_path": file_path,
                    },
                )
            ]
        except Exception as e:
            logger.error(f"Failed to load HTML {file_path}: {e}")
            return []

    def load_text(self, file_path: str, doc_type: str = "text") -> List[Document]:
        """Load plain text or markdown files."""
        filename = Path(file_path).name
        title = (
            filename.replace(".txt", "")
            .replace(".md", "")
            .replace("_", " ")
            .title()
        )

        # For our Wikipedia downloads, extract the real title from line 1
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Parse metadata header written by download_docs.py
            if content.startswith("Title:"):
                lines = content.split("\n")
                for line in lines[:6]:
                    if line.startswith("Title:"):
                        title = line.replace("Title:", "").strip()
                        break
                # Remove the metadata header block (everything before ====)
                sep = "=" * 60
                if sep in content:
                    content = content.split(sep, 1)[1].strip()

            return [
                Document(
                    page_content=content,
                    metadata={
                        "source": filename,
                        "doc_type": doc_type,
                        "title": title,
                        "file_path": file_path,
                    },
                )
            ]
        except Exception as e:
            logger.error(f"Failed to load text {file_path}: {e}")
            return []

    def load_directory(self, directory: str) -> List[Document]:
        """Recursively load all supported documents from a directory."""
        docs = []
        directory = Path(directory)

        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return docs

        extension_map = {
            ".pdf": self.load_pdf,
            ".html": self.load_html,
            ".htm": self.load_html,
            ".txt": lambda p: self.load_text(p, "text"),
            ".md": lambda p: self.load_text(p, "markdown"),
        }

        for file_path in sorted(directory.rglob("*")):
            loader = extension_map.get(file_path.suffix.lower())
            if loader:
                loaded = loader(str(file_path))
                docs.extend(loaded)

        logger.info(f"Loaded {len(docs)} raw documents from {directory}")
        return docs

    # ── Chunking ─────────────────────────────────────────────────────────

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into overlapping chunks, assign unique chunk IDs,
        and ensure all required metadata fields are present.
        """
        chunks = self.splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            # Stable, unique ID: hash of content + position
            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
            chunk.metadata["chunk_id"] = f"{content_hash}_{i:05d}"

            # Guarantee required fields so downstream code never KeyErrors
            chunk.metadata.setdefault("title", "Unknown")
            chunk.metadata.setdefault("source", "Unknown")
            chunk.metadata.setdefault("doc_type", "unknown")
            chunk.metadata.setdefault("page", None)

            chunk.page_content = chunk.page_content.strip()

        # Drop trivially short chunks (headers, stray punctuation, etc.)
        chunks = [c for c in chunks if len(c.page_content) > 100]

        logger.info(
            f"Produced {len(chunks)} chunks from {len(documents)} documents"
        )
        return chunks

    def process_directory(self, directory: str) -> List[Document]:
        """Full pipeline: load → chunk → return ready-to-embed chunks."""
        docs = self.load_directory(directory)
        if not docs:
            return []
        return self.chunk_documents(docs)

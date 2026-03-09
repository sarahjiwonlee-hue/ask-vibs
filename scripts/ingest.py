"""
Document ingestion pipeline — run this ONCE before starting the app.

Steps:
  1. Load and chunk all documents from data/raw/ (or --dir)
  2. Embed chunks using OpenAI text-embedding-3-small
  3. Store embeddings + metadata in chroma_db/ (persistent)

After this script completes, commit chroma_db/ to Git so Streamlit Cloud
can load the pre-built index without re-embedding on every cold start.

Usage:
    python scripts/ingest.py                # ingest data/raw/
    python scripts/ingest.py --dir my/docs  # custom directory
    python scripts/ingest.py --clear        # wipe and rebuild from scratch
    python scripts/ingest.py --dry-run      # preview chunks without storing

Cost estimate: ~55 Wikipedia articles ≈ 4,000 chunks ≈ $0.02 (embedding only)
"""

import os
import sys
import argparse
from pathlib import Path

# Make sure app/ modules are importable from the scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from dotenv import load_dotenv

load_dotenv()


def banner(text: str):
    print(f"\n{'─' * 55}")
    print(f"  {text}")
    print("─" * 55)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--dir",
        default="data/raw",
        help="Directory containing documents (PDF, HTML, TXT, MD)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing collection before ingesting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process documents but do not write to ChromaDB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Chunks per embedding API call (default: 50)",
    )
    args = parser.parse_args()

    # ── Environment check ─────────────────────────────────────────────────
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY is not set.")
        print("  Create a .env file with:  GOOGLE_API_KEY=your-key")
        sys.exit(1)

    banner("RAG Document Ingestion Pipeline")
    print(f"  Source directory : {args.dir}")
    print(f"  Clear existing   : {args.clear}")
    print(f"  Dry run          : {args.dry_run}")

    # ── Imports (after sys.path is set) ───────────────────────────────────
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from document_processor import DocumentProcessor
    from vector_store import VectorStoreManager

    # ── Step 1: Process documents ─────────────────────────────────────────
    banner("Step 1 / 3 — Processing documents")

    doc_dir = Path(args.dir)
    if not doc_dir.exists():
        print(f"ERROR: Directory '{args.dir}' not found.")
        print("  Run:  python scripts/download_docs.py")
        sys.exit(1)

    processor = DocumentProcessor()
    chunks = processor.process_directory(str(doc_dir))

    if not chunks:
        print(f"ERROR: No supported documents found in '{args.dir}'.")
        print("  Supported: .pdf  .html  .htm  .txt  .md")
        sys.exit(1)

    print(f"  Chunks produced  : {len(chunks):,}")
    print(f"  Sample titles    : {', '.join({c.metadata['title'] for c in chunks[:5]})}")

    if args.dry_run:
        print("\n[Dry run] Skipping ChromaDB write.")
        return

    # ── Step 2: Connect to ChromaDB ───────────────────────────────────────
    banner("Step 2 / 3 — Connecting to ChromaDB")

    # Use free local embeddings — no API calls, no rate limits
    print("  Using local sentence-transformers (free, no rate limits)")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vsm = VectorStoreManager(embeddings)

    if args.clear:
        print("  Clearing existing collection…")
        vsm.clear()

    existing = vsm.get_stats()
    if existing["chunk_count"] > 0 and not args.clear:
        print(f"  Collection already has {existing['chunk_count']:,} chunks — resuming.")

    # ── Step 3: Embed and store in batches ────────────────────────────────
    banner("Step 3 / 3 — Embedding and storing chunks")

    import time

    # Free tier: 100 requests/minute (1 request per chunk)
    # Batch of 5 chunks + 3.5s sleep = ~85 req/min (safe margin)
    SAFE_BATCH = 5
    SLEEP_BETWEEN = 3.5

    # Resume support: skip chunks whose chunk_id is already in the DB
    already_stored = vsm.get_stats()["chunk_count"]
    skip = already_stored
    if skip > 0:
        print(f"  Resuming — skipping first {skip} already-stored chunks")
    chunks = chunks[skip:]

    total = len(chunks)
    if total == 0:
        print("  All chunks already stored — nothing to do.")
        return
    batches = (total + SAFE_BATCH - 1) // SAFE_BATCH
    print(f"  Batches: {batches} × {SAFE_BATCH} chunks  (~{batches * SLEEP_BETWEEN / 60:.0f} min total)")

    for batch_i in range(batches):
        start = batch_i * SAFE_BATCH
        batch = chunks[start : start + SAFE_BATCH]

        # Retry up to 5 times on rate-limit errors with longer waits
        for attempt in range(5):
            try:
                vsm.add_documents(batch)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 70 * (attempt + 1)
                    print(f"  Rate limit — waiting {wait}s… (attempt {attempt+1}/5)")
                    time.sleep(wait)
                else:
                    raise
        else:
            print(f"  WARNING: batch {batch_i+1} failed after 5 attempts, skipping")

        pct = (batch_i + 1) / batches * 100
        print(f"  Batch {batch_i + 1:4d}/{batches}  ({pct:5.1f}%)  +{len(batch)} chunks")

        if batch_i < batches - 1:
            time.sleep(SLEEP_BETWEEN)

    # ── Summary ───────────────────────────────────────────────────────────
    final = vsm.get_stats()
    banner("Ingestion Complete")
    print(f"  Unique documents : {final['document_count']}")
    print(f"  Total chunks     : {final['chunk_count']:,}")
    print(f"  Storage path     : chroma_db/")
    print()
    print("Next steps:")
    print("  1. git add chroma_db/ && git commit -m 'Add pre-built vector store'")
    print("  2. streamlit run app/main.py")


if __name__ == "__main__":
    main()

"""
Download 55 Wikipedia articles about Artificial Intelligence & Machine Learning.
Saves each article as a .txt file in data/raw/ with a metadata header.

Usage:
    python scripts/download_docs.py
    python scripts/download_docs.py --output data/raw --limit 20
"""

import os
import sys
import time
import argparse
from pathlib import Path

# ── 55 AI/ML topics ───────────────────────────────────────────────────────
# Covers: core AI, ML algorithms, deep learning architectures, NLP,
# computer vision, reinforcement learning, and AI ethics/society.
TOPICS = [
    # Core AI & ML
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Supervised learning",
    "Unsupervised learning",
    "Reinforcement learning",
    # Deep Learning Architectures
    "Convolutional neural network",
    "Recurrent neural network",
    "Long short-term memory",
    "Transformer (machine learning model)",
    "Attention mechanism",
    "Generative adversarial network",
    "Variational autoencoder",
    "Autoencoder",
    "Residual neural network",
    # NLP
    "Natural language processing",
    "Large language model",
    "BERT (language model)",
    "GPT (language model)",
    "Word2vec",
    "Word embedding",
    "Text summarization",
    "Sentiment analysis",
    "Named-entity recognition",
    "Machine translation",
    "Question answering",
    "Text classification",
    # Computer Vision
    "Computer vision",
    "Object detection",
    "Image segmentation",
    "Facial recognition system",
    "Image classification",
    # Classic ML Algorithms
    "Support vector machine",
    "Random forest",
    "Decision tree",
    "Logistic regression",
    "K-nearest neighbors algorithm",
    "Naive Bayes classifier",
    "K-means clustering",
    "Principal component analysis",
    "Gradient boosting",
    # Training & Optimization
    "Backpropagation",
    "Gradient descent",
    "Stochastic gradient descent",
    "Batch normalization",
    "Dropout (neural networks)",
    "Activation function",
    "Loss function",
    "Overfitting",
    "Transfer learning",
    "Data augmentation",
    # AI & Society
    "Ethics of artificial intelligence",
    "Bias in artificial intelligence",
    "AI safety",
    "Artificial general intelligence",
    "Federated learning",
]


def download(output_dir: str = "data/raw", limit: int = None):
    try:
        import wikipediaapi
    except ImportError:
        print("Installing wikipedia-api…")
        os.system(f"{sys.executable} -m pip install wikipedia-api -q")
        import wikipediaapi

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wiki = wikipediaapi.Wikipedia(
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent="RAG-Chatbot-ClassAssignment/1.0",
    )

    topics = TOPICS[:limit] if limit else TOPICS
    print(f"Downloading {len(topics)} Wikipedia articles → {output_path.resolve()}")
    print("=" * 60)

    success, skipped, failed = 0, 0, []

    for idx, topic in enumerate(topics, 1):
        safe_name = (
            topic.replace("/", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        dest = output_path / f"{safe_name}.txt"

        prefix = f"[{idx:2d}/{len(topics)}]"

        if dest.exists():
            print(f"{prefix} SKIP  {topic}")
            skipped += 1
            continue

        try:
            page = wiki.page(topic)
            if not page.exists():
                print(f"{prefix} 404   {topic}")
                failed.append(topic)
                continue

            # Write metadata header + article body
            content = (
                f"Title: {page.title}\n"
                f"URL: {page.fullurl}\n"
                f"Source: Wikipedia\n"
                f"Document Type: Encyclopedia Article\n"
                f"Domain: Artificial Intelligence & Machine Learning\n"
                + "=" * 60 + "\n\n"
                + page.text[:60_000]   # cap at 60K chars per article
            )

            dest.write_text(content, encoding="utf-8")
            print(f"{prefix} OK    {topic}  ({len(page.text):,} chars)")
            success += 1

        except Exception as exc:
            print(f"{prefix} ERROR {topic}: {exc}")
            failed.append(topic)

        time.sleep(0.4)  # polite rate limiting

    print("\n" + "=" * 60)
    print(f"Done.  Downloaded: {success}  Skipped: {skipped}  Failed: {len(failed)}")
    if failed:
        print(f"Failed topics: {failed}")
    print(f"\nNext step:\n  python scripts/ingest.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Wikipedia articles for RAG")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max articles to download")
    args = parser.parse_args()
    download(args.output, args.limit)

"""
Module: embedder.py
Phase: 4 - Embedding Generation
Responsibility: Convert text (chunks or answers) into dense vector embeddings.
                Uses sentence-transformers. NO TensorFlow dependency.

MODEL CHOICE: all-MiniLM-L6-v2
  - Fast (runs on CPU)
  - 384-dimensional embeddings
  - Strong semantic understanding
  - Great for Q&A and similarity tasks
"""

import os
import numpy as np
from typing import List, Union
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Singleton pattern: load model once, reuse across calls
_model = None


def get_model():
    """
    Lazy-load the embedding model (only once per session).
    Returns the SentenceTransformer model instance.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

        print(f"⏳ Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Model loaded.")
    return _model


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        texts: List of strings to embed.
        batch_size: How many texts to encode at once (tune for memory).

    Returns:
        numpy array of shape (len(texts), embedding_dim)
        e.g., (100, 384) for all-MiniLM-L6-v2

    Raises:
        ValueError: If texts list is empty.
    """
    if not texts:
        raise ValueError("Cannot embed empty list of texts.")

    model = get_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,  # Show progress for large batches
        normalize_embeddings=True,           # L2-normalize for cosine similarity
        convert_to_numpy=True,
    )

    return embeddings  # shape: (N, 384)


def embed_single(text: str) -> np.ndarray:
    """
    Embed a single string. Convenience wrapper around embed_texts.

    Returns:
        1D numpy array of shape (embedding_dim,) e.g. (384,)
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    result = embed_texts([text])
    return result[0]  # Return 1D array


def get_embedding_dim() -> int:
    """Return the dimensionality of the embedding model's output."""
    return get_model().get_sentence_embedding_dimension()


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    test_sentences = [
        "TCP is a connection-oriented, reliable protocol.",
        "UDP is a connectionless, fast protocol.",
        "The sky is blue.",
    ]

    print(f"Embedding {len(test_sentences)} sentences...")
    embeddings = embed_texts(test_sentences)

    print(f"\n✅ Embedding shape: {embeddings.shape}")
    print(f"   Embedding dim: {embeddings.shape[1]}")

    # Verify normalization (L2 norm should be ~1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"   L2 norms (should be ~1.0): {norms}")

    # Quick similarity check
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(embeddings)
    print(f"\n   TCP vs UDP similarity: {sim[0][1]:.3f}")
    print(f"   TCP vs Sky similarity: {sim[0][2]:.3f}")
    print("   (TCP vs UDP should be higher — both are networking topics)")

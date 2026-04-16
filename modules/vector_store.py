"""
Module: vector_store.py
Phase: 5 - Vector Database (FAISS)
Responsibility: Store chunk embeddings in FAISS and perform fast
                nearest-neighbor search. Save/load index to disk.

WHY FAISS:
  - Facebook AI Similarity Search
  - In-memory, no server needed
  - Extremely fast similarity search
  - Supports saving to disk
"""

import os
import pickle
import numpy as np
from typing import List, Tuple
from pathlib import Path

# Default path for persisting the index
DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"


class VectorStore:
    """
    Wraps a FAISS index with the original text chunks.
    Keeps embeddings and text in sync.
    """

    def __init__(self):
        self.index = None        # FAISS index object
        self.chunks: List[str] = []  # Original text chunks (parallel to index)
        self._dim: int = 0       # Embedding dimensionality

    def build(self, chunks: List[str], embeddings: np.ndarray) -> None:
        """
        Build a new FAISS index from chunks + their embeddings.

        Args:
            chunks: List of text chunks (strings).
            embeddings: 2D numpy array, shape (N, dim).

        Raises:
            ValueError: If chunks and embeddings are mismatched.
            ImportError: If faiss-cpu not installed.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu")

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        self.chunks = chunks
        self._dim = embeddings.shape[1]

        # Ensure float32 (FAISS requirement)
        embeddings_f32 = embeddings.astype(np.float32)

        # IndexFlatIP = Inner Product search (equivalent to cosine sim for L2-normalized vecs)
        self.index = faiss.IndexFlatIP(self._dim)
        self.index.add(embeddings_f32)

        print(f"✅ FAISS index built: {self.index.ntotal} vectors, dim={self._dim}")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find the top_k most similar chunks to a query embedding.

        Args:
            query_embedding: 1D numpy array (dim,) or 2D (1, dim).
            top_k: Number of results to return.

        Returns:
            List of (chunk_text, similarity_score) tuples, sorted by score desc.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty. Call build() first.")

        import faiss

        # FAISS needs float32 and 2D input
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_f32 = query_embedding.astype(np.float32)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_f32, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for empty slots
                results.append((self.chunks[idx], float(score)))

        return results  # Already sorted by score descending

    def save(
        self,
        index_path: str = str(INDEX_PATH),
        chunks_path: str = str(CHUNKS_PATH),
    ) -> None:
        """Persist the FAISS index and chunks to disk."""
        import faiss

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)

        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"💾 Saved index → {index_path}")
        print(f"💾 Saved chunks → {chunks_path}")

    def load(
        self,
        index_path: str = str(INDEX_PATH),
        chunks_path: str = str(CHUNKS_PATH),
    ) -> None:
        """Load a previously saved FAISS index and chunks from disk."""
        import faiss

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No saved index at: {index_path}")

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self._dim = self.index.d
        print(f"📂 Loaded index: {self.index.ntotal} vectors from {index_path}")

    @property
    def is_ready(self) -> bool:
        """True if the store has been built or loaded."""
        return self.index is not None and self.index.ntotal > 0

    @property
    def size(self) -> int:
        """Number of vectors stored."""
        return self.index.ntotal if self.index else 0


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from embedder import embed_texts

    sample_chunks = [
        "TCP is connection-oriented and ensures reliable delivery.",
        "UDP is connectionless and prioritizes speed over reliability.",
        "HTTP is an application-layer protocol used for web browsing.",
        "DNS translates domain names to IP addresses.",
        "TCP uses three-way handshake to establish connections.",
    ]

    print("Embedding chunks...")
    embeddings = embed_texts(sample_chunks)

    store = VectorStore()
    store.build(sample_chunks, embeddings)

    # Test search
    from embedder import embed_single
    query = embed_single("What is the difference between TCP and UDP?")

    results = store.search(query, top_k=3)
    print("\n🔍 Search results:")
    for chunk, score in results:
        print(f"  [{score:.3f}] {chunk}")

    # Test save/load
    store.save()
    store2 = VectorStore()
    store2.load()
    print(f"\n✅ Reload test: {store2.size} vectors loaded")

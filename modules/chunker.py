"""
Module: chunker.py
Phase: 3 - Text Chunking
Responsibility: Split extracted text into smaller, overlapping chunks
                for better semantic retrieval. Nothing else.

WHY CHUNKING MATTERS:
  - Embedding models have token limits (~512 tokens for MiniLM)
  - Smaller chunks = more precise retrieval
  - Overlap prevents losing context at chunk boundaries
"""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Read from .env or use defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks using LangChain's splitter.

    Args:
        text: The full extracted text from a document.
        chunk_size: Max characters per chunk (default from .env).
        chunk_overlap: Characters shared between adjacent chunks.

    Returns:
        List of text chunk strings.

    Raises:
        ValueError: If text is empty.
        ImportError: If langchain-text-splitters not installed.
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text.")

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError("Run: pip install langchain-text-splitters")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split at these separators in order (prefer paragraph > sentence > word)
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)

    # Filter out chunks that are too short to be meaningful
    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]

    return chunks


def chunk_documents(documents: List[str]) -> List[str]:
    """
    Chunk multiple documents and return a combined flat list.

    Args:
        documents: List of extracted text strings (one per file).

    Returns:
        Combined list of all chunks from all documents.
    """
    all_chunks = []
    for doc_text in documents:
        try:
            chunks = chunk_text(doc_text)
            all_chunks.extend(chunks)
        except ValueError:
            continue  # Skip empty docs silently
    return all_chunks


def get_chunk_stats(chunks: List[str]) -> dict:
    """Return summary statistics about the chunks (for UI display)."""
    if not chunks:
        return {"count": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

    lengths = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "avg_length": int(sum(lengths) / len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    TCP stands for Transmission Control Protocol. It is a connection-oriented protocol.
    TCP ensures reliable, ordered, and error-checked delivery of data.

    UDP stands for User Datagram Protocol. It is a connectionless protocol.
    UDP is faster than TCP because it does not establish a connection before sending data.
    UDP is used in applications like video streaming, gaming, and DNS.

    The key difference between TCP and UDP is reliability vs speed.
    TCP guarantees delivery; UDP prioritizes speed over reliability.
    """

    chunks = chunk_text(sample, chunk_size=200, chunk_overlap=30)
    stats = get_chunk_stats(chunks)

    print(f"✅ Created {stats['count']} chunks")
    print(f"   Avg length: {stats['avg_length']} chars")
    print("\n--- Chunks ---")
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i+1}] ({len(chunk)} chars)\n{chunk}")

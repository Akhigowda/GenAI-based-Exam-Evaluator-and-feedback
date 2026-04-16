"""
Module: retriever.py
Phase: 6 - RAG Retrieval (IMPROVED)
Key improvement: Concept abstraction — converts raw syllabus sentences
into canonical idea units that match paraphrased student answers.
"""

import os
import re
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 8))  # Increased from 5


def retrieve_relevant_chunks(
    question: str,
    vector_store,
    embedder,
    top_k: int = TOP_K,
) -> List[Tuple[str, float]]:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if not vector_store.is_ready:
        raise RuntimeError("Vector store is not built. Upload documents first.")

    question_embedding = embedder(question)
    results = vector_store.search(question_embedding, top_k=top_k)
    return results


def format_context(retrieved_chunks: List[Tuple[str, float]]) -> str:
    if not retrieved_chunks:
        return "No relevant context found in the syllabus."
    parts = []
    for i, (chunk, score) in enumerate(retrieved_chunks, 1):
        parts.append(f"[Reference {i} | Relevance: {score:.2f}]\n{chunk}")
    return "\n\n---\n\n".join(parts)


def extract_concept_sentences(retrieved_chunks: List[Tuple[str, float]]) -> List[str]:
    """
    Improved pipeline:
      1. Extract raw sentences from chunks
      2. Filter garbage (tables, headers, page markers)
      3. Abstract each sentence into a canonical concept phrase
      4. Deduplicate and rank by quality
      5. Return top 20 concepts
    """
    MAX_CONCEPTS = 20

    raw_sentences = _extract_raw_sentences(retrieved_chunks)
    filtered      = _filter_garbage(raw_sentences)
    abstracted    = _abstract_concepts(filtered)
    unique        = _deduplicate(abstracted)

    return unique[:MAX_CONCEPTS]


# ─────────────────────────────────────────────
# Step 1: Raw sentence extraction
# ─────────────────────────────────────────────
def _extract_raw_sentences(chunks: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Split each chunk into sentences, carrying the chunk's relevance score."""
    results = []
    for chunk, relevance in chunks:
        sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())
        for s in sentences:
            s = s.strip()
            if len(s) > 25:
                results.append((s, relevance))
    return results


# ─────────────────────────────────────────────
# Step 2: Filter garbage sentences
# ─────────────────────────────────────────────
def _filter_garbage(sentences: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Remove table rows, headers, page markers, and non-sentences."""
    clean = []
    for s, rel in sentences:
        # Page markers
        if re.match(r'^\[Page \d+\]', s):
            continue
        # OSI/table row starting with digit
        if re.match(r'^\d\s+[A-Z]', s):
            continue
        # Too many commas = table cell dump
        if s.count(',') > 4 and len(s) < 160:
            continue
        # Mostly uppercase = header
        words = s.split()
        upper = sum(1 for w in words if w.isupper() and len(w) > 1)
        if len(words) >= 3 and upper / len(words) > 0.5:
            continue
        # Column header patterns
        if re.match(r'^(Layer|Name|Feature|Protocol|Example|Function|Task|Tool)\b', s) and len(s) < 80:
            continue
        # Must contain a verb to be a real sentence
        has_verb = bool(re.search(
            r'\b(is|are|was|were|has|have|uses|used|provides|ensures|allows|defines|'
            r'contains|supports|requires|enables|prevents|guarantees|translates|'
            r'connects|transmits|delivers|establishes|sends|receives|handles|manages|'
            r'converts|performs|operates|represents|refers|means|includes|consists)\b',
            s, re.IGNORECASE
        ))
        if not has_verb:
            continue
        clean.append((s, rel))
    return clean


# ─────────────────────────────────────────────
# Step 3: Abstract into canonical concept phrases
# ─────────────────────────────────────────────
def _abstract_concepts(sentences: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Convert verbose syllabus sentences into shorter, meaning-dense concept phrases.
    This is the key fix for paraphrase mismatch.

    Examples of what this does:
      "TCP is a connection-oriented protocol that establishes a connection using
       the three-way handshake before data transfer begins."
      → "TCP is connection-oriented and uses a three-way handshake"

      "UDP does not guarantee that packets will arrive at the destination in order
       or at all, making it an unreliable protocol."
      → "UDP is unreliable and does not guarantee packet delivery"
    """
    abstracted = []
    for s, rel in sentences:
        concepts = _decompose_sentence(s)
        for c in concepts:
            abstracted.append((c, rel))
    return abstracted


def _decompose_sentence(sentence: str) -> List[str]:
    """
    Break one complex sentence into 1-3 atomic concept phrases.
    Each phrase captures ONE idea in the simplest possible form.
    """
    s = sentence.strip()
    concepts = []

    # Pattern 1: "X is/are Y [and Z]" — split at 'and' if both halves are meaningful
    m = re.match(
        r'^([A-Z][^,\.]{2,40})\s+(is|are|was)\s+([^,\.]{5,80})'
        r'(?:\s+and\s+([^,\.]{5,80}))?',
        s, re.IGNORECASE
    )
    if m:
        subj   = m.group(1).strip()
        verb   = m.group(2).strip()
        pred1  = m.group(3).strip()
        pred2  = m.group(4)
        concepts.append(f"{subj} {verb} {pred1}")
        if pred2 and len(pred2) > 10:
            concepts.append(f"{subj} {verb} {pred2}")

    # Pattern 2: Sentences with "which means", "meaning", "i.e." — keep only the main clause
    for sep in [r'\bwhich means\b', r'\bmeaning\b', r'\bi\.e\.\b', r'\bthat is\b']:
        parts = re.split(sep, s, flags=re.IGNORECASE)
        if len(parts) == 2:
            main = parts[0].strip().rstrip(',')
            if len(main) > 25:
                concepts.append(main)
            break

    # Pattern 3: Long sentences with semicolons — split into separate concepts
    if ';' in s:
        parts = [p.strip() for p in s.split(';') if len(p.strip()) > 20]
        concepts.extend(parts)

    # Pattern 4: "X uses/provides/ensures Y" — keep as-is (already atomic)
    m2 = re.match(
        r'^([A-Z][^,\.]{2,30})\s+(uses|provides|ensures|supports|requires|'
        r'allows|enables|prevents|handles|performs|manages)\s+(.{10,80})',
        s, re.IGNORECASE
    )
    if m2:
        concepts.append(f"{m2.group(1)} {m2.group(2)} {m2.group(3)}")

    # Fallback: if no pattern matched or sentence is already short, use as-is
    if not concepts or (len(s) < 100 and not concepts):
        # Trim trailing clauses after comma + because/since/when/as
        trimmed = re.split(r',\s*(because|since|when|as|where|which|that)\b', s)[0]
        trimmed = trimmed.strip().rstrip(',.')
        if len(trimmed) > 25:
            concepts.append(trimmed)
        else:
            concepts.append(s[:120])  # Hard truncate very long sentences

    # Clean up each concept
    cleaned = []
    for c in concepts:
        c = c.strip().rstrip('.,;:')
        # Remove parenthetical notes
        c = re.sub(r'\s*\([^)]{0,40}\)', '', c).strip()
        # Remove "e.g." tails
        c = re.sub(r'\s*,?\s*e\.g\..*$', '', c, flags=re.IGNORECASE).strip()
        # Remove "such as X, Y, Z" tails (keeps the main claim)
        c = re.sub(r'\s*,?\s*such as\s+.{0,60}$', '', c, flags=re.IGNORECASE).strip()
        if len(c) > 20:
            cleaned.append(c)

    return cleaned if cleaned else [s[:100]]


# ─────────────────────────────────────────────
# Step 4: Deduplicate
# ─────────────────────────────────────────────
def _deduplicate(concepts: List[Tuple[str, float]]) -> List[str]:
    """Remove near-duplicate concepts using prefix matching."""
    seen = set()
    unique = []
    # Sort by relevance descending so best chunks come first
    for s, _ in sorted(concepts, key=lambda x: x[1], reverse=True):
        key = re.sub(r'\W+', '', s[:45].lower())
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from modules.vector_store import VectorStore
    from modules.embedder import embed_texts, embed_single

    sample_chunks = [
        (
            "TCP is a connection-oriented protocol that establishes a connection "
            "using the three-way handshake before data transfer begins. "
            "TCP ensures reliable, ordered delivery of data using sequence numbers "
            "and acknowledgements. If a packet is lost, TCP retransmits it automatically.",
            0.95
        ),
        (
            "UDP is a connectionless protocol which means it does not establish "
            "a connection before sending data. UDP is unreliable and does not "
            "guarantee delivery, ordering, or duplicate protection. "
            "UDP is faster than TCP because it skips connection establishment.",
            0.91
        ),
        (
            "7 Application User interface, network services HTTP, FTP, SMTP, DNS",
            0.3
        ),
    ]

    print("--- Extracted concepts ---")
    concepts = extract_concept_sentences(sample_chunks)
    for c in concepts:
        print(f"  • {c}")
    print(f"\nTotal: {len(concepts)} concepts")
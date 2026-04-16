"""
Module: evaluator.py
Phase: 7 - Evaluation Logic (IMPROVED)
"""

import os
import re
import numpy as np
from typing import List, Tuple, Dict
from dotenv import load_dotenv

load_dotenv()

SIMILARITY_THRESHOLD         = float(os.getenv("SIMILARITY_THRESHOLD", 0.30))
PARTIAL_CREDIT_THRESHOLD     = float(os.getenv("PARTIAL_CREDIT_THRESHOLD", 0.20))


def evaluate_answer(
    student_answer: str,
    expected_concepts: List[str],
    embedder_fn,
    threshold: float = SIMILARITY_THRESHOLD,
    partial_threshold: float = PARTIAL_CREDIT_THRESHOLD,
) -> Dict:
    if not student_answer or not student_answer.strip():
        raise ValueError("Student answer cannot be empty.")
    if not expected_concepts:
        raise ValueError("No expected concepts to evaluate against.")

    # Split student answer into sentences for clause-level matching
    student_sentences = _split_sentences(student_answer)

    # Embed everything in one batch
    all_texts        = [student_answer] + student_sentences + expected_concepts
    all_embeddings   = embedder_fn(all_texts)

    answer_emb       = all_embeddings[0]
    sent_embs        = all_embeddings[1: 1 + len(student_sentences)]
    concept_embs     = all_embeddings[1 + len(student_sentences):]

    partial_thresh   = max(threshold - 0.10, partial_threshold)

    matched, partial_list, missing = [], [], []
    concept_scores = []

    for concept, c_emb in zip(expected_concepts, concept_embs):
        # Score against full answer
        full_sim = float(c_emb @ answer_emb)

        # Score against each individual sentence
        best_sent_sim = max(
            (float(c_emb @ s_emb) for s_emb in sent_embs),
            default=0.0
        )

        best = max(full_sim, best_sent_sim)
        concept_scores.append((concept, round(best, 3)))

        if best >= threshold:
            matched.append(concept)
        elif best >= partial_thresh:
            partial_list.append(concept)
        else:
            missing.append(concept)

    total  = len(expected_concepts)
    earned = len(matched) * 1.0 + len(partial_list) * 0.5
    score  = round(min((earned / total * 10) if total > 0 else 0.0, 10.0), 1)
    coverage = round(earned / total * 100, 1) if total > 0 else 0.0

    concept_scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "score":                  score,
        "score_display":          f"{score}/10",
        "matched_concepts":       matched,
        "partial_concepts":       partial_list,
        "missing_concepts":       missing,
        "concept_scores":         concept_scores,
        "coverage_percent":       coverage,
        "raw_similarity":         round(float(np.mean([s for _, s in concept_scores])), 3),
        "total_concepts":         total,
        "covered_concepts":       len(matched),
        "partial_concepts_count": len(partial_list),
    }


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def get_score_grade(score: float) -> Tuple[str, str]:
    if score >= 9:   return "A+", "🏆"
    elif score >= 8: return "A",  "🌟"
    elif score >= 7: return "B",  "✅"
    elif score >= 6: return "C",  "📘"
    elif score >= 5: return "D",  "⚠️"
    else:            return "F",  "❌"


def get_performance_label(coverage: float) -> str:
    if coverage >= 90:   return "Excellent — near-complete concept coverage!"
    elif coverage >= 75: return "Good — most key concepts are present."
    elif coverage >= 60: return "Satisfactory — some important concepts are missing."
    elif coverage >= 40: return "Below average — several key concepts are missing."
    else:                return "Needs improvement — major concepts are not addressed."
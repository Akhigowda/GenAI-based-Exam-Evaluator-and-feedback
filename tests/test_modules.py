"""
tests/test_modules.py
Unit tests for each module — run with: python -m pytest tests/ -v

Tests are designed to work WITHOUT requiring the full model to be loaded.
Each module is tested in isolation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


# ─────────────────────────────────────────────
# PHASE 3: Chunker tests
# ─────────────────────────────────────────────
class TestChunker:
    def test_basic_chunking(self):
        from modules.chunker import chunk_text
        text = "A" * 1000  # 1000-char text
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1, "Should create multiple chunks"

    def test_chunk_size_respected(self):
        from modules.chunker import chunk_text
        text = " ".join(["word"] * 500)
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert len(chunk) <= 120, f"Chunk too large: {len(chunk)}"

    def test_empty_text_raises(self):
        from modules.chunker import chunk_text
        with pytest.raises(ValueError):
            chunk_text("")

    def test_chunk_stats(self):
        from modules.chunker import chunk_text, get_chunk_stats
        text = "Python is a programming language. " * 50
        chunks = chunk_text(text)
        stats = get_chunk_stats(chunks)
        assert "count" in stats
        assert stats["count"] > 0
        assert stats["avg_length"] > 0

    def test_multiple_documents(self):
        from modules.chunker import chunk_documents
        docs = ["Document one. " * 20, "Document two. " * 20]
        chunks = chunk_documents(docs)
        assert len(chunks) > 0

    def test_empty_doc_skipped(self):
        from modules.chunker import chunk_documents
        docs = ["Valid content here. " * 20, ""]
        chunks = chunk_documents(docs)
        assert len(chunks) > 0  # Should not crash


# ─────────────────────────────────────────────
# PHASE 7: Evaluator tests (uses mock embedder)
# ─────────────────────────────────────────────
class TestEvaluator:

    def _make_mock_embedder(self, answer_vec, concept_vecs):
        """Create a mock embedder that returns controlled embeddings."""
        def mock_embed(texts):
            result = np.array([answer_vec] + concept_vecs, dtype=np.float32)
            # L2 normalize
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            return result / np.maximum(norms, 1e-9)
        return mock_embed

    def test_perfect_score(self):
        """Answer that perfectly matches all concepts → score 10."""
        from modules.evaluator import evaluate_answer

        # All concepts are identical to the answer embedding
        answer_vec = [1.0, 0.0, 0.0, 0.0]
        concept_vecs = [[1.0, 0.0, 0.0, 0.0]] * 5  # Perfect match

        mock_embedder = self._make_mock_embedder(answer_vec, concept_vecs)
        concepts = [f"Concept {i}" for i in range(5)]

        result = evaluate_answer("test answer", concepts, mock_embedder, threshold=0.9)
        assert result["score"] == 10.0
        assert result["covered_concepts"] == 5
        assert len(result["missing_concepts"]) == 0

    def test_zero_score(self):
        """Answer that matches nothing → score 0."""
        from modules.evaluator import evaluate_answer

        answer_vec = [1.0, 0.0, 0.0, 0.0]
        concept_vecs = [[0.0, 1.0, 0.0, 0.0]] * 4  # Orthogonal = 0 similarity

        mock_embedder = self._make_mock_embedder(answer_vec, concept_vecs)
        concepts = [f"Concept {i}" for i in range(4)]

        result = evaluate_answer("test answer", concepts, mock_embedder, threshold=0.5)
        assert result["score"] == 0.0
        assert result["covered_concepts"] == 0

    def test_partial_score(self):
        """Answer that matches half the concepts → score 5."""
        from modules.evaluator import evaluate_answer

        answer_vec = [1.0, 0.0]
        match_vecs = [[1.0, 0.0]] * 5      # Identical to answer
        miss_vecs = [[0.0, 1.0]] * 5       # Orthogonal to answer
        concept_vecs = match_vecs + miss_vecs

        mock_embedder = self._make_mock_embedder(answer_vec, concept_vecs)
        concepts = [f"Concept {i}" for i in range(10)]

        result = evaluate_answer("test answer", concepts, mock_embedder, threshold=0.5)
        assert result["score"] == 5.0

    def test_empty_answer_raises(self):
        from modules.evaluator import evaluate_answer
        with pytest.raises(ValueError):
            evaluate_answer("", ["concept"], lambda x: np.zeros((2, 4)))

    def test_empty_concepts_raises(self):
        from modules.evaluator import evaluate_answer
        with pytest.raises(ValueError):
            evaluate_answer("answer", [], lambda x: np.zeros((1, 4)))

    def test_result_structure(self):
        from modules.evaluator import evaluate_answer

        answer_vec = [1.0, 0.0]
        concept_vecs = [[1.0, 0.0]] * 3
        mock_embedder = self._make_mock_embedder(answer_vec, concept_vecs)

        result = evaluate_answer("test", ["c1", "c2", "c3"], mock_embedder)

        required_keys = [
            "score", "score_display", "matched_concepts",
            "missing_concepts", "concept_scores", "coverage_percent",
            "raw_similarity", "total_concepts", "covered_concepts",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_score_grade(self):
        from modules.evaluator import get_score_grade
        assert get_score_grade(9.5) == ("A+", "🏆")
        assert get_score_grade(8.0) == ("A",  "🌟")
        assert get_score_grade(7.0) == ("B",  "✅")
        assert get_score_grade(6.0) == ("C",  "📘")
        assert get_score_grade(5.0) == ("D",  "⚠️")
        assert get_score_grade(3.0) == ("F",  "❌")


# ─────────────────────────────────────────────
# Feedback Generator tests
# ─────────────────────────────────────────────
class TestFeedbackGenerator:

    def _mock_result(self, score=6.0, matched=None, missing=None):
        return {
            "score": score,
            "score_display": f"{score}/10",
            "matched_concepts": matched or ["TCP is reliable."],
            "missing_concepts": missing or ["UDP is faster."],
            "coverage_percent": 60.0,
        }

    def test_template_feedback_returns_string(self):
        from modules.feedback_generator import generate_feedback, feedback_to_markdown

        result = generate_feedback(
            question="Explain TCP vs UDP",
            student_answer="TCP is reliable",
            evaluation_result=self._mock_result(),
            retrieved_context="TCP UDP context",
            retrieved_chunks=[("TCP is reliable and connection-oriented.", 0.9)],
            use_llm=False,   # Force template
        )
        assert isinstance(result, dict)
        assert result["question"] == "Explain TCP vs UDP"
        assert result["score"] == 6.0
        assert result["strengths"]
        assert result["weaknesses"]
        assert result["suggestions"]
        assert isinstance(result["ideal_answer"], str)

        rendered = feedback_to_markdown(result)
        assert "Strengths" in rendered
        assert "Weaknesses" in rendered

    def test_template_feedback_contains_sections(self):
        from modules.feedback_generator import generate_feedback

        result = generate_feedback(
            question="Test question",
            student_answer="Test answer",
            evaluation_result=self._mock_result(),
            retrieved_context="context",
            retrieved_chunks=[("TCP is reliable.", 0.9)],
            use_llm=False,
        )
        assert "strengths" in result
        assert "weaknesses" in result
        assert "suggestions" in result

    def test_perfect_score_feedback(self):
        from modules.feedback_generator import generate_feedback

        result = generate_feedback(
            question="Q",
            student_answer="A",
            evaluation_result=self._mock_result(score=9.0, missing=[]),
            retrieved_context="ctx",
            retrieved_chunks=[("TCP ensures reliable delivery.", 0.9)],
            use_llm=False,
        )
        assert isinstance(result, dict)
        assert result["score"] == 9.0

    def test_zero_score_feedback(self):
        from modules.feedback_generator import generate_feedback

        result = generate_feedback(
            question="Q",
            student_answer="A",
            evaluation_result=self._mock_result(score=1.0, matched=[],
                                                missing=["c1", "c2", "c3"]),
            retrieved_context="ctx",
            retrieved_chunks=[("TCP uses checksum.", 0.9)],
            use_llm=False,
        )
        assert isinstance(result, dict)
        assert len(result["weaknesses"]) >= 1


# ─────────────────────────────────────────────
# Full paper parsing/evaluation tests
# ─────────────────────────────────────────────
class TestFullPaperEvaluator:
    def test_parse_questions_and_answers(self):
        from modules.full_paper_evaluator import parse_questions, parse_answers

        question_text = """Name: Alice
SRN: 123
Course: AI

Q1. Explain TCP.
Q2. Explain UDP.
Q3. Compare TCP and UDP."""
        answer_text = """Name: Alice
SRN: 123

Answer to Q1: TCP is reliable.
Answer to Q3: TCP is reliable, UDP is fast."""

        questions = parse_questions(question_text)
        answers = parse_answers(answer_text)

        assert questions == {
            1: "Explain TCP.",
            2: "Explain UDP.",
            3: "Compare TCP and UDP.",
        }
        assert answers == {
            1: "TCP is reliable.",
            3: "TCP is reliable, UDP is fast.",
        }

    def test_full_paper_evaluation_aggregates_scores(self):
        from modules.full_paper_evaluator import evaluate_all_questions

        call_count = {"retrieve": 0}

        def fake_retrieve(question, vector_store, embedder_fn, top_k=8):
            call_count["retrieve"] += 1
            return [(f"context for {question}", 0.9)]

        def fake_format_context(chunks):
            return "formatted context"

        def fake_extract_concepts(chunks):
            return ["concept one", "concept two"]

        def fake_evaluate_answer(student_answer, expected_concepts, embedder_fn, threshold=0.3, partial_threshold=0.2):
            score = 10.0 if "strong" in student_answer.lower() else 5.0
            return {
                "score": score,
                "score_display": f"{score}/10",
                "matched_concepts": [expected_concepts[0]],
                "partial_concepts": [expected_concepts[1]],
                "missing_concepts": [],
                "concept_scores": [(expected_concepts[0], 0.95), (expected_concepts[1], 0.55)],
                "coverage_percent": 75.0,
                "raw_similarity": 0.75,
                "total_concepts": 2,
                "covered_concepts": 1,
                "partial_concepts_count": 1,
            }

        def fake_feedback(question, student_answer, evaluation_result, retrieved_context, use_llm=True):
            return {
                "question": question,
                "score": evaluation_result["score"],
                "strengths": ["Covered the main point."],
                "weaknesses": ["Could add an example."],
                "suggestions": ["Add one example."],
                "ideal_answer": "A short model answer.",
            }

        def fake_overall_feedback(question_results, total_score, total_max_score, use_llm=True):
            return f"overall {total_score}/{total_max_score}"

        result = evaluate_all_questions(
            question_paper="Q1. What is TCP?\nQ2. What is UDP?",
            answer_script="Answer to Q1: strong answer",
            vector_store=object(),
            retrieve_relevant_chunks_fn=fake_retrieve,
            format_context_fn=fake_format_context,
            extract_concept_sentences_fn=fake_extract_concepts,
            evaluate_answer_fn=fake_evaluate_answer,
            generate_feedback_fn=fake_feedback,
            generate_overall_feedback_fn=fake_overall_feedback,
            embed_single_fn=lambda text: np.zeros(4),
            embed_texts_fn=lambda texts: np.zeros((len(texts), 4)),
            top_k=5,
            threshold=0.3,
            partial_threshold=0.2,
            use_llm=False,
        )

        assert result["total_score"] == 10.0
        assert result["total_max_score"] == 20.0
        assert len(result["results"]) == 2
        assert result["results"][1]["feedback"]["weaknesses"] == ["No answer provided."]
        assert call_count["retrieve"] == 1
        assert result["overall_feedback"].startswith("overall")


# ─────────────────────────────────────────────
# File utils tests
# ─────────────────────────────────────────────
class TestFileUtils:
    def test_extension_detection(self):
        from utils.file_utils import get_file_extension
        assert get_file_extension("notes.pdf") == ".pdf"
        assert get_file_extension("slides.PPTX") == ".pptx"
        assert get_file_extension("file.txt") == ".txt"

    def test_is_document_file(self):
        from utils.file_utils import is_document_file
        assert is_document_file("syllabus.pdf") is True
        assert is_document_file("lecture.pptx") is True
        assert is_document_file("photo.jpg") is False

    def test_is_image_file(self):
        from utils.file_utils import is_image_file
        assert is_image_file("answer.jpg") is True
        assert is_image_file("answer.png") is True
        assert is_image_file("notes.pdf") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

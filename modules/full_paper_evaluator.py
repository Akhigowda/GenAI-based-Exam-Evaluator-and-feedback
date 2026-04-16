"""
Module: full_paper_evaluator.py
Responsibility: Parse full question papers and answer scripts, then evaluate
                each question-answer pair independently.
"""

import re
from typing import Dict, List, Optional


QUESTION_RE = re.compile(r"^\s*Q(?P<num>\d+)\.\s*(?P<body>.*)$", re.IGNORECASE)
ANSWER_RE = re.compile(r"^\s*Answer\s+to\s+Q(?P<num>\d+)\s*:\s*(?P<body>.*)$", re.IGNORECASE)


def parse_questions(question_paper: str) -> Dict[int, str]:
    """Extract only valid questions that start with Q1., Q2., etc."""
    return _parse_numbered_sections(question_paper, QUESTION_RE)


def parse_answers(answer_script: str) -> Dict[int, str]:
    """Extract only answers that start with Answer to Q1:, Answer to Q2:, etc."""
    return _parse_numbered_sections(answer_script, ANSWER_RE)


def evaluate_all_questions(
    question_paper: str,
    answer_script: str,
    vector_store,
    retrieve_relevant_chunks_fn,
    format_context_fn,
    extract_concept_sentences_fn,
    evaluate_answer_fn,
    generate_feedback_fn,
    generate_overall_feedback_fn,
    embed_single_fn,
    embed_texts_fn,
    top_k: int,
    threshold: float,
    partial_threshold: float,
    use_llm: bool,
) -> Dict:
    """Evaluate each question with its matching answer number."""
    questions = parse_questions(question_paper)
    answers = parse_answers(answer_script)

    question_numbers = sorted(questions.keys())
    results: List[Dict] = []
    total_score = 0.0
    total_max_score = 0.0

    for question_number in question_numbers:
        question_text = questions[question_number].strip()
        answer_text = answers.get(question_number, "").strip()

        if not answer_text:
            result = {
                "question_number": question_number,
                "question": question_text,
                "answer": "",
                "answer_provided": False,
                "retrieved_chunks": [],
                "formatted_context": "",
                "concepts": [],
                "evaluation": _build_missing_answer_evaluation(),
                "feedback": {
                    "question": question_text,
                    "score": 0.0,
                    "strengths": [],
                    "weaknesses": ["No answer provided."],
                    "suggestions": ["Write an answer for each question before submitting the script."],
                    "ideal_answer": "",
                },
            }
            results.append(result)
            total_max_score += 10.0
            continue

        retrieved_chunks = retrieve_relevant_chunks_fn(
            question_text,
            vector_store,
            embed_single_fn,
            top_k=top_k,
        )
        formatted_context = format_context_fn(retrieved_chunks)
        concepts = extract_concept_sentences_fn(retrieved_chunks)

        if not concepts:
            evaluation = {
                "score": 0.0,
                "score_display": "0.0/10",
                "matched_concepts": [],
                "partial_concepts": [],
                "missing_concepts": [],
                "concept_scores": [],
                "coverage_percent": 0.0,
                "raw_similarity": 0.0,
                "total_concepts": 0,
                "covered_concepts": 0,
                "partial_concepts_count": 0,
            }
            feedback = "No relevant syllabus content found for this question"
        else:
            evaluation = evaluate_answer_fn(
                student_answer=answer_text,
                expected_concepts=concepts,
                embedder_fn=embed_texts_fn,
                threshold=threshold,
                partial_threshold=partial_threshold,
            )
            feedback = generate_feedback_fn(
                question=question_text,
                student_answer=answer_text,
                evaluation_result=evaluation,
                retrieved_context=formatted_context,
                retrieved_chunks=retrieved_chunks,
                use_llm=use_llm,
            )

        result = {
            "question_number": question_number,
            "question": question_text,
            "answer": answer_text,
            "answer_provided": True,
            "retrieved_chunks": retrieved_chunks,
            "formatted_context": formatted_context,
            "concepts": concepts,
            "evaluation": evaluation,
            "feedback": feedback,
        }
        results.append(result)
        total_score += float(evaluation["score"])
        total_max_score += 10.0

    overall_feedback = generate_overall_feedback_fn(
        question_results=results,
        total_score=total_score,
        total_max_score=total_max_score,
        use_llm=use_llm,
    )

    return {
        "questions": questions,
        "answers": answers,
        "results": results,
        "total_score": round(total_score, 1),
        "total_max_score": round(total_max_score, 1),
        "overall_percentage": round((total_score / total_max_score * 100), 1) if total_max_score else 0.0,
        "overall_feedback": overall_feedback,
    }


def evaluate_full_question_paper(*args, **kwargs) -> Dict:
    """Backward-compatible wrapper for the full-paper evaluator."""
    return evaluate_all_questions(*args, **kwargs)


def evaluate_question_answer(
    question: str,
    answer: str,
    vector_store,
    retrieve_relevant_chunks_fn,
    format_context_fn,
    extract_concept_sentences_fn,
    evaluate_answer_fn,
    generate_feedback_fn,
    embed_single_fn,
    embed_texts_fn,
    top_k: int,
    threshold: float,
    partial_threshold: float,
    use_llm: bool,
) -> Dict:
    """Evaluate a single question-answer pair using the existing RAG pipeline."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if not answer or not answer.strip():
        evaluation = _build_missing_answer_evaluation()
        return {
            "question_number": 1,
            "question": question,
            "answer": "",
            "retrieved_chunks": [],
            "formatted_context": "",
            "concepts": [],
            "evaluation": evaluation,
            "feedback": {
                "question": question,
                "score": 0.0,
                "strengths": [],
                "weaknesses": ["No answer provided."],
                "suggestions": ["Write an answer for the question before submitting."],
                "ideal_answer": "",
            },
        }

    retrieved = retrieve_relevant_chunks_fn(
        question,
        vector_store,
        embed_single_fn,
        top_k=top_k,
    )
    formatted_context = format_context_fn(retrieved)
    concepts = extract_concept_sentences_fn(retrieved)

    if not concepts:
        return {
            "question_number": 1,
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "formatted_context": formatted_context,
            "concepts": [],
            "evaluation": {
                "score": 0.0,
                "score_display": "0.0/10",
                "matched_concepts": [],
                "partial_concepts": [],
                "missing_concepts": [],
                "concept_scores": [],
                "coverage_percent": 0.0,
                "raw_similarity": 0.0,
                "total_concepts": 0,
                "covered_concepts": 0,
                "partial_concepts_count": 0,
            },
            "feedback": {
                "question": question,
                "score": 0.0,
                "strengths": [],
                "weaknesses": ["No relevant syllabus content was found for this question."],
                "suggestions": ["Check the question wording or make sure the syllabus documents are uploaded correctly."],
                "ideal_answer": "",
            },
        }

    evaluation = evaluate_answer_fn(
        student_answer=answer,
        expected_concepts=concepts,
        embedder_fn=embed_texts_fn,
        threshold=threshold,
        partial_threshold=partial_threshold,
    )

    feedback = generate_feedback_fn(
        question=question,
        student_answer=answer,
        evaluation_result=evaluation,
        retrieved_context=formatted_context,
        retrieved_chunks=retrieved,
        use_llm=use_llm,
    )

    return {
        "question_number": 1,
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved,
        "formatted_context": formatted_context,
        "concepts": concepts,
        "evaluation": evaluation,
        "feedback": feedback,
    }


def _parse_numbered_sections(text: str, pattern: re.Pattern) -> Dict[int, str]:
    if not text or not text.strip():
        return {}

    sections: Dict[int, List[str]] = {}
    current_number: Optional[int] = None
    current_lines: List[str] = []

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            if current_number is not None and current_lines:
                current_lines.append("")
            continue

        match = pattern.match(line)
        if match:
            if current_number is not None:
                sections[current_number] = _clean_section_lines(current_lines)
            current_number = int(match.group("num"))
            current_lines = []
            body = match.group("body").strip()
            if body:
                current_lines.append(body)
            continue

        if current_number is None:
            continue

        current_lines.append(line)

    if current_number is not None:
        sections[current_number] = _clean_section_lines(current_lines)

    return {number: text for number, text in sections.items() if text}


def _clean_section_lines(lines: List[str]) -> str:
    cleaned = "\n".join(line.rstrip() for line in lines if line is not None).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _build_missing_answer_evaluation() -> Dict:
    return {
        "score": 0.0,
        "score_display": "0.0/10",
        "matched_concepts": [],
        "partial_concepts": [],
        "missing_concepts": [],
        "concept_scores": [],
        "coverage_percent": 0.0,
        "raw_similarity": 0.0,
        "total_concepts": 0,
        "covered_concepts": 0,
        "partial_concepts_count": 0,
    }
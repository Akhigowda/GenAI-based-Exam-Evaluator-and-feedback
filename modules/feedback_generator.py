"""
Module: feedback_generator.py
Phase: 8 - Feedback Generation
Responsibility: Generate teacher-like feedback using structured data.
"""

import os
import re
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


def generate_feedback(
    question: str,
    student_answer: str,
    evaluation_result: Dict,
    retrieved_context: str,
    retrieved_chunks: Optional[List] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Generate structured, teacher-like feedback for one question."""
    base_strengths = _build_strengths(evaluation_result)
    base_weaknesses = _build_weaknesses(evaluation_result)
    base_suggestions = _build_suggestions(question, evaluation_result, base_strengths, base_weaknesses)
    base_ideal_answer = _build_ideal_answer(
        question=question,
        retrieved_context=retrieved_context,
        retrieved_chunks=retrieved_chunks,
        use_llm=use_llm,
    )

    feedback = {
        "question": question,
        "score": evaluation_result.get("score", 0.0),
        "strengths": base_strengths,
        "weaknesses": base_weaknesses,
        "suggestions": base_suggestions,
        "ideal_answer": base_ideal_answer,
        "llm_used": False,
        "llm_provider": None,
        "llm_error": None,
    }

    provider, api_key = _get_active_llm_config()
    if use_llm and provider and api_key:
        try:
            llm_feedback = _generate_structured_feedback_with_llm(
                provider=provider,
                api_key=api_key,
                question=question,
                student_answer=student_answer,
                evaluation_result=evaluation_result,
                retrieved_context=retrieved_context,
                base_feedback=feedback,
            )
            feedback.update(llm_feedback)
            feedback["llm_used"] = True
            feedback["llm_provider"] = provider
        except Exception as exc:
            feedback["llm_used"] = False
            feedback["llm_provider"] = provider
            feedback["llm_error"] = str(exc)

    return feedback


def generate_overall_feedback(
    question_results: List[Dict],
    total_score: float,
    total_max_score: float,
    use_llm: bool = True,
) -> str:
    """Generate a concise overall summary for a full question paper."""
    provider, api_key = _get_active_llm_config()
    if use_llm and provider and api_key:
        try:
            if provider == "gemini":
                llm_text = _generate_overall_with_gemini(
                    question_results=question_results,
                    total_score=total_score,
                    total_max_score=total_max_score,
                    api_key=api_key,
                )
                return _normalize_overall_feedback_markdown(llm_text, total_score, total_max_score)

            llm_text = _generate_overall_with_openai(
                question_results=question_results,
                total_score=total_score,
                total_max_score=total_max_score,
                api_key=api_key,
            )
            return _normalize_overall_feedback_markdown(llm_text, total_score, total_max_score)
        except Exception as exc:
            print(f"⚠️ {provider.title()} overall feedback failed ({exc}). Using template feedback.")

    template_text = _generate_overall_template_feedback(question_results, total_score, total_max_score)
    return _normalize_overall_feedback_markdown(template_text, total_score, total_max_score)


def feedback_to_markdown(feedback: Dict[str, Any]) -> str:
    """Render structured feedback as readable markdown for the UI."""
    score = feedback.get("score", 0)
    strengths = feedback.get("strengths", [])
    weaknesses = feedback.get("weaknesses", [])
    suggestions = feedback.get("suggestions", [])
    ideal_answer = feedback.get("ideal_answer", "")
    llm_used = feedback.get("llm_used", False)
    llm_provider = feedback.get("llm_provider")
    llm_error = feedback.get("llm_error")

    lines = ["## 📋 Teacher Feedback", f"**Score:** {score}/10", ""]
    if llm_used:
        lines.append(f"**Mode:** LLM-generated ({(llm_provider or '').title()})")
    else:
        lines.append("**Mode:** Template fallback")
        if llm_error:
            lines.append(f"**Reason:** {llm_error[:180]}")
    lines.append("")

    lines.append("### ✅ Strengths")
    if strengths:
        for item in strengths:
            lines.append(f"- {item}")
    else:
        lines.append("- No major strengths detected for this answer.")

    lines.append("")
    lines.append("### ⚠️ Weaknesses")
    if weaknesses:
        for item in weaknesses:
            lines.append(f"- {item}")
    else:
        lines.append("- No major weaknesses detected.")

    lines.append("")
    lines.append("### 💡 Suggestions")
    if suggestions:
        for item in suggestions:
            lines.append(f"- {item}")
    else:
        lines.append("- Keep the same approach and add more depth where needed.")

    lines.append("")
    lines.append("### 📐 Ideal Answer")
    lines.append(ideal_answer or "A concise model answer could not be generated.")

    return "\n".join(lines)


def _build_strengths(evaluation_result: Dict) -> List[str]:
    strengths: List[str] = []
    matched = evaluation_result.get("matched_concepts", [])
    partial = evaluation_result.get("partial_concepts", [])

    for concept in matched:
        strengths.append(_as_positive_sentence(concept))

    for concept in partial:
        strengths.append(_as_positive_sentence(f"you mentioned {concept}, but it needs more detail"))

    return _deduplicate_sentences(strengths)


def _build_weaknesses(evaluation_result: Dict) -> List[str]:
    weaknesses: List[str] = []
    missing = evaluation_result.get("missing_concepts", [])

    for concept in missing:
        weaknesses.append(_as_missing_sentence(concept))

    return _deduplicate_sentences(weaknesses)


def _build_suggestions(
    question: str,
    evaluation_result: Dict,
    strengths: List[str],
    weaknesses: List[str],
) -> List[str]:
    suggestions: List[str] = []
    missing = evaluation_result.get("missing_concepts", [])

    if missing:
        suggestions.append("Add the missing concepts so the answer fully covers the question.")

    if evaluation_result.get("partial_concepts"):
        suggestions.append("Expand the points you mentioned with a short explanation or example.")

    if _needs_structure_hint(question):
        suggestions.append("Organize the answer in clear paragraphs or bullet points so each idea is easy to follow.")

    if not suggestions:
        suggestions.append("Keep the same structure and add one small example to make the answer stronger.")

    if len(strengths) > 0:
        suggestions.append("Keep the concepts you covered correctly, then add the missing points around them.")

    return _deduplicate_sentences(suggestions)


def _build_ideal_answer(
    question: str,
    retrieved_context: str,
    retrieved_chunks: Optional[List],
    use_llm: bool,
) -> str:
    provider, api_key = _get_active_llm_config()
    if use_llm and provider and api_key:
        try:
            if provider == "gemini":
                return _generate_ideal_answer_with_gemini(question, retrieved_context, api_key)
            return _generate_ideal_answer_with_openai(question, retrieved_context, api_key)
        except Exception as exc:
            print(f"⚠️ {provider.title()} ideal answer failed ({exc}). Using template answer.")

    return _build_template_ideal_answer(question, retrieved_context, retrieved_chunks)


def _generate_structured_feedback_with_llm(
    provider: str,
    api_key: str,
    question: str,
    student_answer: str,
    evaluation_result: Dict,
    retrieved_context: str,
    base_feedback: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = _build_structured_feedback_prompt(
        question=question,
        student_answer=student_answer,
        evaluation_result=evaluation_result,
        retrieved_context=retrieved_context,
        base_feedback=base_feedback,
    )

    if provider == "gemini":
        raw = _call_gemini(prompt, api_key)
    else:
        raw = _call_openai(prompt, api_key)

    parsed = _parse_feedback_json(raw)
    return {
        "strengths": _normalize_str_list(parsed.get("strengths"), fallback=base_feedback.get("strengths", [])),
        "weaknesses": _normalize_str_list(parsed.get("weaknesses"), fallback=base_feedback.get("weaknesses", [])),
        "suggestions": _normalize_str_list(parsed.get("suggestions"), fallback=base_feedback.get("suggestions", [])),
        "ideal_answer": _clean_text(parsed.get("ideal_answer") or base_feedback.get("ideal_answer", "")),
    }


def _build_structured_feedback_prompt(
    question: str,
    student_answer: str,
    evaluation_result: Dict,
    retrieved_context: str,
    base_feedback: Dict[str, Any],
) -> str:
    return f"""
You are a helpful teacher. Rewrite feedback in simple, student-friendly language.
Return ONLY valid JSON with keys: strengths, weaknesses, suggestions, ideal_answer.

Rules:
- strengths/weaknesses/suggestions must be short lists (2-6 bullets each)
- weaknesses should sound natural, like guidance from a teacher
- suggestions must be actionable
- ideal_answer must be concise (120-220 words), summarized from context, not copied verbatim

Question:
{question}

Student Answer:
{student_answer}

Evaluation Data:
score: {evaluation_result.get('score_display', '?/10')}
matched: {evaluation_result.get('matched_concepts', [])}
partial: {evaluation_result.get('partial_concepts', [])}
missing: {evaluation_result.get('missing_concepts', [])}

Syllabus context:
{retrieved_context[:2000]}

Base feedback draft (improve this):
strengths={base_feedback.get('strengths', [])}
weaknesses={base_feedback.get('weaknesses', [])}
suggestions={base_feedback.get('suggestions', [])}
ideal_answer={base_feedback.get('ideal_answer', '')}
"""


def _call_openai(prompt: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Respond with JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=700,
    )
    return response.choices[0].message.content or ""


def _call_gemini(prompt: str, api_key: str) -> str:
    return _gemini_generate_text(prompt="Respond with JSON only.\n\n" + prompt, api_key=api_key)


def _gemini_generate_text(prompt: str, api_key: str) -> str:
    """Generate text with Gemini, dynamically discovering available models."""
    try:
        from google import genai  # type: ignore[import-not-found]

        client = genai.Client(api_key=api_key)
        try:
            models = client.models.list()
            available_models = [m.name.replace("models/", "") for m in models if "generateContent" in getattr(m, "supported_generation_methods", [])]
        except Exception:
            available_models = []
        
        # Use discovered models or fall back to standard names
        if not available_models:
            available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        
        for model_name in available_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return _clean_text(getattr(response, "text", "") or "")
            except Exception:
                continue
    except Exception:
        pass

    # Fallback to the deprecated SDK if the new one is unavailable.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai_legacy  # type: ignore[import-not-found]

    genai_legacy.configure(api_key=api_key)
    try:
        models = genai_legacy.list_models()
        available_models = [m.name.replace("models/", "") for m in models if "generateContent" in m.supported_generation_methods]
    except Exception:
        available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    for model_name in available_models:
        try:
            model = genai_legacy.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return _clean_text(getattr(response, "text", "") or "")
        except Exception:
            continue
    
    raise ValueError("No suitable Gemini model available")


def _parse_feedback_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty LLM response")

    # Try direct JSON first.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract first JSON object block.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("LLM response was not valid JSON")
    return json.loads(match.group(0))


def _normalize_str_list(value: Any, fallback: List[str]) -> List[str]:
    if isinstance(value, list):
        cleaned = [_ensure_sentence(_clean_text(item)) for item in value if _clean_text(item)]
        return cleaned if cleaned else fallback
    return fallback


def _generate_ideal_answer_with_openai(question: str, retrieved_context: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a helpful teacher writing a model exam answer. "
        "Write a short, clear, educational answer using the provided syllabus context. "
        "Do not copy raw chunks or list references."
    )
    user_prompt = f"""
Question:
{question}

Syllabus context:
{retrieved_context[:2000]}

Write a concise model answer in 2-4 short paragraphs.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=350,
        temperature=0.5,
    )

    return _clean_text(response.choices[0].message.content)


def _generate_ideal_answer_with_gemini(question: str, retrieved_context: str, api_key: str) -> str:
    prompt = (
        "You are a helpful teacher writing a model exam answer. "
        "Write a short, clear, educational answer using the provided syllabus context. "
        "Do not copy raw chunks or list references.\n\n"
        f"Question:\n{question}\n\n"
        f"Syllabus context:\n{retrieved_context[:2000]}\n\n"
        "Write a concise model answer in 2-4 short paragraphs."
    )
    return _gemini_generate_text(prompt=prompt, api_key=api_key)


def _build_template_ideal_answer(
    question: str,
    retrieved_context: str,
    retrieved_chunks: Optional[List],
) -> str:
    sentences = _extract_candidate_sentences(retrieved_context, retrieved_chunks)
    if not sentences:
        return "A short model answer could not be generated from the retrieved content."

    selected = []
    for sentence in sentences:
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) == 3:
            break

    if not selected:
        return "A short model answer could not be generated from the retrieved content."

    answer = " ".join(selected)
    answer = _clean_text(answer)
    if not answer.endswith("."):
        answer += "."
    return answer


def _extract_candidate_sentences(retrieved_context: str, retrieved_chunks: Optional[List]) -> List[str]:
    raw_text = []
    if retrieved_chunks:
        for chunk in retrieved_chunks:
            if isinstance(chunk, (list, tuple)) and chunk:
                raw_text.append(str(chunk[0]))
            else:
                raw_text.append(str(chunk))

    if not raw_text and retrieved_context:
        for line in retrieved_context.splitlines():
            if line.startswith("[Reference") or line.startswith("---"):
                continue
            raw_text.append(line)

    text = " ".join(raw_text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    candidates = []
    for sentence in sentences:
        cleaned = _clean_text(sentence)
        if len(cleaned) >= 40:
            candidates.append(cleaned)
    return candidates


def _generate_overall_with_openai(
    question_results: List[Dict],
    total_score: float,
    total_max_score: float,
    api_key: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    summary_lines = []
    for result in question_results:
        evaluation = result["evaluation"]
        summary_lines.append(
            f"Q{result['question_number']}: {evaluation['score_display']} | Missing: {', '.join(evaluation.get('missing_concepts', [])[:5]) or 'None'}"
        )

    system_prompt = "You are an experienced university examiner reviewing a full question paper. Keep the feedback concise and constructive."
    user_prompt = f"""
Total score: {total_score:.1f}/{total_max_score:.1f}

Per-question summary:
{chr(10).join(summary_lines) if summary_lines else 'No evaluated questions.'}

Write overall feedback with these sections:
1. Overall performance
2. Strengths
3. Main gaps
4. How to improve across the paper
5. Closing remark
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=650,
        temperature=0.6,
    )

    return response.choices[0].message.content


def _generate_overall_with_gemini(
    question_results: List[Dict],
    total_score: float,
    total_max_score: float,
    api_key: str,
) -> str:
    summary_lines = []
    for result in question_results:
        evaluation = result["evaluation"]
        summary_lines.append(
            f"Q{result['question_number']}: {evaluation['score_display']} | Missing: {', '.join(evaluation.get('missing_concepts', [])[:5]) or 'None'}"
        )

    prompt = (
        "You are an experienced university examiner reviewing a full question paper. "
        "Keep the feedback concise and constructive.\n\n"
        f"Total score: {total_score:.1f}/{total_max_score:.1f}\n\n"
        "Per-question summary:\n"
        f"{chr(10).join(summary_lines) if summary_lines else 'No evaluated questions.'}\n\n"
        "Write overall feedback with these sections:\n"
        "1. Overall performance\n"
        "2. Strengths\n"
        "3. Main gaps\n"
        "4. How to improve across the paper\n"
        "5. Closing remark"
    )

    return _gemini_generate_text(prompt=prompt, api_key=api_key)


def _get_active_llm_config() -> tuple[Optional[str], Optional[str]]:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if provider == "gemini" and gemini_key:
        return "gemini", gemini_key
    if provider == "openai" and openai_key:
        return "openai", openai_key

    # Fallback preference when provider is not explicitly set.
    if gemini_key:
        return "gemini", gemini_key
    if openai_key:
        return "openai", openai_key
    return None, None


def _generate_overall_template_feedback(
    question_results: List[Dict],
    total_score: float,
    total_max_score: float,
) -> str:
    from modules.evaluator import get_score_grade, get_performance_label

    percentage = round((total_score / total_max_score * 100), 1) if total_max_score else 0.0
    average_score = round(total_score / len(question_results), 1) if question_results else 0.0
    grade, emoji = get_score_grade(average_score)

    all_missing = []
    for result in question_results:
        all_missing.extend(result["evaluation"].get("missing_concepts", []))

    lines = []
    lines.append("## 📚 Overall Paper Feedback\n")
    lines.append(f"**Total Score:** {total_score:.1f}/{total_max_score:.1f} {emoji} &nbsp; **Average Grade:** {grade}\n")
    lines.append(f"**Coverage:** {percentage}%")
    lines.append(f"> {get_performance_label(percentage)}\n")
    lines.append("---\n")

    strong_answers = [
        result for result in question_results if result["evaluation"].get("score", 0) >= 8
    ]
    weak_answers = [
        result for result in question_results if result["evaluation"].get("score", 0) < 5
    ]

    lines.append("### ✅ Strengths\n")
    if strong_answers:
        for result in strong_answers:
            lines.append(
                f"- Q{result['question_number']}: strong coverage with {result['evaluation'].get('score_display', '?/10')}"
            )
    else:
        lines.append("- No answer achieved a clearly strong score, but some questions were partially covered.")

    lines.append("\n### ⚠️ Main Gaps\n")
    if weak_answers:
        for result in weak_answers:
            missing = result["evaluation"].get("missing_concepts", [])
            lines.append(
                f"- Q{result['question_number']}: needs more detail on {', '.join(missing[:3]) if missing else 'key syllabus points'}"
            )
    else:
        lines.append("- No major gaps detected in the scored answers.")

    lines.append("\n### 💡 How to Improve Across the Paper\n")
    if all_missing:
        unique_missing = []
        seen = set()
        for concept in all_missing:
            key = concept.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_missing.append(concept)
        for concept in unique_missing[:8]:
            lines.append(f"- Revise: {concept}")
    else:
        lines.append("- Keep the same structure, but add more examples and precise terminology where possible.")

    return "\n".join(lines)


def _normalize_overall_feedback_markdown(text: str, total_score: float, total_max_score: float) -> str:
    """Normalize overall feedback into clean, sectioned markdown."""
    raw = (text or "").strip()
    if not raw:
        return (
            "## 📚 Overall Paper Feedback\n\n"
            f"**Total Score:** {total_score:.1f}/{total_max_score:.1f}\n\n"
            "### 1. Overall Performance\n"
            "No summary generated.\n\n"
            "### 2. Strengths\n"
            "- Not available.\n\n"
            "### 3. Main Gaps\n"
            "- Not available.\n\n"
            "### 4. How to Improve Across the Paper\n"
            "- Not available.\n\n"
            "### 5. Closing Remark\n"
            "Keep practicing consistently."
        )

    # Cleanup common clutter and force heading boundaries onto new lines.
    cleaned = raw.replace("\r\n", "\n")
    cleaned = re.sub(r"^\s*#+\s*overall\s+feedback\s*:?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(
        r"\s+(?=(?:\d+\.)\s*(Overall Performance|Strengths|Main Gaps|How to Improve Across the Paper|Closing Remark))",
        "\n",
        cleaned,
        flags=re.I,
    )

    sections = [
        ("Overall Performance", "Overall Performance"),
        ("Strengths", "Strengths"),
        ("Main Gaps", "Main Gaps"),
        ("How to Improve Across the Paper", "How to Improve Across the Paper"),
        ("Closing Remark", "Closing Remark"),
    ]

    extracted: Dict[str, str] = {}
    for key, label in sections:
        extracted[key] = _extract_named_section(cleaned, label)

    # Fallback: if section extraction fails, improve readability by formatting numbered blocks.
    if not any(extracted.values()):
        fallback = re.sub(r"\s*(\d+\.\s*)", r"\n\n\1", cleaned)
        fallback = re.sub(r"\s+([*•-])\s+", r"\n\1 ", fallback)
        return f"## 📚 Overall Paper Feedback\n\n**Total Score:** {total_score:.1f}/{total_max_score:.1f}\n\n{fallback.strip()}"

    lines: List[str] = [
        "## 📚 Overall Paper Feedback",
        "",
        f"**Total Score:** {total_score:.1f}/{total_max_score:.1f}",
        "",
    ]

    for idx, (key, _label) in enumerate(sections, start=1):
        body = _clean_text(extracted.get(key, ""))
        lines.append(f"### {idx}. {key}")
        if key in {"Strengths", "Main Gaps", "How to Improve Across the Paper"}:
            bullet_points = _extract_bullet_points(body)
            if bullet_points:
                lines.extend([f"- {point}" for point in bullet_points])
            else:
                lines.append(f"- {body or 'Not available.'}")
        else:
            lines.append(body or "Not available.")
        lines.append("")

    return "\n".join(lines).strip()


def _extract_named_section(text: str, name: str) -> str:
    names = [
        "Overall Performance",
        "Strengths",
        "Main Gaps",
        "How to Improve Across the Paper",
        "Closing Remark",
    ]
    other_names = [n for n in names if n.lower() != name.lower()]
    lookahead = "|".join(re.escape(n) for n in other_names)
    pattern = (
        rf"(?is)(?:^|\n|\s)(?:\d+\.\s*)?{re.escape(name)}\s*:?\s*"
        rf"(.*?)(?=(?:\n|\s)(?:\d+\.\s*)?(?:{lookahead})\s*:?|$)"
    )
    match = re.search(pattern, text)
    return _clean_text(match.group(1)) if match else ""


def _extract_bullet_points(text: str) -> List[str]:
    if not text:
        return []

    normalized = text.replace("•", "-")
    normalized = re.sub(r"\s+\*\s+", "\n- ", normalized)
    normalized = re.sub(r"\s+-\s+", "\n- ", normalized)

    points = []
    for line in normalized.splitlines():
        line = _clean_text(line)
        if line.startswith("- ") or line.startswith("* "):
            points.append(_ensure_sentence(line[2:]))

    if points:
        return _deduplicate_sentences(points)

    # If no explicit bullets, split by semicolons as a readable fallback.
    if ";" in text:
        semicolon_points = [_ensure_sentence(_clean_text(p)) for p in text.split(";") if _clean_text(p)]
        return _deduplicate_sentences(semicolon_points)

    return []


def _as_positive_sentence(concept: str) -> str:
    text = _clean_text(concept)
    if not text:
        return "You correctly covered an important concept."
    lowered = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
    if lowered.startswith("you "):
        return _ensure_sentence(text)
    return _ensure_sentence(f"You correctly mentioned that {lowered}")


def _as_missing_sentence(concept: str) -> str:
    text = _clean_text(concept)
    if not text:
        return "You missed an important concept from the question."
    lowered = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
    return _ensure_sentence(f"You did not mention that {lowered}")


def _needs_structure_hint(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in ["compare", "difference", "explain", "describe", "list", "write"])


def _ensure_sentence(text: str) -> str:
    cleaned = _clean_text(text)
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _deduplicate_sentences(sentences: List[str]) -> List[str]:
    seen = set()
    unique = []
    for sentence in sentences:
        cleaned = _clean_text(sentence)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(_ensure_sentence(cleaned))
    return unique
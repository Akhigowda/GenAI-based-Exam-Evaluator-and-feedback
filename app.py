"""
app.py — AI Exam Answer Evaluator & Improvement Coach
Phase: 9 - Full Integration (Streamlit UI)
"""

import os
import sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Exam Evaluator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #a8dadc; font-size: 1rem; margin: 0.5rem 0 0 0; }
    .score-card {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border: 2px solid #e94560; border-radius: 16px;
        padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .score-number { font-size: 4rem; font-weight: 900; color: #e94560; }
    .score-label  { color: #a8dadc; font-size: 1rem; }
    .concept-matched {
        background: #1a3a2a; border-left: 4px solid #2ecc71;
        padding: 0.5rem 1rem; margin: 0.3rem 0; border-radius: 0; color: #a8f0c6;
    }
    .concept-partial {
        background: #2e2010; border-left: 4px solid #f39c12;
        padding: 0.5rem 1rem; margin: 0.3rem 0; border-radius: 0; color: #f5d08a;
    }
    .concept-missing {
        background: #3a1a1a; border-left: 4px solid #e74c3c;
        padding: 0.5rem 1rem; margin: 0.3rem 0; border-radius: 0; color: #f0a8a8;
    }
    .mode-warning {
        background: #2e1f00; border-left: 4px solid #f39c12;
        padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0; color: #f5d08a;
        font-size: 0.9rem;
    }
    .stProgress > div > div { background: #e94560 !important; }
    section[data-testid="stSidebar"] { background: #0d1117; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for key, val in [("vector_store", None), ("kb_built", False), ("chunk_count", 0)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract text from an uploaded PDF file object
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_uploaded_pdf(uploaded_file) -> str:
    from utils.file_utils import save_uploaded_file, cleanup_file
    from modules.document_processor import extract_text
    saved = save_uploaded_file(uploaded_file)
    try:
        text = extract_text(saved)
    finally:
        cleanup_file(saved)
    return text

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Test API connections
# ─────────────────────────────────────────────────────────────────────────────
def test_gemini_connection(api_key: str) -> tuple[bool, str]:
    """Test Gemini API connection by listing available models and testing one."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # List available models
        try:
            models = genai.list_models()
            available_models = [m.name.replace("models/", "") for m in models if "generateContent" in m.supported_generation_methods]
            
            if not available_models:
                return False, "❌ No text generation models available with your API key"
            
            # Try each available model
            for model_name in available_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content("Say 'OK' and nothing else.")
                    if response and response.text:
                        return True, f"✅ Gemini API connection successful! (Model: {model_name})"
                except Exception as e:
                    continue
            
            return False, f"❌ Available models found but all failed: {', '.join(available_models[:3])}"
        except Exception as list_error:
            # If list_models fails, fall back to trying standard models
            models_to_try = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
            last_error = None
            
            for model_name in models_to_try:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content("Say 'OK' and nothing else.")
                    if response and response.text:
                        return True, f"✅ Gemini API connection successful! (Model: {model_name})"
                except Exception as e:
                    last_error = str(e)
                    continue
            
            return False, f"❌ Gemini API error: {last_error or 'No models available'}"
    except Exception as e:
        return False, f"❌ Gemini API error: {str(e)}"

def test_openai_connection(api_key: str) -> tuple[bool, str]:
    """Test OpenAI API connection with a simple prompt."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
            max_tokens=10,
            temperature=0
        )
        if response and response.choices and response.choices[0].message.content:
            return True, "✅ OpenAI API connection successful!"
        else:
            return False, "⚠️ OpenAI API returned empty response"
    except Exception as e:
        return False, f"❌ OpenAI API error: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k = st.slider("Top-K Retrieval", 3, 15, 8,
                      help="Number of syllabus chunks to retrieve per question")
    threshold = st.slider("Similarity Threshold", 0.15, 0.8, 0.30, 0.05,
                          help="Min similarity to count a concept as fully covered")
    partial_threshold = st.slider("Partial Credit Threshold", 0.10, 0.5, 0.20, 0.05,
                                  help="Min similarity for partial credit (0.5 marks)")
    chunk_size    = st.slider("Chunk Size (chars)", 200, 1000, 400, 50)
    chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 200, 80, 10)

    st.markdown("### 🤖 LLM Feedback")
    provider_options = ["None", "Gemini", "OpenAI"]
    default_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if default_provider == "gemini":
        default_index = 1
    elif default_provider == "openai":
        default_index = 2
    else:
        default_index = 0

    llm_provider = st.selectbox(
        "LLM Provider",
        provider_options,
        index=default_index,
        help="Choose Gemini or OpenAI for AI-generated feedback.",
    )

    key_label = "Paste Gemini API Key" if llm_provider == "Gemini" else "Paste OpenAI API Key"
    key_help = "Key is used only for the current app session." if llm_provider != "None" else None
    typed_api_key = ""
    if llm_provider != "None":
        typed_api_key = st.text_input(
            key_label,
            type="password",
            placeholder="Enter API key here...",
            help=key_help,
        ).strip()

    # Keep provider and key in environment so generator modules can read them.
    if llm_provider == "Gemini":
        os.environ["LLM_PROVIDER"] = "gemini"
        if typed_api_key:
            os.environ["GEMINI_API_KEY"] = typed_api_key
    elif llm_provider == "OpenAI":
        os.environ["LLM_PROVIDER"] = "openai"
        if typed_api_key:
            os.environ["OPENAI_API_KEY"] = typed_api_key
    else:
        os.environ["LLM_PROVIDER"] = ""

    if llm_provider == "Gemini":
        llm_available = bool(typed_api_key or os.getenv("GEMINI_API_KEY"))
    elif llm_provider == "OpenAI":
        llm_available = bool(typed_api_key or os.getenv("OPENAI_API_KEY"))
    else:
        llm_available = False

    use_llm = st.toggle("Use LLM Feedback", value=llm_available)
    if use_llm and not llm_available:
        st.warning("Please paste an API key above to enable LLM feedback.")

    # Test API Connection buttons
    if llm_provider in ["Gemini", "OpenAI"] and typed_api_key:
        st.markdown("#### 🧪 Test Connection")
        col1, col2 = st.columns(2)
        
        if llm_provider == "Gemini":
            with col1:
                if st.button("Test Gemini", width="stretch"):
                    with st.spinner("Testing Gemini API..."):
                        success, message = test_gemini_connection(typed_api_key)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        else:  # OpenAI
            with col1:
                if st.button("Test OpenAI", width="stretch"):
                    with st.spinner("Testing OpenAI API..."):
                        success, message = test_openai_connection(typed_api_key)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)

    st.markdown("---")
    st.markdown("### 📊 Knowledge Base Status")
    if st.session_state.kb_built:
        st.success(f"✅ Ready — {st.session_state.chunk_count} chunks indexed")
    else:
        st.info("⏳ Upload documents to build")

    if st.session_state.kb_built:
        if st.button("🗑️ Reset Knowledge Base"):
            st.session_state.vector_store = None
            st.session_state.kb_built     = False
            st.session_state.chunk_count  = 0
            st.rerun()

    st.markdown("---")
    st.markdown("""
    ### 📖 How It Works
    1. Upload syllabus/textbook
    2. System builds knowledge base
    3. Upload a question paper
    4. Upload the matching answer script
    5. Get per-question scores + overall feedback

    ### 🎯 Score Guide
    - ✅ Full match ≥ threshold → 1 point
    - 🔶 Partial ≥ partial threshold → 0.5 points
    - ❌ Below partial threshold → 0 points

    ### 📌 Mode Guide
        - **Full Question Paper** — parses every question and matches answers in order
        - **Single Question** — fallback for one-off checks
    """)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 AI Exam Answer Evaluator</h1>
    <p>Upload your syllabus → Ask a question → Evaluate any answer instantly</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Build Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 📁 Step 1: Build Knowledge Base")
st.caption("Upload your syllabus, textbooks, or lecture slides (PDF, PPTX, TXT)")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "pptx", "txt", "md"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"📎 {len(uploaded_files)} file(s) selected:")
        for f in uploaded_files:
            st.caption(f"  • {f.name} ({f.size // 1024} KB)")
    with col2:
        build_btn = st.button("🔨 Build Knowledge Base", type="primary",
                              width="stretch")

    if build_btn:
        with st.status("Building knowledge base...", expanded=True) as status:

            st.write("📄 **Phase 2:** Extracting text from documents...")
            from modules.document_processor import extract_text
            from utils.file_utils import save_uploaded_file, cleanup_file

            all_texts, failed_files = [], []
            for uf in uploaded_files:
                try:
                    saved_path = save_uploaded_file(uf)
                    text = extract_text(saved_path)
                    all_texts.append(text)
                    cleanup_file(saved_path)
                    st.write(f"  ✅ {uf.name}: {len(text):,} chars extracted")
                except Exception as e:
                    failed_files.append(uf.name)
                    st.write(f"  ❌ {uf.name}: {str(e)}")

            if not all_texts:
                status.update(label="❌ No text could be extracted.", state="error")
                st.stop()

            st.write("✂️ **Phase 3:** Chunking text...")
            from modules.chunker import chunk_documents, get_chunk_stats
            chunks = chunk_documents(all_texts)
            stats  = get_chunk_stats(chunks)
            st.write(f"  ✅ {stats['count']} chunks (avg {stats['avg_length']} chars)")

            st.write("🧠 **Phase 4:** Generating embeddings...")
            from modules.embedder import embed_texts
            embeddings = embed_texts(chunks)
            st.write(f"  ✅ Embeddings shape: {embeddings.shape}")

            st.write("🗄️ **Phase 5:** Building FAISS vector database...")
            from modules.vector_store import VectorStore
            store = VectorStore()
            store.build(chunks, embeddings)
            st.session_state.vector_store = store
            st.session_state.kb_built     = True
            st.session_state.chunk_count  = stats["count"]
            st.write(f"  ✅ FAISS index ready: {store.size} vectors")

            status.update(
                label=f"✅ Knowledge base built! {stats['count']} chunks indexed.",
                state="complete"
            )
        st.balloons()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Question + Answer + Evaluate
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📝 Step 2: Evaluate Questions and Answers")

if not st.session_state.kb_built:
    st.info("👆 Upload documents and build the knowledge base first.")
    st.stop()

# ── Evaluation mode ───────────────────────────────────────────────────────────
st.markdown("### 🔀 Evaluation Mode")
eval_mode = st.radio(
    "Evaluation mode",
    ["Full Question Paper", "Single Question (Fallback)"],
    horizontal=True,
    label_visibility="collapsed",
)

if eval_mode == "Full Question Paper":
    st.info(
        "Upload a full question paper and the full answer script. The evaluator will split both, pair questions with answers in order, and score each pair separately.",
        icon=None,
    )
else:
    st.info(
        "💡 **Single Question mode** — Use this only for one question and one answer. If you already have a full paper, switch back to Full Question Paper.",
        icon=None,
    )

st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# Question input (shared by both modes, label changes)
# ─────────────────────────────────────────────────────────────────────────────
if eval_mode == "Full Question Paper":
    st.markdown("### 🔹 Full Question Paper")
    st.caption("Paste or upload the complete paper with all questions")
else:
    st.markdown("### 🔹 Exam Question")
    st.caption("Enter one specific question only")

question_input_type = st.radio(
    "Question input type",
    ["Type question", "Upload question PDF", "Upload question image"],
    horizontal=True,
    label_visibility="collapsed",
    key="q_input_mode",
)

question_text = ""

if question_input_type == "Type question":
    placeholder = (
        "Paste the full question paper here with Q1., Q2., Q3., ..."
        if eval_mode == "Full Question Paper"
        else "e.g., What does UDP stand for?"
    )
    question_text = st.text_area(
        "Question input",
        placeholder=placeholder,
        height=100,
        label_visibility="collapsed",
        key="q_type_input",
    )

elif question_input_type == "Upload question PDF":
    q_pdf = st.file_uploader(
        "Upload question paper PDF", type=["pdf"], key="q_pdf"
    )
    if q_pdf:
        with st.spinner("Extracting text from PDF..."):
            try:
                raw_q = extract_text_from_uploaded_pdf(q_pdf)
                st.success(f"✅ Extracted {len(raw_q):,} characters from PDF")
            except Exception as e:
                st.error(f"PDF extraction failed: {e}")
                raw_q = ""

        if raw_q:
            if eval_mode == "Full Question Paper":
                # Keep the entire paper for parsing into individual questions.
                question_text = st.text_area(
                    "Extracted question paper (edit if needed):",
                    value=raw_q,
                    height=180,
                    key="q_pdf_edit",
                )
            else:
                # Show full text in expander, let user copy one question
                with st.expander("📄 Full extracted PDF — copy the question you want to evaluate"):
                    st.text(raw_q)
                question_text = st.text_area(
                    "Paste the specific question to evaluate:",
                    placeholder="Copy one question from the PDF above...",
                    height=100,
                    key="q_single_paste",
                )

else:
    q_image = st.file_uploader(
        "Upload question image", type=["png", "jpg", "jpeg"], key="q_img"
    )
    if q_image:
        from utils.ocr_utils import extract_text_from_image_bytes, is_tesseract_available
        if is_tesseract_available():
            ocr_q = extract_text_from_image_bytes(q_image.read())
            question_text = st.text_area(
                "OCR result (edit if needed):",
                value=ocr_q,
                height=120,
                key="q_ocr_edit",
            )
        else:
            st.error("Tesseract not installed. Use typed input or install Tesseract.")

# ─────────────────────────────────────────────────────────────────────────────
# Answer input (label changes based on mode)
# ─────────────────────────────────────────────────────────────────────────────
if eval_mode == "Full Question Paper":
    st.markdown("### 🔹 Full Answer Script")
    st.caption("Upload or paste the complete answer script")
else:
    st.markdown("### 🔹 Student Answer")
    st.caption("Enter only the student's answer to the question above")

answer_input_type = st.radio(
    "Answer input type",
    ["Type answer", "Upload answer PDF", "Upload answer image"],
    horizontal=True,
    label_visibility="collapsed",
    key="a_input_mode",
)

student_answer = ""

if answer_input_type == "Type answer":
    placeholder_a = (
        "Paste the complete answer script with Answer to Q1:, Answer to Q2:, ..."
        if eval_mode == "Full Question Paper"
        else "Paste only the answer to the specific question above..."
    )
    student_answer = st.text_area(
        "Answer input",
        placeholder=placeholder_a,
        height=200,
        label_visibility="collapsed",
        key="a_type_input",
    )

elif answer_input_type == "Upload answer PDF":
    a_pdf = st.file_uploader(
        "Upload answer script PDF", type=["pdf"], key="a_pdf"
    )
    if a_pdf:
        with st.spinner("Extracting text from PDF..."):
            try:
                raw_a = extract_text_from_uploaded_pdf(a_pdf)
                st.success(f"✅ Extracted {len(raw_a):,} characters from PDF")
            except Exception as e:
                st.error(f"PDF extraction failed: {e}")
                raw_a = ""

        if raw_a:
            if eval_mode == "Full Question Paper":
                # Full paper mode — keep the entire answer script.
                student_answer = st.text_area(
                    "Extracted answer script (edit if needed):",
                    value=raw_a,
                    height=220,
                    key="a_pdf_edit",
                )
            else:
                # Show full text, let user paste just the relevant answer
                with st.expander("📄 Full extracted answer PDF — copy the relevant section"):
                    st.text(raw_a)
                student_answer = st.text_area(
                    "Paste the specific answer for this question:",
                    placeholder="Copy only the answer to the question above...",
                    height=200,
                    key="a_single_paste",
                )

else:
    a_image = st.file_uploader(
        "Upload answer image (handwritten)", type=["png", "jpg", "jpeg"], key="a_img"
    )
    if a_image:
        from utils.ocr_utils import extract_text_from_image_bytes, is_tesseract_available
        if is_tesseract_available():
            ocr_a = extract_text_from_image_bytes(a_image.read())
            student_answer = st.text_area(
                "OCR Result (edit if needed):",
                value=ocr_a,
                height=150,
                key="ocr_answer_edit",
            )
        else:
            st.error("Tesseract not installed. Use typed input or install Tesseract.")

# ── Evaluate Button ───────────────────────────────────────────────────────────
st.markdown("")
evaluate_btn = st.button(
    "🚀 Evaluate Paper" if eval_mode == "Full Question Paper" else "🚀 Evaluate Answer",
    type="primary",
    width="stretch",
    disabled=(not question_text.strip() or not student_answer.strip()),
)

if evaluate_btn:
    if not question_text.strip():
        st.warning("Please enter a question.")
        st.stop()
    if not student_answer.strip():
        st.warning("Please enter the student's answer.")
        st.stop()

    with st.spinner("Evaluating... ⏳"):
        from modules.full_paper_evaluator import (
            evaluate_all_questions,
            evaluate_question_answer,
        )
        from modules.retriever import (
            retrieve_relevant_chunks,
            format_context,
            extract_concept_sentences,
        )
        from modules.evaluator import evaluate_answer, get_score_grade, get_performance_label
        from modules.embedder import embed_single, embed_texts as _embed_texts
        from modules.feedback_generator import generate_feedback, generate_overall_feedback, feedback_to_markdown

        if eval_mode == "Full Question Paper":
            evaluation_bundle = evaluate_all_questions(
                question_paper=question_text,
                answer_script=student_answer,
                vector_store=st.session_state.vector_store,
                retrieve_relevant_chunks_fn=retrieve_relevant_chunks,
                format_context_fn=format_context,
                extract_concept_sentences_fn=extract_concept_sentences,
                evaluate_answer_fn=evaluate_answer,
                generate_feedback_fn=generate_feedback,
                generate_overall_feedback_fn=generate_overall_feedback,
                embed_single_fn=embed_single,
                embed_texts_fn=_embed_texts,
                top_k=top_k,
                threshold=threshold,
                partial_threshold=partial_threshold,
                use_llm=use_llm,
            )
        else:
            single_result = evaluate_question_answer(
                question=question_text,
                answer=student_answer,
                vector_store=st.session_state.vector_store,
                retrieve_relevant_chunks_fn=retrieve_relevant_chunks,
                format_context_fn=format_context,
                extract_concept_sentences_fn=extract_concept_sentences,
                evaluate_answer_fn=evaluate_answer,
                generate_feedback_fn=generate_feedback,
                embed_single_fn=embed_single,
                embed_texts_fn=_embed_texts,
                top_k=top_k,
                threshold=threshold,
                partial_threshold=partial_threshold,
                use_llm=use_llm,
            )
            evaluation_bundle = {
                "questions": [question_text],
                "answers": [student_answer],
                "pairs": [{"question_number": "1", "question": question_text, "answer": student_answer}],
                "results": [single_result],
                "total_score": single_result["evaluation"]["score"],
                "total_max_score": 10.0,
                "overall_percentage": single_result["evaluation"]["coverage_percent"],
                "overall_feedback": feedback_to_markdown(single_result["feedback"]),
            }

        if eval_mode == "Full Question Paper" and not evaluation_bundle["results"]:
            st.warning("No evaluable questions were detected. Check the numbering or formatting of the paper and answer script.")

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Evaluation Results")

    mode_label = "Full Paper" if eval_mode == "Full Question Paper" else "Single Question"
    results = evaluation_bundle["results"]
    total_score = evaluation_bundle["total_score"]
    total_max_score = evaluation_bundle["total_max_score"]
    overall_percentage = evaluation_bundle.get("overall_percentage", 0.0)
    average_score = round(total_score / len(results), 1) if results else 0.0
    grade, emoji = get_score_grade(average_score)

    st.caption(
        f"📌 Mode: **{mode_label}** | Questions evaluated: **{len(results)}** | Total: **{total_score:.1f}/{total_max_score:.1f}**"
    )

    col_score, col_grade, col_coverage = st.columns(3)

    with col_score:
        st.markdown(
            f"""
        <div class="score-card">
            <div class="score-number">{total_score:.1f}/{total_max_score:.1f}</div>
            <div class="score-label">Total Score</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with col_grade:
        st.markdown(
            f"""
        <div class="score-card">
            <div class="score-number">{emoji}</div>
            <div class="score-label">Average Grade: {grade}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with col_coverage:
        st.markdown(
            f"""
        <div class="score-card">
            <div class="score-number">{overall_percentage}%</div>
            <div class="score-label">Overall Coverage</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.progress(overall_percentage / 100 if total_max_score else 0)
    st.caption(get_performance_label(overall_percentage))

    summary_rows = []
    for item in results:
        evaluation = item["evaluation"]
        summary_rows.append(
            {
                "Question": f"Q{item['question_number']}",
                "Score": evaluation.get("score_display", "?/10"),
                "Coverage": f"{evaluation.get('coverage_percent', 0)}%",
                "Missing concepts": "; ".join(evaluation.get("missing_concepts", [])[:4]) or "None",
            }
        )

    if summary_rows:
        st.markdown("### 🧾 Marks by Question")
        st.dataframe(summary_rows, width="stretch", hide_index=True)

    for item in results:
        evaluation = item["evaluation"]
        question = item["question"]
        answer = item["answer"]
        grade_i, emoji_i = get_score_grade(evaluation["score"])

        st.markdown(f"### Question {item['question_number']}")
        st.write(question)
        st.caption(f"Answer length: {len(answer.strip())} chars | Score: {evaluation['score_display']} | Grade: {grade_i} {emoji_i}")

        q_score, q_grade, q_cov = st.columns(3)
        with q_score:
            st.metric("Score", evaluation["score_display"])
        with q_grade:
            st.metric("Grade", f"{grade_i} {emoji_i}")
        with q_cov:
            st.metric("Coverage", f"{evaluation['coverage_percent']}%")

        q_total = evaluation.get("total_concepts", 0)
        q_covered = evaluation.get("covered_concepts", 0)
        q_partial = evaluation.get("partial_concepts", [])
        q_missing = evaluation.get("missing_concepts", [])

        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Fully covered", f"{q_covered} / {q_total}")
        c2.metric("🔶 Partial credit", f"{len(q_partial)} / {q_total}")
        c3.metric("❌ Missing", f"{len(q_missing)} / {q_total}")

        col_matched, col_missing_col = st.columns(2)
        with col_matched:
            st.markdown(f"#### ✅ Fully Covered ({q_covered})")
            if evaluation["matched_concepts"]:
                for concept in evaluation["matched_concepts"]:
                    st.markdown(f'<div class="concept-matched">✓ {concept}</div>', unsafe_allow_html=True)
            else:
                st.info("No concepts fully matched.")

            if q_partial:
                st.markdown(f"#### 🔶 Partial Credit ({len(q_partial)})")
                for concept in q_partial:
                    st.markdown(f'<div class="concept-partial">~ {concept}</div>', unsafe_allow_html=True)

        with col_missing_col:
            st.markdown(f"#### ❌ Missing ({len(q_missing)})")
            if q_missing:
                for concept in q_missing:
                    st.markdown(f'<div class="concept-missing">✗ {concept}</div>', unsafe_allow_html=True)
            else:
                st.success("All expected concepts covered! 🎉")

        with st.expander(f"📈 Detailed Concept Similarity Scores - Q{item['question_number']}"):
            st.caption(f"Full match ≥ {threshold} | Partial ≥ {partial_threshold}")
            for concept, score in evaluation["concept_scores"]:
                col_bar, col_text = st.columns([1, 4])
                with col_bar:
                    if score >= threshold:
                        mark = "✅"
                    elif score >= partial_threshold:
                        mark = "🔶"
                    else:
                        mark = "❌"
                    st.write(f"{mark} `{score:.3f}`")
                with col_text:
                    st.write(concept[:120])

        with st.expander(f"🔍 Retrieved Syllabus Context - Q{item['question_number']}"):
            st.caption(f"{len(item['concepts'])} concepts extracted from {len(item['retrieved_chunks'])} retrieved chunks")
            st.markdown(item["formatted_context"])

        st.markdown("### 💬 Teacher Feedback")
        feedback_obj = item["feedback"]
        if isinstance(feedback_obj, dict):
            if feedback_obj.get("llm_used"):
                provider = str(feedback_obj.get("llm_provider") or "LLM").title()
                st.caption(f"Feedback mode: {provider} (LLM)")
            else:
                st.caption("Feedback mode: Template fallback")
        st.markdown(feedback_to_markdown(feedback_obj))

    st.markdown("---")
    st.markdown("## 🧠 Overall Feedback")
    st.markdown(evaluation_bundle["overall_feedback"])

    report_lines = [
        "# Exam Evaluation Report",
        f"## Mode: {mode_label}",
        "",
        f"## Total Score: {total_score:.1f}/{total_max_score:.1f}",
        f"**Overall Coverage:** {overall_percentage}%",
        "",
    ]

    for item in results:
        evaluation = item["evaluation"]
        matched_lines = [f"- {concept}" for concept in evaluation.get("matched_concepts", [])]
        partial_lines = [f"- {concept}" for concept in evaluation.get("partial_concepts", [])]
        missing_lines = [f"- {concept}" for concept in evaluation.get("missing_concepts", [])]

        report_lines.extend([
            f"### Question {item['question_number']}",
            item["question"],
            "",
            f"**Answer:** {item['answer']}",
            f"**Score:** {evaluation['score_display']} | **Coverage:** {evaluation['coverage_percent']}%",
            "",
            "**Fully Covered**",
        ])
        report_lines.extend(matched_lines if matched_lines else ["- None"])
        report_lines.extend([
            "",
            "**Partial Credit**",
        ])
        report_lines.extend(partial_lines if partial_lines else ["- None"])
        report_lines.extend([
            "",
            "**Missing**",
        ])
        report_lines.extend(missing_lines if missing_lines else ["- None"])
        report_lines.extend([
            "",
            "**Feedback**",
            feedback_to_markdown(item["feedback"]),
            "",
        ])

    report_lines.extend([
        "## Overall Feedback",
        evaluation_bundle["overall_feedback"],
    ])

    full_report = "\n".join(report_lines)
    st.download_button(
        "⬇️ Download Full Report",
        data=full_report,
        file_name="exam_evaluation_report.md",
        mime="text/markdown",
    )
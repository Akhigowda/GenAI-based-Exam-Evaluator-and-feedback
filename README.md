# 🎓 AI Exam Answer Evaluator & Improvement Coach

A RAG-powered system that evaluates student answers against uploaded syllabus documents.

## 📁 Project Structure

```
ai_exam_evaluator/
│
├── app.py                      # Streamlit entry point (UI)
│
├── modules/
│   ├── document_processor.py   # Phase 2: Text extraction from PDF/PPT/TXT
│   ├── chunker.py              # Phase 3: Text chunking
│   ├── embedder.py             # Phase 4: Embedding generation
│   ├── vector_store.py         # Phase 5: FAISS vector DB
│   ├── retriever.py            # Phase 6: RAG retrieval
│   ├── evaluator.py            # Phase 7: Scoring logic
│   └── feedback_generator.py  # Phase 8: LLM feedback
│
├── utils/
│   ├── ocr_utils.py            # OCR for image/handwritten answers
│   └── file_utils.py           # File handling helpers
│
├── data/
│   └── .gitkeep               # Stores FAISS index & temp files
│
├── tests/
│   └── test_modules.py        # Unit tests per module
│
├── requirements.txt
└── .env.example
```

## 🚀 Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key (optional, for LLM feedback)
cp .env.example .env
# Edit .env and add OPENAI_API_KEY=sk-...

# 4. Run the app
streamlit run app.py
```

## 🔄 Workflow
1. Upload syllabus (PDF/PPT/TXT)
2. System builds knowledge base (chunks → embeddings → FAISS)
3. Enter exam question
4. Enter/upload student answer
5. Click Evaluate → Get Score + Feedback

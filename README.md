# 🧠 Resume Skill Gap Analyzer

An AI-powered Streamlit app that matches your resume against any job description in seconds. It combines a deterministic skill-taxonomy matcher with a fine-tuned Cross-Encoder for semantic similarity, then uses Google Gemini to turn the gaps into concrete, actionable career advice.

---

## ✨ Key Features

* 📄 **PDF Resume Parsing:** Extracts clean text from uploaded PDF resumes using `pdfplumber`.
* 🎯 **Taxonomy-Based Skill Extraction:** Matches resume and JD text against a curated taxonomy (`skills.py`) spanning Programming Languages, Databases, ML/DL, NLP & LLM tooling, Cloud/MLOps, Web Frameworks, and Soft Skills — using word-boundary safe regex so partial-word false positives (e.g. "js" inside another word) are avoided.
* 🤖 **Fine-Tuned Cross-Encoder Semantic Score:** Loads a custom fine-tuned Cross-Encoder (`./finetuned_cross_encoder`) if available, falling back to `cross-encoder/stsb-distilroberta-base` otherwise, to score deep semantic fit beyond keyword overlap.
* ⚖️ **Weighted Final Score:** Blends Exact Skill Match (50%) and AI Semantic Score (50%) into one final compatibility percentage.
* 💡 **Gemini-Powered Career Coaching:** Sends the resume + JD to `gemini-2.0-flash` and returns 5 specific, actionable recommendations — missing-skill learning paths, a portfolio project idea, domain gaps, and ATS keyword fixes.
* 📊 **Interactive Visual Dashboard:** Plotly gauge charts for Overall, Exact, and Semantic scores, plus tabbed breakdowns of Matched ✅, Missing ❌, and Extra ⭐ skills.

---

## 🛠️ Architecture & Technical Decisions

* **Taxonomy Matching over NER:** Skill extraction (`logic.py`) pre-compiles word-boundary-safe regex patterns for every alias in `SKILL_TAXONOMY` at startup, rather than relying on a trained NER model. This is deterministic, fast, and avoids false positives like matching "R" inside "Server."
* **Hybrid Scoring:** Exact taxonomy overlap and Cross-Encoder semantic similarity are combined 50/50 in `compute_match_score()`, balancing hard keyword requirements with contextual fit.
* **Custom Fine-Tuning Pipeline:** `train.py` fine-tunes `cross-encoder/stsb-roberta-base` on the [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) dataset using `sentence-transformers`, with a binary classification evaluator, and saves the result to `./finetuned_cross_encoder` for `logic.py` to auto-load.
* **Cached Resources:** The Sentence-Transformer/Cross-Encoder model and NLTK downloads are wrapped in `@st.cache_resource` so they load only once per session.

---

## 📁 Project Structure

resume-skill-gap-analyzer/
├── app.py # Streamlit UI: upload, JD input, results dashboard, gauges
├── logic.py # PDF extraction, skill matching, scoring, Gemini recommendations
├── skills.py # SKILL_TAXONOMY: canonical skills mapped to their aliases
└── train.py # Fine-tunes the Cross-Encoder on resume/JD fit dataset

---

## 📦 Requirements

```text
streamlit
sentence-transformers
pdfplumber
google-generativeai
plotly
nltk
torch
datasets
numpy
```

---

## 🚀 Installation & Setup

1. **Clone the repository:**

```bash
git clone https://github.com/Ali-Muhammad24/resume-skill-gap-analyzer.git
cd resume-skill-gap-analyzer
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Configure your Gemini API Key:**

Create a `.streamlit/secrets.toml` file in the project root:

```toml
GEMINI_API_KEY = "your_google_gemini_api_key_here"
```

4. **(Optional) Fine-tune the Cross-Encoder:**

```bash
python train.py
```

This trains on the `cnamuangtoun/resume-job-description-fit` dataset and saves the model to `./finetuned_cross_encoder`. If skipped, the app automatically falls back to the pre-trained `cross-encoder/stsb-distilroberta-base` model.

5. **Run the app:**

```bash
streamlit run app.py
```

---

## 🧭 How It Works

1. Upload a PDF resume and paste a job description (min. 50 characters).
2. `extract_text_from_pdf()` pulls raw text from the PDF.
3. `extract_skills()` scans both texts against the taxonomy to find matched, missing, and extra skills.
4. The Cross-Encoder scores semantic similarity between resume and JD.
5. `compute_match_score()` combines both into exact, semantic, and final scores.
6. If a `GEMINI_API_KEY` is present, Gemini generates 5 tailored recommendations.
7. Results render as gauges, metrics, and skill-breakdown tabs.

---

## 💻 Tech Stack

* **Frontend & UI:** Streamlit, Plotly
* **NLP & ML:** PyTorch, Sentence-Transformers (Cross-Encoder), NLTK, pdfplumber
* **LLM Integration:** Google Gemini 2.0 Flash API
* **Language:** Python 3.10+

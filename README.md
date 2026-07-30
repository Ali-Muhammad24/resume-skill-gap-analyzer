# 🧠 SkillSync AI

An AI-powered Streamlit application that matches resumes against job descriptions using a hybrid approach combining deterministic skill matching, a fine-tuned Cross-Encoder, and Groq's **Llama 3.3 70B** to generate actionable insights for both job seekers and recruiters.

---

## 📖 Overview

SkillSync AI offers two modes:

### 👤 Job Seeker Mode
- Upload a resume and paste a job description.
- Get a compatibility score, matched/missing/extra skills, and AI-powered career recommendations.

### 🏢 Recruiter Mode
- Upload up to **25 resumes** against a single job description.
- Receive ranked candidates, aggregate skill-gap analysis, AI hiring summaries, and a downloadable PDF report.

Both modes use the same hybrid scoring engine for consistent evaluation.

---

## ✨ Features

- 📄 **PDF Resume Parsing** using **pdfplumber**.
- 🎯 **Taxonomy-Based Skill Extraction** with regex-based matching across Programming, ML/DL, NLP, Cloud, Databases, Web, and Soft Skills.
- 🤖 **Fine-Tuned Cross-Encoder** with automatic fallback to `cross-encoder/stsb-distilroberta-base`.
- 📊 **Hybrid Scoring:** 50% Exact Skill Match + 50% Semantic Similarity.
- 📁 **Batch Resume Screening** with candidate ranking and progress tracking.
- 📈 **Aggregate Skill Gap Analysis** across all candidates.
- 🧠 **Groq Llama 3.3 70B** for career recommendations and recruiter summaries.
- 📑 **PDF Screening Report** generation.
- 📉 **Interactive Plotly Dashboard** with gauges and skill breakdowns.

---

## 🏗️ Architecture

- **Deterministic Skill Matching:** Uses a curated skill taxonomy with word-boundary-safe regex instead of NER for faster, more reliable extraction.
- **Hybrid Matching Engine:** Combines exact skill overlap with semantic similarity using a fine-tuned Cross-Encoder.
- **Custom Fine-Tuning:** `train.py` fine-tunes `cross-encoder/stsb-roberta-base` on the `cnamuangtoun/resume-job-description-fit` dataset and saves the model to `./finetuned_cross_encoder`.
- **Optimized Performance:** Batch processing, cached models (`@st.cache_resource`), and fault-tolerant PDF generation.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Plotly
- **NLP & ML:** PyTorch, Sentence-Transformers, NLTK, pdfplumber
- **LLM:** Groq API (Llama 3.3 70B)
- **PDF:** fpdf2
- **Language:** Python 3.10+

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Ali-Muhammad24/resume-skill-gap-analyzer.git
cd skillsync-ai
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure Groq API

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### (Optional) Fine-Tune the Model

```bash
python train.py
```

If skipped, the app automatically uses the pre-trained `cross-encoder/stsb-distilroberta-base` model.

### Run the application

```bash
streamlit run app.py
```

---

## ⚙️ Workflow

### 👤 Job Seeker
1. Upload resume and paste a job description.
2. Extract text and identify matched, missing, and extra skills.
3. Compute exact and semantic scores.
4. Generate AI-powered career recommendations.
5. Display results through charts and skill panels.

### 🏢 Recruiter
1. Upload a job description and multiple resumes.
2. Rank candidates using the hybrid scoring engine.
3. Identify common skill gaps.
4. Generate AI hiring summaries.
5. Export a PDF screening report.

---

## 📁 Project Structure

```text
├── app.py           # Streamlit interface
├── logic.py         # Matching engine & AI integration
├── skills.py        # Skill taxonomy
├── train.py         # Cross-Encoder fine-tuning
└── requirements.txt
```

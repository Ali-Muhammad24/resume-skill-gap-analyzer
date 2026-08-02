# 🧠 SkillSync AI

An AI-powered Streamlit application that matches resumes against job descriptions using a hybrid approach combining deterministic skill matching, a fine-tuned Cross-Encoder, and Groq's Llama 3.3 70B.

## 🚀 Live Demo

**Try it here:** https://skillsync-ai-demo.streamlit.app

---

## 📖 Overview

SkillSync AI offers two modes:

### 👤 Job Seeker Mode
- Upload a resume and paste a job description.
- Get a compatibility score, matched/missing/extra skills, and AI-powered career recommendations.

### 🏢 Recruiter Mode
- Upload up to **25 resumes** against a single job description.
- Receive ranked candidates, AI hiring summaries, and a downloadable PDF report.

Both modes use the same hybrid scoring engine for consistent evaluation.

---

## ✨ Features

- 📄 **PDF Resume Parsing** using **pdfplumber**.
- 🎯 **Taxonomy-Based Skill Extraction** with regex-based matching across Programming, ML/DL, NLP, Cloud, Databases, Web, and Soft Skills.
- 🤖 **Fine-Tuned Cross-Encoder**, hosted on Hugging Face Hub, with automatic fallback to `cross-encoder/stsb-distilroberta-base`.
- 📊 **Hybrid Scoring:** 50% Exact Skill Match + 50% Semantic Similarity.
- 📁 **Batch Resume Screening** with candidate ranking and progress tracking.
- 🧠 **Groq Llama 3.3 70B** for career recommendations and recruiter summaries.
- 📑 **PDF Screening Report** generation.
- 📉 **Interactive Plotly Dashboard** with gauges and skill breakdowns.

---

## 🏗️ Architecture

- **Deterministic Skill Matching:** Uses a curated skill taxonomy with word-boundary-safe regex instead of NER for faster, more reliable extraction.
- **Hybrid Matching Engine:** Combines exact skill overlap with semantic similarity using a fine-tuned Cross-Encoder.
- **Custom Fine-Tuning:** `train.py` fine-tunes `cross-encoder/stsb-roberta-base` on the `cnamuangtoun/resume-job-description-fit` dataset.
- **Model Hosting:** The fine-tuned Cross-Encoder is published on Hugging Face Hub and loaded directly at runtime via its repo ID—no large model files are stored in this repository. If the Hub load fails for any reason, the app automatically falls back to the pre-trained `cross-encoder/stsb-distilroberta-base` model, so the app never breaks.
- **Optimized Performance:** Batch processing, cached models (`@st.cache_resource`), and fault-tolerant PDF generation.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Plotly
- **NLP & ML:** PyTorch, Sentence-Transformers, NLTK, pdfplumber
- **LLM:** Groq API (Llama 3.3 70B)
- **Model Hosting:** Hugging Face Hub
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

> **Note:** The deployed app loads the fine-tuned Cross-Encoder directly from Hugging Face Hub at runtime—running `train.py` locally is **not required** to use the app. It's only needed if you want to reproduce the fine-tuning yourself or experiment with improving the model. If the Hub model can't be reached, the app automatically falls back to the pre-trained `cross-encoder/stsb-distilroberta-base` model.

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
3. Generate AI hiring summaries.
4. Export a PDF screening report.

---

## 📁 Project Structure

```text
├── app.py           # Streamlit interface
├── logic.py         # Matching engine & AI integration
├── skills.py        # Skill taxonomy
├── train.py         # Cross-Encoder fine-tuning (optional, for reproducing the HF-hosted model)
└── requirements.txt
```

---

## 🤗 Model

The fine-tuned Cross-Encoder used in production is publicly available on Hugging Face Hub:

https://huggingface.co/alimuhammad24/resume-jd-cross-encoder
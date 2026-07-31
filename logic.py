# Importing Libraries 
from pdfminer import pdfdocument
import re
import os
from collections import Counter

# For PDF text extraction
import pdfplumber

# Cross-Encoder for semantic matching
from sentence_transformers import CrossEncoder

# For LLM-based recommendations
from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile"

# For PDF report generation (recruiter export)
from fpdf import FPDF
# Text position (x, y) 
from fpdf.enums import XPos, YPos 

# Skills DB from skills.py
from skills import SKILL_TAXONOMY

# Load fine-tuned Cross-Encoder from Hugging Face with fallback to the pre-trained base model
HF_MODEL_REPO = "alimuhammad24/resume-jd-cross-encoder"
FALLBACK_CROSS_ENCODER = "cross-encoder/stsb-distilroberta-base"

try:
    cross_model = CrossEncoder(HF_MODEL_REPO)
except Exception:
    cross_model = CrossEncoder(FALLBACK_CROSS_ENCODER)

# Pre-compiles word-boundary safe regex patterns for fast skill matching at startup.
_COMPILED_PATTERNS = {}
for _default_skill_name, _aliases in SKILL_TAXONOMY.items():
    _all_terms = set(_aliases) | {_default_skill_name}
    _patterns = []
    for _term in _all_terms:
        _term_norm = _term.strip().lower()
        if not _term_norm:
            continue

        _pattern = re.compile(r'(?<![a-z0-9])' + re.escape(_term_norm) + r'(?![a-z0-9])')
        _patterns.append(_pattern)
    _COMPILED_PATTERNS[_default_skill_name] = _patterns

# Normalize text 
def _normalize(text):
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    return f" {cleaned} "

# Skill taxonomy matcher
def extract_skills(text):
    if not text or not text.strip():
        return set()

    text_norm = _normalize(text)
    found = set()

    for default_skill, patterns in _COMPILED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text_norm):
                found.add(default_skill)
                break  

    return found

# Categorize Skills
def categorize_skills(skills):
    if skills:
        return {"Extracted Skills": sorted(s.lower() for s in skills)}
    return {}

# Extract PDF text
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        return ""
    return text.strip()

# Match Score 
def compute_match_score(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    # 1. Exact Match 
    matching = resume_skills & jd_skills
    missing  = jd_skills - resume_skills
    extra    = resume_skills - jd_skills

    exact_score = (len(matching) / len(jd_skills) * 100) if jd_skills else 0.0

    # 2. Cross-Encoder semantic score
    try:
        pair = (resume_text[:2000], jd_text[:2000])
        raw  = cross_model.predict(pair)
        semantic_score = float(max(0, min(100, raw * 100)))
    except Exception:
        semantic_score = 0.0

    # 3. Final Weighted Score
    final_score = 0.50 * exact_score + 0.50 * semantic_score

    # 4. Structured match results
    return {
        "matching_skills":  sorted(matching),
        "missing_skills":   sorted(missing),
        "extra_skills":     sorted(extra),
        "resume_skills":    sorted(resume_skills),
        "jd_skills":        sorted(jd_skills),
        "exact_score":      round(exact_score,    1),
        "semantic_score":   round(semantic_score, 1),
        "final_score":      round(final_score,    1),
        "total_jd_skills":  len(jd_skills),
        "matched_count":    len(matching),
        "missing_count":    len(missing),
    }

# AI recommendations for Job Seeker
def get_ai_recommendations(resume_text, jd_text, api_key):
    try:
        client = Groq(api_key=api_key)

        prompt = f"""
        You are an expert Career Coach and ATS Optimization Specialist.
        Analyze this Resume against the Job Description below.

        RESUME (first 2500 chars):
        {resume_text[:2500]}

        JOB DESCRIPTION (first 2500 chars):
        {jd_text[:2500]}

        Give 5 high-impact, specific, actionable recommendations to improve this resume's match.
        Focus on:
        1. How to add/learn the missing skills (courses, certs, or project ideas)
        2. One concrete portfolio project idea that bridges the skill gap
        3. Any domain knowledge gaps you notice
        4. ATS keyword improvements

        Format: Clean bullet points without emojis. Be specific, not generic.
        """

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"


# ---------- Recruiter Mode ----------

# Number of resumes to screen at once
MAX_BATCH_SIZE = 25

# Batch resume matching
def batch_match(resume_files, jd_text, progress_callback=None):
    # Enforce batch limit
    resume_files = resume_files[:MAX_BATCH_SIZE]
    total = len(resume_files)
    results = []

    for idx, f in enumerate(resume_files, 1):
        text = extract_text_from_pdf(f)
        if not text or len(text) < 100:
            results.append({
                "filename": f.name,
                "error": "Could not extract text (scanned/image-only PDF?)"
            })
        else:
            scores = compute_match_score(text, jd_text)
            scores["filename"] = f.name
            scores["resume_text"] = text  # kept for optional LLM explanation later
            results.append(scores)

        if progress_callback:
            progress_callback(idx, total, f.name)

    # Valid results sorted by final_score descending
    valid = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]
    valid.sort(key=lambda x: x["final_score"], reverse=True)

    return valid + errored

# Combine missing skills across all resumes
def aggregate_missing_skills(batch_results, top_n=5):
    all_missing = []
    for r in batch_results:
        if "error" not in r:
            all_missing.extend(r["missing_skills"])

    counter = Counter(all_missing)
    return counter.most_common(top_n)

# AI recommendations for Recruiter
def get_recruiter_summary(resume_text, jd_text, api_key):
    try:
        client = Groq(api_key=api_key)

        prompt = f"""
        You are an expert Technical Recruiter assisting a hiring manager.
        Evaluate this candidate's resume against the job description below,
        from the employer's perspective — NOT the candidate's.

        RESUME (first 2500 chars):
        {resume_text[:2500]}

        JOB DESCRIPTION (first 2500 chars):
        {jd_text[:2500]}

        Give a concise hiring-focused summary covering:
        1. Overall fit verdict (Strong Fit / Possible Fit / Weak Fit) with one-line reasoning
        2. Key strengths relevant to this specific role
        3. Notable skill or experience gaps that could be a risk
        4. 2-3 targeted interview questions to probe the gaps or verify claimed strengths

        Format: Clean bullet points, no fluff, written for a hiring manager who has
        limited time and is scanning many candidates.
        """

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

# PDF Report Export for Recruiter
def _pdf_safe(text, max_word_len=35):
    if text is None:
        return ""
    text = str(text)

    text = text.encode("latin-1", errors="ignore").decode("latin-1")

    words = text.split(" ")
    safe_words = []
    for w in words:
        if len(w) > max_word_len:
            chunks = [w[i:i + max_word_len] for i in range(0, len(w), max_word_len)]
            safe_words.append(" ".join(chunks))
        else:
            safe_words.append(w)
    return " ".join(safe_words)

# Recruiter PDF report
def generate_recruiter_report(batch_results, jd_text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def _write_multiline(text, font=("Arial", "", 10), line_height=6, max_word_len=35):
        # Reset left alignment
        pdf.set_x(pdf.l_margin)
        pdf.set_font(*font)
        pdf.multi_cell(0, line_height, _pdf_safe(text, max_word_len=max_word_len))

    def _write_line(text, font=("Arial", "", 10)):
        # Again Reset left alignment
        pdf.set_x(pdf.l_margin)
        pdf.set_font(*font)
        pdf.cell(0, 6, _pdf_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    _write_line("Candidate Screening Report", font=("Arial", "B", 16))

    jd_excerpt = jd_text[:300].strip()
    _write_multiline(f"Job Description (excerpt): {jd_excerpt}...", font=("Arial", "", 9), line_height=5)
    pdf.ln(4)

    _write_line(f"Total candidates screened: {len(batch_results)}")
    pdf.ln(4)

    # loop for each candidate
    for i, r in enumerate(batch_results, 1):
        try:
            # Candidate header
            _write_multiline(
                f"#{i}  {r['filename']}  -  Overall Match: {r['final_score']}%",
                font=("Arial", "B", 12), line_height=8, max_word_len=40
            )

            # Sub-scores
            _write_line(
                f"Exact Skill Match: {r['exact_score']}%   |   AI Semantic Score: {r['semantic_score']}%",
                font=("Arial", "I", 9)
            )

            # Skill breakdown
            matched = ", ".join(r["matching_skills"]) or "None"
            missing = ", ".join(r["missing_skills"]) or "None"
            extra = ", ".join(r["extra_skills"]) or "None"

            _write_multiline(f"Matched Skills: {matched}")
            _write_multiline(f"Missing Skills: {missing}")
            _write_multiline(f"Extra Skills: {extra}")
            pdf.ln(4)

        except Exception as e:
            # skip broken candidate data safely without throwing an exception
            _write_line(f"[Could not render full details for {r.get('filename', 'this candidate')}]")
            pdf.ln(4)

    # fpdf2 returns bytearray with dest="S"; ensure plain bytes for Streamlit
    return bytes(pdf.output(dest="S"))
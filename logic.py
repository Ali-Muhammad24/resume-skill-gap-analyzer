# Importing Libraries 
import re
import os

# For PDF text extraction
import pdfplumber

# Cross-Encoder for semantic matching
from sentence_transformers import CrossEncoder

# For LLM-based recommendations
import google.generativeai as genai

# Skills DB from skills.py
from skills import SKILL_TAXONOMY

# Model Loading (from train.py) with fallback to pre-trained base
CROSS_ENCODER_PATH = "./finetuned_cross_encoder"
FALLBACK_CROSS_ENCODER = "cross-encoder/stsb-distilroberta-base"

if os.path.exists(CROSS_ENCODER_PATH):
    cross_model = CrossEncoder(CROSS_ENCODER_PATH)
else:
    cross_model = CrossEncoder(FALLBACK_CROSS_ENCODER)

# Pre-compiles word-boundary safe regex patterns for fast skill matching at startup.
_COMPILED_PATTERNS = {}
for _canonical, _aliases in SKILL_TAXONOMY.items():
    _all_terms = set(_aliases) | {_canonical}
    _patterns = []
    for _term in _all_terms:
        _term_norm = _term.strip().lower()
        if not _term_norm:
            continue
        
        _pattern = re.compile(r'(?<![a-z0-9])' + re.escape(_term_norm) + r'(?![a-z0-9])')
        _patterns.append(_pattern)
    _COMPILED_PATTERNS[_canonical] = _patterns

# Normalize text 
def _normalize(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    return f" {cleaned} "

# Skill taxonomy matcher
def extract_skills(text: str) -> set:
    if not text or not text.strip():
        return set()

    text_norm = _normalize(text)
    found = set()

    for canonical, patterns in _COMPILED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text_norm):
                found.add(canonical)
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

# AI recommendations 
def get_ai_recommendations(resume_text, jd_text, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

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
        3. One concrete portfolio project idea that bridges the skill gap
        4. Any domain knowledge gaps you notice
        5. ATS keyword improvements

        Format: Clean bullet points with relevant emojis. Be specific, not generic.
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

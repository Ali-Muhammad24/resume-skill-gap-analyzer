# Importing Libraries 
import streamlit as st
import pandas as pd
import nltk
import plotly.graph_objects as go

from logic import (
    extract_text_from_pdf,
    compute_match_score,
    get_ai_recommendations,
    get_recruiter_summary,
    batch_match,
    aggregate_missing_skills,
    generate_recruiter_report,
    MAX_BATCH_SIZE,
)

# Page Configuration
st.set_page_config(
    page_title="SkillSync AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Theme
BG = "#121315"
SURFACE = "#1c1d20"
BORDER = "#333438"
PRIMARY = "#f97316"
PRIMARY_LIGHT = "#fb923c"
TEXT = "#f2f2f0"
TEXT_MUTED = "#9a9a9e"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
DANGER = "#f87171"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* App background */
.stApp {{
    background: radial-gradient(circle at 15% 0%, #24201c 0%, {BG} 45%);
    color: {TEXT};
}}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {{visibility: hidden;}}
div[data-testid="stDecoration"] {{display: none;}}
div[data-testid="stToolbar"] {{display: none;}}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}}

/* Header */
.app-header {{
    text-align: center;
    margin-bottom: 2.25rem;
}}
.app-title {{
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: {TEXT};
    line-height: 1.2;
}}
.app-title span {{
    color: {PRIMARY_LIGHT};
}}
.app-subtitle {{
    font-size: 0.95rem;
    color: {TEXT_MUTED};
    margin-top: 0.35rem;
}}

/* Section labels used inside cards */
.section-label {{
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {PRIMARY_LIGHT};
    margin-bottom: 0.6rem;
}}

/* Bordered containers -> cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE};
    border: 1px solid {BORDER} !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    padding: 0.25rem;
}}

/* Tabs */
button[data-baseweb="tab"] {{
    font-weight: 600;
    color: {TEXT_MUTED};
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {PRIMARY_LIGHT};
}}
div[data-baseweb="tab-highlight"] {{
    background-color: {PRIMARY} !important;
}}
div[data-baseweb="tab-border"] {{
    background-color: {BORDER} !important;
}}

/* Buttons */
div[data-testid="stButton"] button {{
    background: linear-gradient(135deg, {PRIMARY} 0%, #6d28d9 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    transition: filter 0.15s ease;
}}
div[data-testid="stButton"] button:hover {{
    filter: brightness(1.12);
    color: white;
}}
div[data-testid="stDownloadButton"] button {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {PRIMARY};
    border-radius: 10px;
    font-weight: 600;
}}
div[data-testid="stDownloadButton"] button:hover {{
    border-color: {PRIMARY_LIGHT};
    color: {PRIMARY_LIGHT};
}}

/* Metrics */
div[data-testid="stMetric"] {{
    background: rgba(139, 92, 246, 0.07);
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 0.9rem 1rem;
}}
div[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED};
}}
div[data-testid="stMetricValue"] {{
    color: {TEXT};
}}

/* Text areas / file uploader */
textarea, div[data-testid="stFileUploaderDropzone"] {{
    background: #0f0d18 !important;
    border-radius: 10px !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
}}

/* Progress bar */
div[data-testid="stProgress"] > div > div {{
    background: linear-gradient(90deg, {PRIMARY} 0%, {PRIMARY_LIGHT} 100%);
}}

/* Dataframe */
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
}}

/* Alerts */
div[data-testid="stAlert"] {{
    border-radius: 10px;
}}

hr {{
    border-color: {BORDER};
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Cached Resource Loading
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
    except Exception:
        pass

@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

download_nltk_data()

# Gauge Chart
def plot_gauge(score, title):
    if score < 40:
        color = DANGER
    elif score < 70:
        color = WARNING
    else:
        color = SUCCESS

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"size": 13, "color": TEXT_MUTED}},
            number={"suffix": "%", "font": {"size": 26, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": BORDER},
                "bar": {"color": color},
                "bgcolor": SURFACE,
                "borderwidth": 1,
                "bordercolor": BORDER,
                "steps": [
                    {"range": [0, 40], "color": "#232324"},
                    {"range": [40, 70], "color": "#252220"},
                    {"range": [70, 100], "color": "#1f2420"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(t=35, b=10, l=25, r=25),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter"),
    )
    return fig

def card_title(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)

# Job Seeker Mode
def render_job_seeker_mode():
    with st.container(border=True):
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            card_title("Resume")
            uploaded_resume = st.file_uploader(
                "Upload PDF resume",
                type=["pdf"],
                help="Only PDF format is supported.",
                label_visibility="collapsed",
            )
            if uploaded_resume:
                st.success(f"Uploaded: **{uploaded_resume.name}**")

        with col_right:
            card_title("Job Description")
            job_description = st.text_area(
                "Job description",
                height=160,
                placeholder="Paste the full job description here...",
                label_visibility="collapsed",
            )
            jd_word_count = len(job_description.split()) if job_description.strip() else 0
            st.caption(f"Word count: {jd_word_count}")

        st.write("")
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            analyze_clicked = st.button("Analyze Resume", use_container_width=True)

    if not analyze_clicked:
        st.write("")
        st.info(
            "Upload your resume and paste a job description above, then click "
            "**Analyze Resume** to get your results."
        )
        return

    if not uploaded_resume:
        st.error("Please upload your resume PDF first.")
        st.stop()
    if not job_description.strip() or len(job_description.strip()) < 50:
        st.error("Please paste a job description (at least 50 characters).")
        st.stop()

    with st.spinner("Extracting skills and analyzing your resume... this may take 20–40 seconds on first run."):
        resume_text = extract_text_from_pdf(uploaded_resume)
        if not resume_text or len(resume_text) < 100:
            st.error("Could not extract text from the PDF. Make sure it's not a scanned image-only file.")
            st.stop()

        load_model()
        results = compute_match_score(resume_text, job_description)

    st.write("")
    with st.container(border=True):
        card_title("Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Match", f"{results['final_score']}%")
        m2.metric("Skills Matched", results["matched_count"])
        m3.metric("Skills Missing", results["missing_count"])
        m4.metric("Bonus Skills", len(results["extra_skills"]))

        st.write("")
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.plotly_chart(
                plot_gauge(results["final_score"], "Overall Match"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with gc2:
            st.plotly_chart(
                plot_gauge(results["exact_score"], "Exact Skill Match"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with gc3:
            st.plotly_chart(
                plot_gauge(results["semantic_score"], "AI Semantic Score"),
                use_container_width=True, config={"displayModeBar": False},
            )

    st.write("")
    with st.container(border=True):
        card_title("Skill Breakdown")
        tab1, tab2, tab3 = st.tabs(["Matched", "Missing", "Extra"])

        with tab1:
            if results["matching_skills"]:
                st.write(" · ".join(results["matching_skills"]))
                st.progress(min(results["matched_count"] / max(results["total_jd_skills"], 1), 1.0))
                st.caption(f"{results['matched_count']} JD skills found in your resume")
            else:
                st.warning("No skill overlaps detected. Try expanding your skills section in the resume.")

        with tab2:
            if results["missing_skills"]:
                st.write(" · ".join(results["missing_skills"]))
                st.caption("These skills are required by the JD but not found in your resume.")
            else:
                st.success("You have all the skills mentioned in the job description!")

        with tab3:
            if results["extra_skills"]:
                st.write(" · ".join(results["extra_skills"]))
                st.caption("Skills you have beyond what the JD requires — great for standing out.")
            else:
                st.info("No extra skills beyond the JD requirements detected.")

    st.write("")
    with st.container(border=True):
        card_title("AI-Powered Career Advice")
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
            with st.spinner("Generating personalized recommendations..."):
                ai_advice = get_ai_recommendations(resume_text, job_description, api_key)
                st.markdown(ai_advice)
        else:
            st.warning("API key missing. Add GROQ_API_KEY to your secrets.toml to enable this feature.")

# Recruiter Mode
def render_recruiter_mode():
    with st.container(border=True):
        card_title("Job Description")
        jd_text = st.text_area(
            "Job description",
            height=160,
            key="recruiter_jd",
            placeholder="Paste the job description you're screening candidates against...",
            label_visibility="collapsed",
        )
        jd_word_count = len(jd_text.split()) if jd_text.strip() else 0
        st.caption(f"Word count: {jd_word_count}")

        st.write("")
        card_title("Candidate Resumes")
        resume_files = st.file_uploader(
            "Upload multiple PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload as many resumes as you want to screen against the JD above.",
            label_visibility="collapsed",
        )
        if resume_files:
            if len(resume_files) > MAX_BATCH_SIZE:
                st.warning(
                    f"You uploaded {len(resume_files)} resumes, but this app screens a maximum of "
                    f"{MAX_BATCH_SIZE} at a time. Only the first {MAX_BATCH_SIZE} will be processed."
                )
            else:
                st.success(f"{len(resume_files)} resume(s) uploaded")

        st.write("")
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            screen_clicked = st.button("Screen Candidates", use_container_width=True)

    if screen_clicked:
        if not jd_text.strip() or len(jd_text.strip()) < 50:
            st.error("Please paste a job description (at least 50 characters).")
            st.stop()
        if not resume_files:
            st.error("Please upload at least one resume.")
            st.stop()

        load_model()

        progress_bar = st.progress(0, text="Starting screening...")
        status_text = st.empty()

        def _update_progress(current, total, filename):
            fraction = current / total if total else 1.0
            progress_bar.progress(fraction, text=f"Screening {current}/{total}")
            status_text.caption(f"Processing: {filename}")

        batch_results = batch_match(resume_files, jd_text, progress_callback=_update_progress)

        progress_bar.empty()
        status_text.empty()
        st.success(f"Screening complete — {len(resume_files[:MAX_BATCH_SIZE])} resume(s) processed.")

        st.session_state["batch_results"] = batch_results
        st.session_state["recruiter_jd_text"] = jd_text

    if "batch_results" not in st.session_state:
        st.write("")
        st.info(
            "Paste a job description and upload multiple resumes above, then click "
            "**Screen Candidates** to get a ranked shortlist."
        )
        return

    batch_results = st.session_state["batch_results"]
    jd_text_saved = st.session_state.get("recruiter_jd_text", "")
    valid_results = [r for r in batch_results if "error" not in r]
    errored_results = [r for r in batch_results if "error" in r]

    st.write("")
    with st.container(border=True):
        card_title(f"Ranked Results — {len(valid_results)} screened, {len(errored_results)} failed")

        table_data = [
            {
                "Rank": i,
                "Candidate": r["filename"],
                "Overall %": r["final_score"],
                "Exact %": r["exact_score"],
                "Semantic %": r["semantic_score"],
                "Matched": r["matched_count"],
                "Missing": r["missing_count"],
            }
            for i, r in enumerate(valid_results, 1)
        ]
        if table_data:
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
        else:
            st.warning("No resumes could be successfully processed.")

        if errored_results:
            with st.expander(f"{len(errored_results)} resume(s) failed to process"):
                for r in errored_results:
                    st.write(f"- **{r['filename']}**: {r['error']}")

    if valid_results:
        st.write("")
        with st.container(border=True):
            card_title("Top 5 Most Commonly Missing Skills")
            agg = aggregate_missing_skills(valid_results, top_n=5)
            if agg:
                for skill, count in agg:
                    st.write(f"- **{skill}** — missing in {count}/{len(valid_results)} candidates")
                st.caption(
                    "If a skill is missing across most candidates, consider whether the JD "
                    "requirement is realistic for your talent pool."
                )
            else:
                st.info("No common missing skills detected — candidates broadly cover the JD requirements.")

        st.write("")
        with st.container(border=True):
            card_title("Candidate Details")

            options = [f"#{i} — {r['filename']} ({r['final_score']}%)" for i, r in enumerate(valid_results, 1)]
            selected = st.selectbox("Choose a candidate", options, label_visibility="collapsed")
            r = valid_results[options.index(selected)]

            m1, m2, m3 = st.columns(3)
            m1.metric("Overall", f"{r['final_score']}%")
            m2.metric("Exact", f"{r['exact_score']}%")
            m3.metric("Semantic", f"{r['semantic_score']}%")

            st.write("")
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("**Matched**")
                st.caption(", ".join(r["matching_skills"]) or "None")
            with d2:
                st.markdown("**Missing**")
                st.caption(", ".join(r["missing_skills"]) or "None")
            with d3:
                st.markdown("**Extra**")
                st.caption(", ".join(r["extra_skills"]) or "None")

        st.write("")
        with st.container(border=True):
            card_title("AI Hiring Insights (Top 5 Candidates)")
            if st.button("Generate AI Explanation for Top 5"):
                if "GROQ_API_KEY" in st.secrets:
                    api_key = st.secrets["GROQ_API_KEY"]
                    for r in valid_results[:5]:
                        with st.spinner(f"Analyzing {r['filename']}..."):
                            advice = get_recruiter_summary(r["resume_text"], jd_text_saved, api_key)
                        with st.expander(f"AI Insight — {r['filename']}"):
                            st.markdown(advice)
                else:
                    st.warning("API key missing. Add GROQ_API_KEY to your secrets.toml to enable this feature.")

            st.write("")
            pdf_bytes = generate_recruiter_report(valid_results, jd_text_saved)
            st.download_button(
                "Download Screening Report (PDF)",
                data=pdf_bytes,
                file_name="candidate_screening_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# Header + Mode Tabs
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">Skill<span>Sync</span> AI</div>
        <div class="app-subtitle">AI-powered matching between resumes and job descriptions</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_seeker, tab_recruiter = st.tabs(["Job Seeker", "Recruiter · Batch Screening"])

with tab_seeker:
    render_job_seeker_mode()

with tab_recruiter:
    render_recruiter_mode()

st.write("")
st.caption("© 2026 SkillSync AI. All rights reserved.")
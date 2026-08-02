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

# Theme Palette
BG = "#F8FAFC"
SECONDARY_BG = "#F1F5F9"
SURFACE = "#FFFFFF"

PRIMARY = "#5B7DB1"
PRIMARY_HOVER = "#4E6F9D"
PRIMARY_ACTIVE = "#45658F"
LIGHT_ACCENT = "#EAF1F8"

TEXT = "#1F2937"
TEXT_SECONDARY = "#6B7280"
TEXT_MUTED = "#9CA3AF"

BORDER = "#A9BFDA"
BORDER_LIGHT = "#DCE5F0"

SUCCESS = "#4CAF7D"
WARNING = "#D4A017"
DANGER = "#D95D5D"

PILL_GREEN_BG, PILL_GREEN_TEXT = "#DCFCE7", "#15803D"
PILL_RED_BG, PILL_RED_TEXT = "#FEE2E2", "#B91C1C"
PILL_BLUE_BG, PILL_BLUE_TEXT = "#DBEAFE", "#1D4ED8"

#Styling
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}

/* App background */
.stApp {{
    background: {BG};
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
    color: {PRIMARY};
}}
.app-subtitle {{
    font-size: 0.95rem;
    color: {TEXT_SECONDARY};
    margin-top: 0.35rem;
}}

/* Section labels used inside cards */
.section-label {{
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {PRIMARY};
    margin-bottom: 0.6rem;
}}

/* Bordered containers -> cards */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"] > div,
div[data-testid="stBorderWrapper"] {{
    background: {SURFACE} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    padding: 0.25rem;
}}

/* Tabs */
button[data-baseweb="tab"] {{
    font-weight: 600;
    color: {TEXT_SECONDARY};
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {PRIMARY};
}}
div[data-baseweb="tab-highlight"] {{
    background-color: {PRIMARY} !important;
}}
div[data-baseweb="tab-border"] {{
    background-color: {BORDER} !important;
}}

/* Buttons */
div[data-testid="stButton"] button,
div[data-testid="stButton"] button p,
div[data-testid="stButton"] button span,
div[data-testid="stButton"] button div {{
    color: #FFFFFF !important;
}}
div[data-testid="stButton"] button {{
    background: {PRIMARY} !important;
    border: none !important;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    transition: background-color 0.15s ease;
}}
div[data-testid="stButton"] button:hover {{
    background: {PRIMARY_HOVER} !important;
}}
div[data-testid="stButton"] button:active,
div[data-testid="stButton"] button:focus,
div[data-testid="stButton"] button:focus:not(:active) {{
    background: {PRIMARY_ACTIVE} !important;
    border-color: {PRIMARY_ACTIVE} !important;
    box-shadow: none !important;
}}

div[data-testid="stDownloadButton"] button {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.15s ease;
}}
div[data-testid="stDownloadButton"] button:hover {{
    border-color: {PRIMARY};
    color: {PRIMARY};
    background: {LIGHT_ACCENT};
}}

/* Metrics */
div[data-testid="stMetric"] {{
    background: {LIGHT_ACCENT};
    border: 1px solid {BORDER_LIGHT};
    border-radius: 10px;
    padding: 0.9rem 1rem;
}}
div[data-testid="stMetricLabel"] {{
    color: {TEXT_SECONDARY};
}}
div[data-testid="stMetricValue"] {{
    color: {TEXT};
}}

/* Text areas / file uploader */
textarea, div[data-testid="stFileUploaderDropzone"] {{
    background: {SECONDARY_BG} !important;
    border-radius: 8px !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
}}
textarea::placeholder, input::placeholder, ::placeholder {{
    color: {TEXT_SECONDARY} !important;
    opacity: 1 !important;
}}
textarea:focus, div[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {PRIMARY} !important;
}}

/* Dropdowns */
div[data-baseweb="select"] > div {{
    background: {SECONDARY_BG} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 8px !important;
}}

/* Progress bar */
div[data-testid="stProgress"] > div > div {{
    background: {PRIMARY};
}}

/* Dataframe */
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
}}

/* Alerts */
div[data-testid="stAlert"] {{
    background-color: {LIGHT_ACCENT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
}}
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div {{
    color: {TEXT} !important;
}}
div[data-testid="stAlert"] svg {{
    color: {PRIMARY} !important;
    fill: {PRIMARY} !important;
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

# Score color helper
def get_score_color(score):
    if score < 40:
        return DANGER
    elif score < 70:
        return WARNING
    else:
        return SUCCESS

# Combined Score Comparison Chart (concentric radial gauge / donut rings)
def plot_score_comparison(overall, exact, semantic):
    # (label, value, domain padding, hole size)
    rings = [
        ("Overall Match", overall, 0.00, 0.82),
        ("Exact Skill Match", exact, 0.12, 0.763),
        ("AI Semantic Score", semantic, 0.24, 0.654),
    ]

    fig = go.Figure()
    for label, value, pad, hole in rings:
        color = get_score_color(value)
        fig.add_trace(
            go.Pie(
                values=[value, 100 - value],
                labels=[label, ""],
                hole=hole,
                domain=dict(x=[pad, 1 - pad], y=[pad, 1 - pad]),
                marker=dict(colors=[color, BORDER_LIGHT], line=dict(color="#FFFFFF", width=2)),
                direction="clockwise",
                rotation=0,
                sort=False,
                textinfo="none",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=240,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=f"<b>{overall}%</b>",
                x=0.5, y=0.5,
                font=dict(size=18, color=TEXT, family="Inter"),
                showarrow=False,
            )
        ],
        font=dict(family="Inter", color=TEXT),
    )
    return fig

def card_title(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)

def render_skill_pills(skills, bg, color):
    pills = "".join(
        f'<span style="background:{bg};color:{color};padding:0.35rem 0.9rem;'
        f'border-radius:999px;font-size:0.85rem;font-weight:600;margin:0.25rem;'
        f'display:inline-block;">{s}</span>'
        for s in skills
    )
    st.markdown(pills, unsafe_allow_html=True)

def render_ranked_list(valid_results):
    rows = []
    for i, r in enumerate(valid_results, 1):
        score = r["final_score"]
        color = get_score_color(score)
        initials = r["filename"][:2].upper()
        rank_style = f"background:{PRIMARY};color:#FFFFFF;" if i == 1 else f"background:transparent;color:{TEXT_SECONDARY};"
        rows.append(
            f'<div style="display:flex;align-items:center;padding:0.85rem 0.4rem;border-bottom:1px solid {BORDER_LIGHT};gap:1rem;">'
            f'<div style="width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;{rank_style}">{i}</div>'
            f'<div style="width:34px;height:34px;border-radius:50%;background:{LIGHT_ACCENT};color:{PRIMARY};display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.78rem;flex-shrink:0;">{initials}</div>'
            f'<div style="flex:1;min-width:0;">'
            f'<div style="font-weight:600;color:{TEXT};margin-bottom:0.35rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{r["filename"]}</div>'
            f'<div style="background:{BORDER_LIGHT};border-radius:999px;height:6px;width:100%;overflow:hidden;">'
            f'<div style="background:{color};height:100%;width:{score}%;border-radius:999px;"></div>'
            f'</div></div>'
            f'<div style="font-weight:700;color:{color};min-width:60px;text-align:right;">{score}%</div>'
            f'</div>'
        )
    st.markdown(f'<div style="max-height:325px;overflow-y:auto;">{"".join(rows)}</div>', unsafe_allow_html=True)

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

        with col_right:
            card_title("Job Description")
            job_description = st.text_area(
                "Job description",
                height=130,
                placeholder="Paste the full job description here...",
                label_visibility="collapsed",
            )

        st.write("")
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
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            card_title("Skill Breakdown")
            tab1, tab2, tab3 = st.tabs(["Matched", "Missing", "Extra"])

            with tab1:
                if results["matching_skills"]:
                    render_skill_pills(results["matching_skills"], PILL_GREEN_BG, PILL_GREEN_TEXT)
                    st.caption(f"{results['matched_count']} JD skills found in your resume.")
                else:
                    st.warning("No skill overlaps detected. Try expanding your skills section in the resume.")

            with tab2:
                if results["missing_skills"]:
                    render_skill_pills(results["missing_skills"], PILL_RED_BG, PILL_RED_TEXT)
                    st.caption(f"{results['missing_count']} skills required by the JD but not found in your resume.")
                else:
                    st.success("You have all the skills mentioned in the job description!")

            with tab3:
                if results["extra_skills"]:
                    render_skill_pills(results["extra_skills"], PILL_BLUE_BG, PILL_BLUE_TEXT)
                    st.caption(f"{len(results['extra_skills'])} extra skills found beyond what the JD requires.")
                else:
                    st.info("No extra skills beyond the JD requirements detected.")

        with col_right:
            card_title("Match Score")
            chart_col, legend_col = st.columns([2.2, 1], gap="small")

            with chart_col:
                st.plotly_chart(
                    plot_score_comparison(
                        results["final_score"], results["exact_score"], results["semantic_score"]
                    ),
                    use_container_width=True, config={"displayModeBar": False},
                )

            with legend_col:
                st.markdown(
                    f"""
                    <div style="display:flex; flex-direction:column; justify-content:center; height:260px; gap:1.4rem;">
                        <div style="text-align:center;">
                            <div style="width:9px;height:9px;border-radius:50%;background:{get_score_color(results['final_score'])};margin:0 auto 4px;"></div>
                            <div style="font-size:0.72rem;color:{TEXT_MUTED};">Overall</div>
                            <div style="font-weight:700;font-size:0.9rem;color:{TEXT};">{results['final_score']}%</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="width:9px;height:9px;border-radius:50%;background:{get_score_color(results['exact_score'])};margin:0 auto 4px;"></div>
                            <div style="font-size:0.72rem;color:{TEXT_MUTED};">Exact</div>
                            <div style="font-weight:700;font-size:0.9rem;color:{TEXT};">{results['exact_score']}%</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="width:9px;height:9px;border-radius:50%;background:{get_score_color(results['semantic_score'])};margin:0 auto 4px;"></div>
                            <div style="font-size:0.72rem;color:{TEXT_MUTED};">Semantic</div>
                            <div style="font-weight:700;font-size:0.9rem;color:{TEXT};">{results['semantic_score']}%</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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

        st.write("")
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
        card_title("Ranked Results")

        if valid_results:
            render_ranked_list(valid_results)
        else:
            st.warning("No resumes could be successfully processed.")

        if errored_results:
            with st.expander(f"{len(errored_results)} resume(s) failed to process"):
                for r in errored_results:
                    st.write(f"- **{r['filename']}**: {r['error']}")

    if valid_results:
        st.write("")
        with st.container(border=True):
            card_title("Candidate Details")

            options = [f"#{i} — {r['filename']} ({r['final_score']}%)" for i, r in enumerate(valid_results, 1)]
            selected = st.selectbox("Choose a candidate", options, label_visibility="collapsed", key="recruiter_candidate_select")
            r = valid_results[options.index(selected)]

            st.write("")
            col_left, col_right = st.columns(2, gap="large")

            with col_left:
                card_title("Skill Breakdown")
                tab1, tab2, tab3 = st.tabs(["Matched", "Missing", "Extra"])

                with tab1:
                    if r["matching_skills"]:
                        render_skill_pills(r["matching_skills"], PILL_GREEN_BG, PILL_GREEN_TEXT)
                        st.caption(f"{r['matched_count']} JD skills found in candidate's resume.")
                    else:
                        st.warning("No skill overlaps detected.")

                with tab2:
                    if r["missing_skills"]:
                        render_skill_pills(r["missing_skills"], PILL_RED_BG, PILL_RED_TEXT)
                        st.caption(f"{r['missing_count']} skills required by the JD but missing in candidate's resume.")
                    else:
                        st.success("Candidate has all skills mentioned in the job description!")

                with tab3:
                    if r["extra_skills"]:
                        render_skill_pills(r["extra_skills"], PILL_BLUE_BG, PILL_BLUE_TEXT)
                        st.caption(f"{len(r['extra_skills'])} extra skills found beyond what the JD requires.")
                    else:
                        st.info("No extra skills beyond the JD requirements detected.")

            with col_right:
                card_title("Match Score")
                chart_col, legend_col = st.columns([2.2, 1], gap="small")

                with chart_col:
                    st.plotly_chart(
                        plot_score_comparison(
                            r["final_score"], r["exact_score"], r["semantic_score"]
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key=f"candidate_chart_{options.index(selected)}",
                    )

                with legend_col:
                    st.markdown(
                        f"""
                        <div style="display:flex; flex-direction:column; justify-content:center; height:240px; gap:1.4rem;">
                            <div style="text-align:center;">
                                <div style="width:9px;height:9px;border-radius:50%;background:{get_score_color(r['final_score'])};margin:0 auto 4px;"></div>
                                <div style="font-size:0.72rem;color:{TEXT_MUTED};">Overall</div>
                                <div style="font-weight:700;font-size:0.9rem;color:{TEXT};">{r['final_score']}%</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="width:9px;height:9px;border-radius:50%;background:{get_score_color(r['exact_score'])};margin:0 auto 4px;"></div>
                                <div style="font-size:0.72rem;color:{TEXT_MUTED};">Exact</div>
                                <div style="font-weight:700;font-size:0.9rem;color:{TEXT};">{r['exact_score']}%</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="width:9px;height:9px;border-radius:50%;background:{get_score_color(r['semantic_score'])};margin:0 auto 4px;"></div>
                                <div style="font-size:0.72rem;color:{TEXT_MUTED};">Semantic</div>
                                <div style="font-weight:700;font-size:0.9rem;color:{TEXT};">{r['semantic_score']}%</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.write("")
        with st.container(border=True):
            card_title("AI Hiring Insights (Top 5 Candidates)")
            if st.button("Generate AI Explanation"):
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

mode = st.segmented_control(
    "Mode",
    ["Job Seeker", "Recruiter · Batch Screening"],
    default="Job Seeker",
    label_visibility="collapsed",
) or "Job Seeker"

if mode == "Job Seeker":
    render_job_seeker_mode()
else:
    render_recruiter_mode()

st.write("")
st.caption("© 2026 SkillSync AI. All rights reserved.")
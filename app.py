# Importing libraries
import streamlit as st

# For visualization
import plotly.graph_objects as go

# Importing everything from the logic file
from logic import (
    extract_text_from_pdf,
    compute_match_score,
    get_ai_recommendations,
    categorize_skills
)

@st.cache_resource
def download_nltk_data():
    """Download required NLTK data files silently."""
    import nltk
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception:
        pass

@st.cache_resource
def load_model():
    """Load the sentence transformer model (cached — loads once)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

download_nltk_data()

# Score Gauge Chart
def plot_gauge(score, title):
    if score < 40:
        color = "#ef4444"
    elif score < 70:
        color = "#f59e0b"
    else:
        color = "#22c55e"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'size': 14, 'color': '#94a3b8'}},
        number={'suffix': "%", 'font': {'size': 28, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#475569'},
            'bar': {'color': color},
            'bgcolor': "#1e293b",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 40], 'color': '#1f1f2e'},
                {'range': [40, 70], 'color': '#1e2535'},
                {'range': [70, 100], 'color': '#1e2d25'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=40, b=10, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0')
    )
    return fig

# Page configuration
st.set_page_config(
    page_title="Resume Skill Gap Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header
st.title("🧠 Resume Skill Gap Analyzer")
st.caption("AI-powered analysis to match your resume with any job description — instantly.")
st.divider()

# Input Section: Two columns for Resume Upload and Job Description
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("📄 Upload Your Resume")
    uploaded_resume = st.file_uploader(
        "Upload PDF Resume",
        type=["pdf"],
        help="Only PDF format is supported."
    )
    if uploaded_resume:
        st.success(f"✅ Resume uploaded: **{uploaded_resume.name}**")

with col_right:
    st.subheader("💼 Paste Job Description")
    job_description = st.text_area(
        "Job Description",
        height=180,
        placeholder=(
            "Paste the full job description here...\n\n"
            "Example:\nWe are looking for a Data Scientist with 2+ years of experience "
            "in Python, Machine Learning, SQL, and data visualization using Tableau or Power BI..."
        )
    )
    jd_word_count = len(job_description.split()) if job_description.strip() else 0
    st.caption(f"Word count: {jd_word_count}")

# Analyze Button
st.write("")
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze_clicked = st.button("Analyze Resume", use_container_width=True)

# Results Section (only shown after analysis is run)
if analyze_clicked:
    # Input Validation
    if not uploaded_resume:
        st.error("⚠️ Please upload your resume PDF first.")
        st.stop()
    if not job_description.strip() or len(job_description.strip()) < 50:
        st.error("⚠️ Please paste a job description (at least 50 characters).")
        st.stop()

    # Processing with spinner
    with st.spinner("🔍 Extracting skills & analyzing your resume with AI... This may take 20–40 seconds on first run."):

        # Step 1: Extract resume text from PDF
        resume_text = extract_text_from_pdf(uploaded_resume)
        if not resume_text or len(resume_text) < 100:
            st.error("⚠️ Could not extract text from the PDF. Make sure it's not a scanned image-only PDF.")
            st.stop()

        # Step 2: Load model and compute all scores
        model = load_model()
        results = compute_match_score(resume_text, job_description)

    st.divider()
    st.header("📊 Analysis Results")

    # Section 1: Key Metrics
    m1, m2, m3, m4 = st.columns(4)

    score = results['final_score']

    m1.metric(label="Overall Match", value=f"{results['final_score']}%")
    m2.metric(label="Skills Matched", value=results['matched_count'])
    m3.metric(label="Skills Missing", value=results['missing_count'])
    m4.metric(label="Bonus Skills", value=len(results['extra_skills']))

    st.write("")

    # Section 2: Score Gauges
    gc1, gc2, gc3 = st.columns(3)

    with gc1:
        st.plotly_chart(
            plot_gauge(results['final_score'], "Overall Match Score"),
            use_container_width=True, config={'displayModeBar': False}
        )
    with gc2:
        st.plotly_chart(
            plot_gauge(results['exact_score'], "Exact Skill Score"),
            use_container_width=True, config={'displayModeBar': False}
        )
    with gc3:
        st.plotly_chart(
            plot_gauge(results['semantic_score'], "AI Semantic Score"),
            use_container_width=True, config={'displayModeBar': False}
        )

    # Section 3: Skills breakdown
    st.subheader("🎯 Skill Breakdown")
    tab1, tab2, tab3 = st.tabs(["✅ Matched Skills", "❌ Missing Skills", "⭐ Extra Skills"])

    with tab1:
        if results['matching_skills']:
            st.write(" · ".join(results['matching_skills']))
            st.progress(min(results['matched_count'] / max(results['total_jd_skills'], 1), 1.0))
            st.caption(f"{results['matched_count']} out of {results['total_jd_skills']} JD skills found in your resume")
        else:
            st.warning("No skill overlaps detected. Try expanding your skills section in the resume.")

    with tab2:
        if results['missing_skills']:
            st.write(" · ".join(results['missing_skills']))
            st.caption("These skills are required by the JD but not found in your resume.")
        else:
            st.success("🎉 You have all the skills mentioned in the job description!")

    with tab3:
        if results['extra_skills']:
            st.write(" · ".join(results['extra_skills']))
            st.caption("These are skills you have beyond what the JD requires. Great for standing out!")
        else:
            st.info("No extra skills beyond the JD requirements detected.")

    st.write("")

    # Section 4: AI Recommendations
    st.divider()
    st.subheader("💡 AI-Powered Career Advice")

    ai_advice = "No AI advice generated."

    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        with st.spinner("🤖 Gemini is analyzing your profile..."):
            ai_advice = get_ai_recommendations(resume_text, job_description, api_key)
            st.info(ai_advice)
    else:
        st.warning("⚠️ API Key missing! Please add GEMINI_API_KEY to your secrets.toml.")

    # Footer
    st.write("")
    st.caption("Built with ❤️ using Streamlit · sentence-transformers · Plotly · pdfplumber · AI Resume Skill Gap Analyzer")


# Initial State (shown before analysis is run)
else:
    st.write("")
    st.info(
        "📋 Upload your resume and paste a job description above, "
        "then click **Analyze Resume** to get your results."
    )
    st.write("")
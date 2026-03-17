import streamlit as st
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>AI Resume Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your resume and get AI-powered insights</p>", unsafe_allow_html=True)

# -------------------------------
# Upload Section
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(" Upload Resume", type=["pdf"])

with col2:
    job_desc = st.text_area(" Paste Job Description (Optional)")

# -------------------------------
# Load Job Skills Dataset
# -------------------------------
df_jobs = pd.read_csv("skills_jobs.csv")

if uploaded_file is not None:
    # -------------------------------
    # Extract Resume Text
    # -------------------------------
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()

    # Resume Content
    st.subheader(" Resume Content")
    st.write(text)

    # -------------------------------
    # Skill Detection
    # -------------------------------
    skills_db = df_jobs["Skill"].unique().tolist()
    detected_skills = [skill for skill in skills_db if skill.lower() in text.lower()]
    detected_skills = list(set(detected_skills))

    st.subheader(" Detected Skills")
    skill_html = ""
    for skill in detected_skills:
        skill_html += f"<span style='background:#4CAF50;color:white;padding:6px 10px;margin:5px;border-radius:10px;display:inline-block'>{skill}</span>"
    st.markdown(skill_html, unsafe_allow_html=True)

    # -------------------------------
    # AI Recommended Jobs
    # -------------------------------
    job_skill_map = df_jobs.groupby("JobRole")["Skill"].apply(lambda x: " ".join(x)).to_dict()
    job_list = list(job_skill_map.keys())
    skill_docs = list(job_skill_map.values())
    documents = skill_docs + [" ".join(detected_skills)]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    resume_vector = vectors[-1]
    job_vectors = vectors[:-1]

    similarity = cosine_similarity(resume_vector, job_vectors).flatten()

    results = pd.DataFrame({"Job Role": job_list, "Match Score": similarity}).sort_values(by="Match Score", ascending=False)

    st.subheader(" AI Recommended Jobs")
    for i, row in results.head(3).iterrows():
        st.markdown(f"""
       <div style="padding:15px; margin:10px 0; border-radius:10px; border: 2px solid #3498db;">
    <h4>{row['Job Role']}</h4>
    <p>Match Score: {int(row['Match Score']*100)}%</p>
</div>
        """, unsafe_allow_html=True)

    best_job = results.iloc[0]["Job Role"]
    st.success(f" Best Matching Job Role: {best_job}")

    # -------------------------------
    # Job Description Matching
    # -------------------------------
    if job_desc:
        docs = [text, job_desc]
        vec = TfidfVectorizer()
        vectors2 = vec.fit_transform(docs)
        similarity_score = cosine_similarity(vectors2[0], vectors2[1])[0][0]

        st.subheader(" Job Match Score (JD vs Resume)")
        st.metric(label="Match %", value=f"{int(similarity_score*100)}%")
        st.success(" Analysis based on Job Description")

        # Extract skills from job description
        required_skills = [skill for skill in skills_db if skill.lower() in job_desc.lower()]
    else:
        st.subheader(" Job Analysis Based on AI Predicted Role")
        required_skills = df_jobs[df_jobs["JobRole"] == best_job]["Skill"].tolist()

    # -------------------------------
    # Skill Gap Analysis
    # -------------------------------
    missing_skills = [skill for skill in required_skills if skill not in detected_skills]

    st.subheader(" Missing Skills")
    if missing_skills:
        for skill in missing_skills:
            st.warning(skill)
    else:
        st.success("No missing skills detected!")

    # -------------------------------
    # Learning Roadmap
    # -------------------------------
    st.subheader(" Learning Roadmap")
    for skill in missing_skills:
        st.markdown(f"""
        <div style=" padding:10px; border-radius:8px; margin-bottom:10px; border: 2px solid #ffc107;">
    <b>{skill}</b><br>
    → Learn basics<br>
    → Build mini project<br>
    → Add to resume
</div>
        """, unsafe_allow_html=True)

    # -------------------------------
    # Resume Score
    # -------------------------------
    matched = len(set(detected_skills) & set(required_skills))
    resume_score = int((matched / len(required_skills))*100) if required_skills else 0

    st.subheader(" Resume Score")
    st.markdown(f"<h1 style='text-align:center;color:#4CAF50'>{resume_score}%</h1>", unsafe_allow_html=True)
    st.progress(resume_score)

    # -------------------------------
    # Score Breakdown
    # -------------------------------
    st.subheader(" Score Breakdown")
    skills_score = resume_score
    project_score = 70 if "project" in text.lower() else 40
    experience_score = 80 if "experience" in text.lower() else 50

    col1, col2, col3 = st.columns(3)
    col1.metric("Skills", f"{skills_score}%")
    col2.metric("Projects", f"{project_score}%")
    col3.metric("Experience", f"{experience_score}%")

    # -------------------------------
    # Skill Analysis Table
    # -------------------------------
    st.subheader(" Skill Analysis Table")
    data = {"Detected Skills": detected_skills, "Missing Skills": missing_skills}
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    st.table(df)
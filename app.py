import streamlit as st
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("AI Resume Analyzer")
st.write("Upload your resume and analyze your skills")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

# Load dataset
df_jobs = pd.read_csv("skills_jobs.csv")

if uploaded_file is not None:

    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()

    st.subheader("Resume Text")
    st.write(text)

    # Get all skills from dataset
    skills_db = df_jobs["Skill"].unique().tolist()

    detected_skills = []

    for skill in skills_db:
        if skill.lower() in text.lower():
            detected_skills.append(skill)

    detected_skills = list(set(detected_skills))

    st.subheader("Detected Skills")
    st.success(detected_skills)

    # AI Job Role Prediction
    job_skill_map = df_jobs.groupby("JobRole")["Skill"].apply(lambda x: " ".join(x)).to_dict()

    job_list = list(job_skill_map.keys())
    skill_docs = list(job_skill_map.values())

    documents = skill_docs + [" ".join(detected_skills)]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)

    resume_vector = vectors[-1]
    job_vectors = vectors[:-1]

    similarity = cosine_similarity(resume_vector, job_vectors).flatten()

    results = pd.DataFrame({
        "Job Role": job_list,
        "Match Score": similarity
    })

    results = results.sort_values(by="Match Score", ascending=False)

    st.subheader("AI Recommended Jobs")
    st.dataframe(results.head(5))

    best_job = results.iloc[0]["Job Role"]

    st.success(f"Best Matching Job Role: {best_job}")

    # Skill Gap Analysis
    required_skills = df_jobs[df_jobs["JobRole"] == best_job]["Skill"].tolist()

    missing_skills = []

    for skill in required_skills:
        if skill not in detected_skills:
            missing_skills.append(skill)

    st.subheader("Missing Skills")
    st.error(missing_skills)

    # Resume Score
    matched = len(set(detected_skills) & set(required_skills))

    if len(required_skills) > 0:
        score = int((matched / len(required_skills)) * 100)
    else:
        score = 0

    st.subheader("Resume Score")
    st.progress(score)
    st.write("Score:", score, "%")

    data = {
        "Detected Skills": detected_skills,
        "Missing Skills": missing_skills
    }

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

    st.subheader("Skill Analysis Table")
    st.table(df)
# app.py – Karuna's Resume Analyzer (Final Version)

import streamlit as st
from PyPDF2 import PdfReader

# Optional: JD Matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit page setup
st.set_page_config(page_title="Resume Analyzer", layout="centered")

# Title
st.title("📄 Karuna's AI Resume Analyzer")
st.write("Upload your resume PDF to get smart feedback on skills, score, and JD match!")

# Upload Resume
uploaded_file = st.file_uploader("📤 Upload Your Resume (PDF Only)", type=["pdf"])

if uploaded_file is not None:
    st.success("✅ Resume file uploaded!")

    if st.button("🔍 Analyze Resume"):
        with st.spinner("Reading and analyzing your resume..."):
            # STEP 1: Extract text from PDF
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # STEP 2: Show Resume Text Preview
            st.subheader("🔎 Extracted Resume Text:")
            st.write(text[:1000] + "...")

            # STEP 3: Detect Skills
            st.subheader("🧠 Detected Skills")

            known_skills = [
                "Python", "Java", "C++", "C", "SQL", "HTML", "CSS", "JavaScript",
                "Machine Learning", "Deep Learning", "Pandas", "NumPy", "React",
                "Node.js", "MongoDB", "Git", "Docker", "Kubernetes", "Power BI", "Excel"
            ]

            found_skills = [skill for skill in known_skills if skill.lower() in text.lower()]

            if found_skills:
                st.success("✅ Found Skills:")
                for skill in found_skills:
                    st.write(f"• {skill}")
            else:
                st.warning("⚠️ No known tech skills found in the resume.")

            # STEP 4: Resume Score
            st.subheader("📊 Resume Score")
            total_skills = len(known_skills)
            matched = len(found_skills)
            score = int((matched / total_skills) * 100)

            if score >= 70:
                st.success(f"🎉 Excellent! Your resume score is {score}/100")
            elif score >= 40:
                st.info(f"👍 Decent! Your resume score is {score}/100 — Can be improved")
            else:
                st.warning(f"⚠️ Low Resume Score: {score}/100")

            # STEP 5: Suggestions
            st.subheader("💡 Tips to Improve Your Resume")
            if score < 70:
                missing_skills = [skill for skill in known_skills if skill not in found_skills]
                for skill in missing_skills[:5]:  # Top 5 missing
                    st.write(f"• Consider adding: **{skill}** if relevant")
            else:
                st.write("✅ Your resume covers many important technical skills!")

# STEP 6: JD Upload + Match
st.subheader("📥 Upload a Job Description (Optional)")
jd_file = st.file_uploader("Upload JD as .txt file", type=["txt"])

if jd_file is not None:
    jd_text = jd_file.read().decode('utf-8')
    st.success("📄 Job Description uploaded!")

    docs = [text, jd_text]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)
    sim_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    percent = round(sim_score * 100, 2)

    st.subheader("📌 JD Match Score")
    if percent >= 70:
        st.success(f"🔥 Strong match! Your resume matches {percent}% of the JD.")
    elif percent >= 40:
        st.info(f"👍 Moderate match: {percent}%")
    else:
        st.warning(f"⚠️ Weak match: Only {percent}% of JD covered by your resume.")

    st.subheader("🛠 Suggestions")
    st.write("→ Make sure your resume contains keywords from the JD.")
    st.write("→ Add relevant projects, skills, or tools mentioned in the job post.")

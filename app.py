# app.py – Karuna's Resume Analyzer (Streamlit Cloud Ready)

import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import datetime
import spacy
import os

# ✅ Safe spaCy model loading for Streamlit Cloud
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Session login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    st.set_page_config(page_title="Resume Analyzer", layout="centered")
    st.title("🔐 Login")
    username = st.text_input("Enter your email to continue")
    if st.button("Login") and username:
        st.session_state.logged_in = True
        st.session_state.username = username
    st.stop()

# Main interface
st.title("📄 AI Resume Analyzer")
st.write("Upload your resume PDF to get smart feedback on skills, score, and JD match!")

# Upload resume
uploaded_file = st.file_uploader("📤 Upload Your Resume (PDF Only)", type=["pdf"])

if uploaded_file is not None:
    st.success("✅ Resume uploaded successfully!")

    if st.button("🔍 Analyze Resume"):
        with st.spinner("Reading and analyzing your resume..."):
            # Extract PDF text
            pdf_reader = PdfReader(uploaded_file)
            text = "".join(page.extract_text() for page in pdf_reader.pages)

            # Preview extracted text
            st.subheader("🔎 Extracted Resume Text")
            st.write(text[:1000] + "...")

            # NLP entity extraction
            st.subheader("🧠 NLP Entities")
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "EMAIL", "DATE"]:
                    st.write(f"• {ent.label_}: {ent.text}")

            # Skill detection
            st.subheader("💼 Detected Skills")
            known_skills = ["Python", "Java", "C++", "C", "SQL", "HTML", "CSS", "JavaScript",
                            "Machine Learning", "Deep Learning", "Pandas", "NumPy", "React",
                            "Node.js", "MongoDB", "Git", "Docker", "Kubernetes", "Power BI", "Excel"]
            found_skills = [skill for skill in known_skills if skill.lower() in text.lower()]

            if found_skills:
                st.success("✅ Skills Found:")
                for skill in found_skills:
                    st.write(f"• {skill}")
            else:
                st.warning("⚠️ No known tech skills found.")

            # Resume score
            st.subheader("📊 Resume Score")
            score = int((len(found_skills) / len(known_skills)) * 100)
            if score >= 70:
                st.success(f"🎉 Excellent Resume! Score: {score}/100")
            elif score >= 40:
                st.info(f"👍 Decent Resume. Score: {score}/100 – Can be improved")
            else:
                st.warning(f"⚠️ Weak Resume. Score: {score}/100")

            # Skill match chart
            st.subheader("📈 Skill Match Chart")
            matched = len(found_skills)
            missing = len([s for s in known_skills if s not in found_skills])
            fig, ax = plt.subplots()
            ax.bar(['Matched', 'Missing'], [matched, missing], color=['#22c55e', '#ef4444'])
            st.pyplot(fig)

            # Suggestions
            st.subheader("💡 Suggestions")
            if score < 70:
                st.info("Consider adding:")
                for skill in [s for s in known_skills if s not in found_skills][:5]:
                    st.write(f"• {skill}")
            else:
                st.write("✅ You're covering most important skills!")

            # Report download
            report = f"""Resume Analysis Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
User: {st.session_state.get("username", "Anonymous")}
Resume Score: {score}/100
Skills Found: {', '.join(found_skills)}
Missing Skills: {', '.join([s for s in known_skills if s not in found_skills])}
"""
            st.download_button("📥 Download Report", report, file_name="resume_report.txt")

# JD match section
st.subheader("📥 Optional: Upload Job Description")
jd_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"])
if jd_file is not None and uploaded_file is not None:
    jd_text = jd_file.read().decode('utf-8')
    docs = [text, jd_text]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)
    sim_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    percent = round(sim_score * 100, 2)

    st.subheader("📌 JD Match Score")
    if percent >= 70:
        st.success(f"🔥 Strong match! Resume matches {percent}% of JD")
    elif percent >= 40:
        st.info(f"👍 Moderate match: {percent}%")
    else:
        st.warning(f"⚠️ Weak match: Only {percent}% of JD covered")

    st.subheader("🛠 JD Match Suggestions")
    st.write("→ Include keywords from the job description in your resume.")
    st.write("→ Mention tools, skills, and experience relevant to the job.")

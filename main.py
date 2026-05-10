import os
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# JOB DESCRIPTION
# -----------------------------
job_description = """
Python developer with knowledge of data analysis,
machine learning, pandas, numpy, and SQL.
"""

# -----------------------------
# READ RESUME TEXT
# -----------------------------
resume_folder = "resumes"

resume_data = []

for file in os.listdir(resume_folder):

    if file.endswith(".pdf"):

        path = os.path.join(resume_folder, file)

        reader = PdfReader(path)

        text = ""

        for page in reader.pages:
            text += page.extract_text()

        resume_data.append({
            "Resume": file,
            "Text": text
        })

# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------
documents = [job_description]

for item in resume_data:
    documents.append(item["Text"])

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(documents)

# -----------------------------
# CALCULATE SIMILARITY
# -----------------------------
scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

# -----------------------------
# STORE RESULTS
# -----------------------------
results = []

for i in range(len(resume_data)):

    score = round(scores[i] * 100, 2)

    status = "Shortlisted" if score >= 30 else "Rejected"

    results.append({
        "Resume": resume_data[i]["Resume"],
        "Score": score,
        "Status": status
    })

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
df = pd.DataFrame(results)

# Sort by score
df = df.sort_values(by="Score", ascending=False)

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
print("\nResume Screening Results\n")

print(df)

# -----------------------------
# SAVE CSV REPORT
# -----------------------------
df.to_csv("shortlisted_candidates.csv", index=False)

print("\nReport generated successfully ✅")
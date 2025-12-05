from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load scraped data
df = pd.read_csv("scrapping_results.csv")

# Prepare TF-IDF vectorizer on article text + keywords
df["combined_text"] = df["Text"].fillna("") + " " + df["Keywords"].fillna("")

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])


def get_recommendations(query_text, top_n=10):
    # Convert input text/keywords into vector
    query_vec = vectorizer.transform([query_text])

    # Compute cosine similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Sort by similarity + claps
    df["similarity"] = similarities

    # Convert claps to numeric
    df["Claps_Num"] = df["Claps"].replace("K", "000", regex=True)
    df["Claps_Num"] = pd.to_numeric(df["Claps_Num"], errors="coerce").fillna(0)

    # Sort by:
    # 1) Highest similarity
    # 2) Highest claps
    results = df.sort_values(
        by=["similarity", "Claps_Num"], ascending=False
    ).head(top_n)

    # Return list of dicts
    return [
        {
            "title": row["Title"],
            "url": row["URL"],
            "claps": row["Claps"]
        }
        for i, row in results.iterrows()
    ]


@app.get("/")
def home():
    return {"message": "Medium Article Similarity API is running."}


@app.get("/recommend")
def recommend(q: str):
    results = get_recommendations(q)
    return {"query": q, "results": results}
#Run with: uvicorn Code_Deploy:app --host 0.0.0.0 --port 8000

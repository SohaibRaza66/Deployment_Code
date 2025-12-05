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


def compute_claps_numeric():
    # Convert claps like "1.2K" or "500" into numbers
    return (
        df["Claps"]
        .str.replace("K", "000", regex=False)
        .str.replace(".", "", regex=False)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


df["Claps_Num"] = compute_claps_numeric()


def get_recommendations(query_text, top_n=10):
    # Convert input text into vector
    query_vec = vectorizer.transform([query_text])

    # Compute cosine similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    df["similarity"] = similarities

    # Sort by similarity first, then by claps
    results = df.sort_values(
        by=["similarity", "Claps_Num"], ascending=False
    ).head(top_n)

    return [
        {
            "title": row["Title"],
            "url": row["URL"],
            "claps": row["Claps"],
            "similarity_score": float(row["similarity"])
        }
        for _, row in results.iterrows()
    ]


@app.get("/")
def home():
    return {"message": "Medium Article Similarity API is running."}


@app.get("/recommend")
def recommend(q: str):
    results = get_recommendations(q)
    return {"query": q, "results": results}


# 🔥 NEW ENDPOINT: Top 10 most-clapped similar articles
@app.get("/top_clapped_similar")
def top_clapped_similar(q: str):
    results = get_recommendations(q, top_n=10)

    # Sort only by claps_num to return the top clapped ones among the similar results
    sorted_by_claps = sorted(
        results, key=lambda x: float(x["claps"].replace("K", "000")), reverse=True
    )

    return {
        "query": q,
        "top_clapped_similar_articles": sorted_by_claps
    }


# Run with:
# uvicorn Code_Deploy:app --host 0.0.0.0 --port 8000

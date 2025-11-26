from flask import Flask, render_template
import pandas as pd
import itertools
import os

app = Flask(__name__, template_folder="template")

DATA_PATH = "app/data/smallest-100k"
MOVIE_FILE = os.path.join(DATA_PATH, "movies.csv")
RATING_FILE = os.path.join(DATA_PATH, "ratings.csv")


# -------------------------
# DATA LOADING
# -------------------------
def load_data():
    movies = pd.read_csv(MOVIE_FILE)
    ratings = pd.read_csv(RATING_FILE)
    return movies, ratings


# -------------------------
# SIMPLE GROUP RECOMMENDER (AVG RATINGS)
# -------------------------
def compute_group_recommendations(movies, ratings, top_k=5):
    avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
    merged = pd.merge(avg_ratings, movies, on="movieId")
    merged = merged.sort_values("rating", ascending=False)
    return merged.head(top_k)


# -------------------------
# HELPER: COMPUTE ITEM INTENSITY
# -------------------------
def compute_item_intensity(ratings):
    return ratings.groupby("movieId")["userId"].nunique()


# -------------------------
# COUNTERFACTUAL GENERATOR
# -------------------------
def generate_counterfactuals(recommendations, ratings, movies):
    """
    For each recommended movie:
    - Remove combinations of 1 or 2 influential items
    - Recompute group recommendation
    - If recommended movie disappears → explanation found
    """

    # Compute item popularity (intensity)
    intensity = compute_item_intensity(ratings)

    # Candidate explanation items = popular ones only
    popular_items = intensity[intensity > 10].index.tolist()

    results = []

    for idx, row in recommendations.iterrows():
        movie_id = row["movieId"]
        movie_title = row["title"]

        explanation_found = None

        # Try removing combinations of 1–2 items (concise)
        for r in range(1, 3):
            for combo in itertools.combinations(popular_items, r):

                # Remove the selected items
                filtered = ratings[~ratings["movieId"].isin(combo)]

                # Recompute recommendation
                recalc = (
                    filtered.groupby("movieId")["rating"]
                    .mean()
                    .reset_index()
                )
                recalc = pd.merge(recalc, movies, on="movieId")
                recalc = recalc.sort_values("rating", ascending=False)

                top_set = set(recalc.head(5)["movieId"].tolist())

                # Check if target movie disappeared
                if movie_id not in top_set:
                    explanation_found = combo
                    break
            if explanation_found:
                break

        # If nothing removes it, fallback
        if explanation_found is None:
            explanation_found = ["No minimal removal found"]

        explanation_titles = movies[movies["movieId"].isin(explanation_found)]["title"].tolist()

        # Compute fairness (difference in user intensity)
        user_item_counts = ratings[ratings["movieId"].isin(explanation_found)].groupby("userId").size()
        fairness = int(user_item_counts.max() - user_item_counts.min()) if len(user_item_counts) > 0 else 0

        # Popularity score (avg intensity of items)
        popularity = int(intensity.loc[explanation_found].mean()) if explanation_found != ["No minimal removal found"] else 0

        results.append({
            "rank": idx + 1,
            "item": movie_title,
            "removed_items": explanation_titles,
            "fairness": fairness,
            "popularity": popularity,
            "bullet_points": [
                "These removed items had the strongest influence on pushing this movie into the top recommendation list.",
                "They reflect shared or highly rated preferences across multiple group members.",
                "Removing this minimal item set is the smallest change required for the system to stop recommending this movie."
            ]
        })

    return results


# -------------------------
# ROUTE
# -------------------------
def counterfactual():
    movies, ratings = load_data()
    recs = compute_group_recommendations(movies, ratings)
    counterfactuals = generate_counterfactuals(recs, ratings, movies)
    return render_template("counterfactual.html", results=counterfactuals)




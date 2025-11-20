import os
import pandas as pd
import numpy as np
from flask import Flask, render_template

# ============================================================
# 1. LOAD MOVIELENS DATA
# ============================================================

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "data", "smallest-100k"))
ratings = pd.read_csv(os.path.join(base_dir, "ratings.csv"))
movies = pd.read_csv(os.path.join(base_dir, "movies.csv")) 


# build genre index
all_genres = set()
for g in movies["genres"]:
    for x in g.split("|"):
        if x != "(no genres listed)":
            all_genres.add(x)

all_genres = sorted(list(all_genres))
genre_index = {g: i for i, g in enumerate(all_genres)}
num_genres = len(all_genres)


def movie_to_vec(genres_str):
    v = np.zeros(num_genres)
    for g in genres_str.split("|"):
        if g in genre_index:
            v[genre_index[g]] = 1
    if np.linalg.norm(v) > 0:
        v = v / np.linalg.norm(v)
    return v


movies["genre_vec"] = movies["genres"].apply(movie_to_vec)

# ============================================================
# 2. GROUP PROFILE
# ============================================================

group = [1, 7, 32, 555]
group_ratings = ratings[ratings["userId"].isin(group)]

user_prefs = {}
for uid, df in group_ratings.groupby("userId"):
    vec = np.zeros(num_genres)
    for _, row in df.iterrows():
        mv = movies[movies.movieId == row.movieId].iloc[0]
        vec += row.rating * mv["genre_vec"]
    if np.linalg.norm(vec) > 0:
        vec /= np.linalg.norm(vec)
    user_prefs[uid] = vec

# ============================================================
# 3. FUNCTIONS: relevance, distance, coverage
# ============================================================


def group_relevance(movie_vec):
    return np.mean([np.dot(pref, movie_vec) for pref in user_prefs.values()])


def dissimilarity (v1, v2):
    return max(0, 1 - np.dot(v1, v2))


def coverage_gain(movie_vec, selected_vecs):
    if len(selected_vecs) == 0:
        return np.sum(movie_vec)
    covered = np.minimum(np.sum(selected_vecs, axis=0), 1.0)
    new_mass = np.maximum(movie_vec - covered, 0)
    return np.sum(new_mass)


movies["relevance"] = movies["genre_vec"].apply(group_relevance)

watched = set(group_ratings.movieId.unique())
candidates = movies[~movies.movieId.isin(watched)].copy()
candidates = candidates.sort_values("relevance", ascending=False)

# ============================================================
# 4. DIVERSITY FUNCTION (previously SG_MMR)
# ============================================================

def diversity(candidates, k=10, alpha=0.5, beta=0.3, gamma=0.2):
    """
    Your SG-MMR diversification, but named `diversity()`
    so it can be imported in the root repository.
    """
    selected = []
    selected_vecs = []

    pool = candidates.copy()

    for _ in range(k):
        scores = []
        for _, row in pool.iterrows():
            vec = row["genre_vec"]
            rel = row["relevance"]

            if len(selected_vecs) == 0:
                diversity_score = 1.0
            else:
                diversity_score = min(dissimilarity (vec, s) for s in selected_vecs)
            diversity_score = np.clip(diversity_score, 0, 1)

            cov = coverage_gain(vec, selected_vecs) / num_genres

            score = alpha * rel + beta * diversity_score + gamma * cov
            scores.append((score, row))

        scores.sort(key=lambda x: x[0], reverse=True)
        best = scores[0][1]

        selected.append(best)
        selected_vecs.append(best["genre_vec"])

        pool = pool[pool.movieId != best.movieId]

    return pd.DataFrame(selected)

# ============================================================
# 5. METRICS
# ============================================================

def average_relevance(df):
    return float(df["relevance"].mean())


def intra_list_diversity(df):
    vecs = list(df["genre_vec"])
    if len(vecs) <= 1:
        return 0
    d_sum = 0
    pairs = 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            d_sum += dissimilarity (vecs[i], vecs[j])
            pairs += 1
    return float(d_sum / pairs)


def genre_coverage(df):
    union_vec = np.sum(list(df["genre_vec"]), axis=0)
    union_vec = np.minimum(union_vec, 1.0)
    return float(np.sum(union_vec) / num_genres)


def evaluate_metrics(df):
    return (
        round(average_relevance(df), 4),
        round(intra_list_diversity(df), 4),
        round(genre_coverage(df), 4),
    )


# ============================================================
# 6. FLASK APP
# ============================================================



def show_diversity():

    k = 10
    result = diversity(candidates, k=k)

    avg_rel, ild, cov = evaluate_metrics(result)

    rows = result[["movieId", "title", "relevance"]].to_dict(orient="records")

    return render_template(
        "diversity.html",
        k=k,
        avg_rel=avg_rel,
        ild=ild,
        cov=cov,
        results=rows
    )


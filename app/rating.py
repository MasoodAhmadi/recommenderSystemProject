import os
import math
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request
from app.utils import pearson_similarity, cosine_similarity

# main_bp = Blueprint("main", __name__)

# @main_bp.route("/part1", methods=["GET", "POST"])
def index():
    # ---------- Load dataset ----------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "data", "smallest-100k"))
    ratings = pd.read_csv(os.path.join(base_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(base_dir, "movies.csv"))  # expects columns: movieId, title

    # Ensure same type
    movies["movieId"] = movies["movieId"].astype(int)
    movie_map = dict(zip(movies["movieId"], movies["title"]))

    # ---------- Merge ratings with movies ----------
    ratings = ratings.merge(movies[["movieId", "title"]], on="movieId", how="left")
    ratings.rename(columns={"title": "movieName"}, inplace=True)

    # ---------- Dataset tab pagination ----------
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 100))
    per_page = per_page if per_page in [100, 200, 500] else 100

    row_count = len(ratings)
    total_pages = math.ceil(row_count / per_page)
    dataset_page = ratings.iloc[(page - 1) * per_page : page * per_page].to_dict(orient="records")

    # ---------- User-Based CF ----------
    rating_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    users = rating_matrix.index.tolist()
    similarity_method = "pearson"

    all_predictions = []

    # Compute predictions for top 10 users
    for selected_user in users[:10]:
        similarities = []
        for other_user in users:
            if other_user == selected_user:
                continue
            sim = pearson_similarity(rating_matrix.loc[selected_user], rating_matrix.loc[other_user]) \
                  if similarity_method == "pearson" \
                  else cosine_similarity(rating_matrix.loc[selected_user], rating_matrix.loc[other_user])
            similarities.append({"user": other_user, "similarity": round(sim, 3)})

        similarities_df = pd.DataFrame(similarities).sort_values(by="similarity", ascending=False)

        def predict_rating(target_user, movie_id):
            numerator, denominator = 0, 0
            target_mean = rating_matrix.loc[target_user].mean()
            contributors = []

            for _, row in similarities_df.iterrows():
                other_user, sim = row["user"], row["similarity"]
                if sim <= 0 or pd.isna(rating_matrix.loc[other_user, movie_id]):
                    continue
                other_mean = rating_matrix.loc[other_user].mean()
                diff = rating_matrix.loc[other_user, movie_id] - other_mean
                numerator += sim * diff
                denominator += abs(sim)
                contributors.append((other_user, sim))

            if denominator == 0:
                return np.nan, (None, 0)

            pred_rating = target_mean + numerator / denominator
            top_contributor = max(contributors, key=lambda x: x[1]) if contributors else (None, 0)
            return pred_rating, top_contributor

        unrated_movies = rating_matrix.loc[selected_user][rating_matrix.loc[selected_user].isna()].index.tolist()
        for movie_id in unrated_movies[:5]:
            pred, top_user = predict_rating(selected_user, movie_id)
            all_predictions.append({
                "userId": selected_user,
                "movieId": movie_id,
                "movieName": movie_map.get(movie_id, f"Movie {movie_id}"),
                "predicted_rating": round(pred, 2) if not pd.isna(pred) else "N/A",
                "similar_user": f"User {top_user[0]}" if top_user[0] else "N/A",
                "similarity": round(top_user[1], 3) if top_user[1] else "N/A"
            })

    # ---------- CF pagination ----------
    cf_page = int(request.args.get("cf_page", 1))
    cf_per_page = int(request.args.get("cf_per_page", 100))
    cf_per_page = cf_per_page if cf_per_page in [100, 200, 500] else 100
    cf_total = len(all_predictions)
    cf_total_pages = math.ceil(cf_total / cf_per_page) if cf_total > 0 else 1
    predictions_page = all_predictions[(cf_page - 1) * cf_per_page : cf_page * cf_per_page]

    # ---------- User Similarity Matrix for Tab 2 ----------
    top_users = users[:5]
    similarity_matrix = pd.DataFrame(index=top_users, columns=top_users, dtype=float)
    for u1 in top_users:
        for u2 in top_users:
            similarity_matrix.loc[u1, u2] = 1.0 if u1 == u2 else pearson_similarity(rating_matrix.loc[u1], rating_matrix.loc[u2])
    user_similarities = similarity_matrix.reset_index().rename(columns={"index": "User"}).fillna(0).to_dict(orient="records")

    # ---------- Group Recommendation Aggregation ----------
    pred_df = pd.DataFrame(all_predictions)
    pred_df = pred_df[pd.to_numeric(pred_df["predicted_rating"], errors="coerce").notna()]
    pred_df["predicted_rating"] = pred_df["predicted_rating"].astype(float)
    pred_df["movieName"] = pred_df["movieId"].map(movie_map)

    group_users = users[:5]
    group_preds = pred_df[pred_df["userId"].isin(group_users)]

    # Average Method
    avg_group_recs = (
        group_preds.groupby(["movieId", "movieName"])["predicted_rating"]
        .mean()
        .reset_index()
        .sort_values(by="predicted_rating", ascending=False)
    )

    # Least Misery Method
    least_misery_recs = (
        group_preds.groupby(["movieId", "movieName"])["predicted_rating"]
        .min()
        .reset_index()
        .sort_values(by="predicted_rating", ascending=False)
    )

    # Disagreement-Aware Group Recommendation
    disagreement_df = (
        group_preds.groupby(["movieId", "movieName"])["predicted_rating"]
        .var()
        .reset_index()
        .rename(columns={"predicted_rating": "disagreement"})
    )
    disagreement_aware = pd.merge(avg_group_recs, disagreement_df, on=["movieId", "movieName"], how="left")
    disagreement_aware["disagreement"] = disagreement_aware["disagreement"].fillna(0)
    alpha = 0.7
    disagreement_aware["group_score"] = disagreement_aware["predicted_rating"] - alpha * disagreement_aware["disagreement"]
    disagreement_aware = disagreement_aware.sort_values(by="group_score", ascending=False)

    # Convert to lists for rendering
    top_avg_recs = avg_group_recs.head(10).to_dict(orient="records")
    top_misery_recs = least_misery_recs.head(10).to_dict(orient="records")
    top_disagreement_recs = disagreement_aware.head(10).to_dict(orient="records")

    print("Sample Average Group Recommendations:")
    print(avg_group_recs.head(5))

    movie_id = 6
    count = ratings[ratings["movieId"] == movie_id]["rating"].count()

    print(f"Movie ID {movie_id} has been rated {count} times.")

    return render_template(
        "rating.html",
        row_count=row_count,
        data=dataset_page,
        page=page,
        total_pages=total_pages,
        per_page=per_page,
        predictions=predictions_page,
        cf_page=cf_page,
        cf_total_pages=cf_total_pages,
        cf_per_page=cf_per_page,
        user_similarities=user_similarities,
        top_avg_recs=top_avg_recs,
        top_misery_recs=top_misery_recs,
        total_predictions=len(all_predictions),
        top_disagreement_recs=top_disagreement_recs,
    )

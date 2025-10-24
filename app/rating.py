import os
import math
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request

main_bp = Blueprint("main", __name__)

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    # ------------------- Load Dataset -------------------
    file_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'data', 'smallest-100k', 'ratings.csv')
    file_path = os.path.abspath(file_path)
    ratings = pd.read_csv(file_path)

    # ------------------- Dataset Tab Pagination -------------------
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 100))
    per_page = per_page if per_page in [100, 200, 500] else 100

    row_count = len(ratings)
    total_pages = math.ceil(row_count / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    dataset_page = ratings.iloc[start:end].to_dict(orient='records')

    # ------------------- Prepare Rating Matrix -------------------
    rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

    # ------------------- Pearson Similarity Function -------------------
    def pearson_similarity(user1, user2):
        """Compute Pearson correlation between two users"""
        common = rating_matrix.loc[[user1, user2]].dropna(axis=1, how='any')
        if common.shape[1] == 0:
            return 0
        u1 = common.loc[user1]
        u2 = common.loc[user2]
        if np.std(u1) == 0 or np.std(u2) == 0:
            return 0
        return np.corrcoef(u1, u2)[0, 1]

    # ------------------- Prediction Function -------------------
    def predict_rating(user_id, movie_id, k=5):
        """Predict rating of a movie for a user using top-k similar users"""
        if movie_id not in rating_matrix.columns:
            return np.nan

        users_who_rated = rating_matrix[movie_id].dropna().index
        sims = []
        for other_user in users_who_rated:
            if other_user != user_id:
                sim = pearson_similarity(user_id, other_user)
                if sim > 0: 
                    sims.append((other_user, sim))

        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:k]

        if not sims:
            return rating_matrix.loc[user_id].mean()  

        user_mean = rating_matrix.loc[user_id].mean()
        numerator, denominator = 0, 0
        for other_user, sim in sims:
            other_mean = rating_matrix.loc[other_user].mean()
            numerator += sim * (rating_matrix.loc[other_user, movie_id] - other_mean)
            denominator += abs(sim)

        if denominator == 0:
            return user_mean
        return round(user_mean + numerator / denominator, 2)

    # ------------------- Handle User Selection -------------------
    users = rating_matrix.index.tolist()
    selected_user = int(request.args.get('user_id', users[0]))

    user_ratings = rating_matrix.loc[selected_user]
    unrated_movies = user_ratings[user_ratings.isna()].index.tolist()

    predictions = []
    for movie_id in unrated_movies[:50]: 
        predicted = predict_rating(selected_user, movie_id)
        predictions.append({'movieId': movie_id, 'predicted_rating': predicted})

    predictions = sorted(predictions, key=lambda x: x['predicted_rating'], reverse=True)

    # ------------------- Render Template -------------------
    return render_template("index.html",
                           # Tab 0 (Dataset)
                           row_count=row_count,
                           data=dataset_page,
                           page=page,
                           total_pages=total_pages,
                           per_page=per_page,
                           # Tab 1 (User-based CF)
                           users=users,
                           selected_user=selected_user,
                           predictions=predictions)

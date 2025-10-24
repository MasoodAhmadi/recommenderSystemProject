import os
import math
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request

main_bp = Blueprint("main", __name__)

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    # ---------- Load dataset ----------
    file_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'data', 'smallest-100k', 'ratings.csv')
    file_path = os.path.abspath(file_path)
    ratings = pd.read_csv(file_path)

    # ---------- Dataset tab pagination ----------
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 100))
    per_page = per_page if per_page in [100, 200, 500] else 100

    row_count = len(ratings)
    total_pages = math.ceil(row_count / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    dataset_page = ratings.iloc[start:end].to_dict(orient='records')

    # ---------- User-Based CF ----------
    rating_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
    users = rating_matrix.index.tolist()

    # ---------- Similarity functions ----------
    def pearson_similarity(u1, u2):
        """Compute Pearson correlation between two users"""
        u1_ratings = rating_matrix.loc[u1]
        u2_ratings = rating_matrix.loc[u2]
        common = u1_ratings.notna() & u2_ratings.notna()
        if not common.any():
            return 0
        u1_common = u1_ratings[common]
        u2_common = u2_ratings[common]
        if len(u1_common) < 2:
            return 0
        sim = np.corrcoef(u1_common, u2_common)[0, 1]
        return 0 if np.isnan(sim) else sim

    def cosine_similarity(u1, u2):
        """Compute Cosine similarity between two users"""
        u1_ratings = rating_matrix.loc[u1]
        u2_ratings = rating_matrix.loc[u2]
        common_mask = u1_ratings.notna() & u2_ratings.notna()
        if not common_mask.any():
            return 0
        u1_common = u1_ratings[common_mask].values
        u2_common = u2_ratings[common_mask].values
        numerator = np.dot(u1_common, u2_common)
        denominator = np.linalg.norm(u1_common) * np.linalg.norm(u2_common)
        if denominator == 0:
            return 0
        return numerator / denominator

    # Choose similarity method: "pearson" or "cosine"
    similarity_method = "pearson"  # change to "cosine" if desired

    all_predictions = []

    # loop over top 10 users
    for selected_user in users[:10]:
        similarities = []
        for other_user in users:
            if other_user != selected_user:
                if similarity_method == "pearson":
                    sim = pearson_similarity(selected_user, other_user)
                else:
                    sim = cosine_similarity(selected_user, other_user)
                similarities.append({'user': other_user, 'similarity': round(sim, 3)})

        similarities_df = pd.DataFrame(similarities).sort_values(by='similarity', ascending=False)

        def predict_rating(target_user, movie_id):
            numerator, denominator = 0, 0
            target_mean = rating_matrix.loc[target_user].mean()
            contributors = []
            for _, row in similarities_df.iterrows():
                other_user = row['user']
                sim = row['similarity']
                if sim <= 0:
                    continue
                if not np.isnan(rating_matrix.loc[other_user, movie_id]):
                    other_mean = rating_matrix.loc[other_user].mean()
                    diff = rating_matrix.loc[other_user, movie_id] - other_mean
                    numerator += sim * diff
                    denominator += abs(sim)
                    contributors.append((other_user, sim))

            if denominator == 0:
                return np.nan, []

            pred_rating = target_mean + numerator / denominator
            top_contributor = max(contributors, key=lambda x: x[1]) if contributors else (None, 0)
            return pred_rating, top_contributor

        unrated_movies = rating_matrix.loc[selected_user][rating_matrix.loc[selected_user].isna()].index.tolist()
        for movie_id in unrated_movies[:5]:
            pred, top_user = predict_rating(selected_user, movie_id)
            all_predictions.append({
                'userId': selected_user,
                'movieId': movie_id,
                'predicted_rating': round(pred, 2) if not np.isnan(pred) else 'N/A',
                'similar_user': f'User {top_user[0]}' if top_user[0] else 'N/A',
                'similarity': round(top_user[1], 3) if top_user[1] else 'N/A'
            })

    # ---------- CF pagination ----------
    cf_page = int(request.args.get('cf_page', 1))
    cf_per_page = int(request.args.get('cf_per_page', 100))
    cf_per_page = cf_per_page if cf_per_page in [100, 200, 500] else 100
    cf_total = len(all_predictions)
    cf_total_pages = math.ceil(cf_total / cf_per_page) if cf_total > 0 else 1
    cf_start = (cf_page - 1) * cf_per_page
    cf_end = cf_start + cf_per_page
    predictions_page = all_predictions[cf_start:cf_end]
    


    # ---------- User Similarity Matrix for Tab 2 ----------
    top_users = users[:20]  # first 20 users
    
    similarity_matrix = pd.DataFrame(index=top_users, columns=top_users, dtype=float)

    for u1 in top_users:
        for u2 in top_users:
            if u1 == u2:
                similarity_matrix.loc[u1, u2] = 1.0
            else:
                similarity_matrix.loc[u1, u2] = pearson_similarity(u1, u2)
            print(f"Similarity between User {u1} and User {u2}: {similarity_matrix.loc[u1, u2]}")

    # Convert to list of dicts for Jinja rendering
    user_similarities = similarity_matrix.reset_index().rename(columns={'index': 'User'}).fillna(0).to_dict(orient='records')

    print(f"Similarities for User {user_similarities}:")


    return render_template(
        "index.html",
        # Dataset tab
        row_count=row_count,
        data=dataset_page,
        page=page,
        total_pages=total_pages,
        per_page=per_page,
        # CF tab
        predictions=predictions_page,
        cf_page=cf_page,
        cf_total_pages=cf_total_pages,
        cf_per_page=cf_per_page,
        # User similarity tab
        user_similarities=user_similarities,
    )

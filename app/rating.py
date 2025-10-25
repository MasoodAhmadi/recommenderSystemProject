import os
import math
import pandas as pd
from flask import Blueprint, render_template, request
from app.utils import pearson_similarity, cosine_similarity  # Import the functions

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

    # Choose similarity method: "pearson" or "cosine"
    similarity_method = "pearson"  # change to "cosine" if desired

    all_predictions = []

    # loop over top 10 users
    for selected_user in users[:10]:
        similarities = []
        for other_user in users:
            if other_user != selected_user:
                if similarity_method == "pearson":
                    sim = pearson_similarity(rating_matrix.loc[selected_user], rating_matrix.loc[other_user])
                else:
                    sim = cosine_similarity(rating_matrix.loc[selected_user], rating_matrix.loc[other_user])
                similarities.append({'user': other_user, 'similarity': round(sim, 3)})

        similarities_df = pd.DataFrame(similarities).sort_values(by='similarity', ascending=False)

        def predict_rating(target_user, movie_id):
            numerator, denominator = 0, 0
            target_mean = rating_matrix.loc[target_user].mean()
            contributors = []
            for _, row in similarities_df.iterrows():
                other_user = row['user']
                sim = row['similarity']
                if sim < 0:
                    continue
                if not pd.isna(rating_matrix.loc[other_user, movie_id]):
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
                'predicted_rating': round(pred, 2) if not pd.isna(pred) else 'N/A',
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
                similarity_matrix.loc[u1, u2] = pearson_similarity(rating_matrix.loc[u1], rating_matrix.loc[u2])

    # Convert to list of dicts for Jinja rendering
    user_similarities = similarity_matrix.reset_index().rename(columns={'index': 'User'}).fillna(0).to_dict(orient='records')
        # ---------- Group Recommendation Aggregation ----------
    # Convert user predictions into DataFrame
    pred_df = pd.DataFrame(all_predictions)

    # Keep only numeric predictions
    pred_df = pred_df[pd.to_numeric(pred_df['predicted_rating'], errors='coerce').notna()]
    pred_df['predicted_rating'] = pred_df['predicted_rating'].astype(float)

    # Example: define a group (you can change the users)
    group_users = users[:5]  # first 5 users in the dataset

    # Filter predictions for group members
    group_preds = pred_df[pred_df['userId'].isin(group_users)]

    # Compute average rating per movie (Average Method)
    avg_group_recs = (
        group_preds.groupby('movieId')['predicted_rating']
        .mean()
        .reset_index()
        .sort_values(by='predicted_rating', ascending=False)
    )

    # Compute least misery rating per movie (Least Misery Method)
    least_misery_recs = (
        group_preds.groupby('movieId')['predicted_rating']
        .min()
        .reset_index()
        .sort_values(by='predicted_rating', ascending=False)
    )

    # Take top 10 movies for each method
    top_avg_recs = avg_group_recs.head(10).to_dict(orient='records')
    top_misery_recs = least_misery_recs.head(10).to_dict(orient='records')


    print(f"✅ Total predictions computed: {len(all_predictions)}")
    if len(all_predictions) > 0:
        print(pd.DataFrame(all_predictions).head(10))
    else:
        print("⚠️ No predictions generated. Try adjusting similarity threshold or data sample.")


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

         # Group recommendations
        top_avg_recs=top_avg_recs,
        top_misery_recs=top_misery_recs,
        total_predictions=len(all_predictions),

    )

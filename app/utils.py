import numpy as np

def pearson_similarity(u1_ratings, u2_ratings):
    """Compute Pearson correlation between two users."""
    common = u1_ratings.notna() & u2_ratings.notna()
    if not common.any():
        return 0
    u1_common = u1_ratings[common]
    u2_common = u2_ratings[common]
    if len(u1_common) < 2:
        return 0
    sim = np.corrcoef(u1_common, u2_common)[0, 1]
    return 0 if np.isnan(sim) else sim

def cosine_similarity(u1_ratings, u2_ratings):
    """Compute Cosine similarity between two users."""
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
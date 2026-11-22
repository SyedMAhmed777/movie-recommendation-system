
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NEIGHBORS = 20
TOP_K = 10
RELEVANCE_THRESHOLD = 4


# ==================================================
# STEP 1: LOAD AND CLEAN RATINGS
# ==================================================

def load_ratings():

    ratings = pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t",
        names=[
            "user_id",
            "movie_id",
            "rating",
            "timestamp"
        ]
    )

    ratings = ratings.dropna(
        subset=["user_id", "movie_id", "rating"]
    ).copy()

    ratings = ratings[
        ratings["rating"].between(1, 5)
    ]

    ratings = ratings.drop_duplicates(
        subset=["user_id", "movie_id"],
        keep="last"
    )

    return ratings


# ==================================================
# STEP 2: PREPARE COLLABORATIVE FILTERING MODEL
# ==================================================

def prepare_model(train):

    # Create user-movie rating matrix
    matrix = train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    )

    # Calculate average rating for each user
    user_means = matrix.mean(axis=1)

    # Center ratings around user averages
    centered = matrix.sub(
        user_means,
        axis=0
    ).fillna(0)

    # Calculate cosine similarity between users
    similarities = cosine_similarity(centered)

    similarity_df = pd.DataFrame(
        similarities,
        index=matrix.index,
        columns=matrix.index
    )

    # Global average rating
    global_mean = float(train["rating"].mean())

    return (
        matrix,
        user_means,
        similarity_df,
        global_mean
    )


# ==================================================
# STEP 3: PREDICT USER RATINGS
# ==================================================

def predict_rating(
    user_id,
    movie_id,
    matrix,
    user_means,
    similarity_df,
    global_mean
):

    # Handle users not found in training data
    if user_id not in matrix.index:
        return float(global_mean)

    user_mean = float(user_means.loc[user_id])

    # Handle movies not found in training data
    if movie_id not in matrix.columns:
        return user_mean

    # Find users who rated the movie
    movie_ratings = matrix[movie_id].dropna()

    if movie_ratings.empty:
        return user_mean

    # Retrieve similarity scores
    similarities = similarity_df.loc[
        user_id,
        movie_ratings.index
    ]

    # Exclude target user
    similarities = similarities.drop(
        labels=[user_id],
        errors="ignore"
    )

    # Select top positive neighbors
    similarities = similarities[
        similarities > 0
    ].sort_values(
        ascending=False
    ).head(NEIGHBORS)

    if similarities.empty:
        return user_mean

    neighbor_ratings = movie_ratings.loc[
        similarities.index
    ]

    neighbor_means = user_means.loc[
        similarities.index
    ]

    # Weighted rating deviations
    weighted_deviations = (
        similarities
        * (neighbor_ratings - neighbor_means)
    )

    prediction = (
        user_mean
        + weighted_deviations.sum()
        / similarities.sum()
    )

    # Keep ratings within 1 to 5
    return float(np.clip(prediction, 1, 5))


# ==================================================
# STEP 4: EVALUATE RECOMMENDATION SYSTEM
# ==================================================

def evaluate():

    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM EVALUATION")
    print("=" * 60)

    # ----------------------------------------------
    # LOAD DATA
    # ----------------------------------------------

    ratings = load_ratings()

    # Split dataset into training and testing data
    train, test = train_test_split(
        ratings,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    print("\nTraining ratings:", len(train))
    print("Testing ratings:", len(test))

    # Prepare model using training data only
    (
        matrix,
        user_means,
        similarity_df,
        global_mean
    ) = prepare_model(train)

    # ----------------------------------------------
    # COLLABORATIVE FILTERING PREDICTIONS
    # ----------------------------------------------

    actual = []
    predicted = []
    baseline_predictions = []

    for row in test.itertuples(index=False):

        # Collaborative filtering prediction
        prediction = predict_rating(
            row.user_id,
            row.movie_id,
            matrix,
            user_means,
            similarity_df,
            global_mean
        )

        actual.append(float(row.rating))
        predicted.append(prediction)

        # Baseline prediction: user average
        if row.user_id in user_means.index:

            baseline_prediction = float(
                user_means.loc[row.user_id]
            )

        else:

            baseline_prediction = global_mean

        baseline_predictions.append(
            baseline_prediction
        )

    # ----------------------------------------------
    # CALCULATE RMSE
    # ----------------------------------------------

    collaborative_rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )
    )

    baseline_rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                baseline_predictions
            )
        )
    )

    # Percentage improvement over baseline
    improvement = (
        (baseline_rmse - collaborative_rmse)
        / baseline_rmse
    ) * 100

    print("\nMODEL COMPARISON")
    print("-" * 45)

    print(
        "Baseline RMSE:",
        round(baseline_rmse, 4)
    )

    print(
        "Collaborative Filtering RMSE:",
        round(collaborative_rmse, 4)
    )

    print(
        "RMSE Improvement (%):",
        round(improvement, 2)
    )

    if collaborative_rmse < baseline_rmse:

        print(
            "Result: Collaborative filtering "
            "outperformed the baseline."
        )

    else:

        print(
            "Result: Baseline performed "
            "as well or better."
        )

    # ----------------------------------------------
    # PRECISION@K EVALUATION
    # ----------------------------------------------
    # Rank held-out test movies for each user.
    # Ratings of 4 or 5 are considered relevant.
    #
    # This is held-out candidate Precision@K,
    # NOT full-catalog Precision@K.
    # ----------------------------------------------

    precision_scores = []

    # Reuse predictions already calculated above
    test_predictions = test.copy()

    test_predictions["predicted_rating"] = predicted

    for user_id, user_test in test_predictions.groupby(
        "user_id"
    ):

        # Skip users not present in training data
        if user_id not in matrix.index:
            continue

        # Skip users without relevant test movies
        if not (
            user_test["rating"] >= RELEVANCE_THRESHOLD
        ).any():
            continue

        # Rank movies by predicted rating
        ranked = user_test.sort_values(
            by="predicted_rating",
            ascending=False
        ).head(TOP_K)

        if ranked.empty:
            continue

        # Count relevant recommendations
        relevant_count = (
            ranked["rating"] >= RELEVANCE_THRESHOLD
        ).sum()

        # Use actual number of ranked candidates
        precision = relevant_count / len(ranked)

        precision_scores.append(
            float(precision)
        )

    precision_at_k = (
        float(np.mean(precision_scores))
        if precision_scores
        else 0.0
    )

    print("\nRANKING EVALUATION")
    print("-" * 45)

    print(
        f"Precision@{TOP_K} (held-out candidates):",
        round(precision_at_k, 4)
    )

    print(
        "Users evaluated:",
        len(precision_scores)
    )

    # ----------------------------------------------
    # SAVE EVALUATION RESULTS
    # ----------------------------------------------

    results = pd.DataFrame([
        {
            "metric": "Baseline_RMSE",
            "value": baseline_rmse
        },
        {
            "metric": "Collaborative_Filtering_RMSE",
            "value": collaborative_rmse
        },
        {
            "metric": "RMSE_Improvement_Percent",
            "value": improvement
        },
        {
            "metric": f"Precision@{TOP_K}_heldout",
            "value": precision_at_k
        },
        {
            "metric": "Users_Evaluated",
            "value": len(precision_scores)
        }
    ])

    output_file = (
        OUTPUT_DIR / "evaluation_results.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print("\nResults saved to:", output_file)
    print("Evaluation completed successfully!")


# ==================================================
# RUN EVALUATION
# ==================================================

if __name__ == "__main__":
    evaluate()

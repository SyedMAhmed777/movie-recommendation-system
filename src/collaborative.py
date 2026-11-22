
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load ratings and movie titles."""

    ratings = pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        DATA_DIR / "u.item",
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1],
        names=["movie_id", "title"]
    )

    return ratings, movies


def recommend_for_user(user_id=1, top_n=10, neighbors=20):
    """Recommend unseen movies using similar users."""

    ratings, movies = load_data()

    # Create a user-movie rating matrix
    user_movie_matrix = ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    )

    if user_id not in user_movie_matrix.index:
        print(f"User {user_id} was not found.")
        return pd.DataFrame()

    # Center each user's ratings around their own average
    user_means = user_movie_matrix.mean(axis=1)
    centered_matrix = user_movie_matrix.sub(
        user_means, axis=0
    )

    # Missing ratings are set to zero only for similarity calculations
    similarity_input = centered_matrix.fillna(0)

    # Compare user rating patterns
    similarities = cosine_similarity(similarity_input)

    similarity_df = pd.DataFrame(
        similarities,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )

    # Find the most similar users
    similar_users = similarity_df.loc[user_id].drop(
        user_id
    ).sort_values(ascending=False)

    similar_users = similar_users[
        similar_users > 0
    ].head(neighbors)

    if similar_users.empty:
        print("No similar users found.")
        return pd.DataFrame()

    # Find movies the target user has not rated
    watched_movies = set(
        ratings.loc[
            ratings["user_id"] == user_id,
            "movie_id"
        ]
    )

    neighbor_ratings = ratings[
        ratings["user_id"].isin(similar_users.index)
        & ~ratings["movie_id"].isin(watched_movies)
    ].copy()

    if neighbor_ratings.empty:
        print("No unseen movies found among neighbors.")
        return pd.DataFrame()

    # Add neighbor similarity and rating averages
    neighbor_ratings["similarity"] = (
        neighbor_ratings["user_id"].map(similar_users)
    )

    neighbor_ratings["user_mean"] = (
        neighbor_ratings["user_id"].map(user_means)
    )

    # Weighted rating deviation from neighbor averages
    neighbor_ratings["weighted_deviation"] = (
        neighbor_ratings["similarity"]
        * (
            neighbor_ratings["rating"]
            - neighbor_ratings["user_mean"]
        )
    )

    neighbor_ratings["weight"] = (
        neighbor_ratings["similarity"].abs()
    )

    # Aggregate neighbor ratings by movie
    scores = neighbor_ratings.groupby("movie_id").agg(
        weighted_sum=("weighted_deviation", "sum"),
        total_weight=("weight", "sum"),
        neighbor_count=("user_id", "nunique")
    ).reset_index()

    # Predict the target user's rating
    scores["predicted_rating"] = (
        user_means.loc[user_id]
        + scores["weighted_sum"] / scores["total_weight"]
    ).clip(1, 5)

    # Require at least two neighbors to support a recommendation
    scores = scores[
        scores["neighbor_count"] >= 2
    ]

    scores = scores.sort_values(
        by=["predicted_rating", "neighbor_count"],
        ascending=[False, False]
    ).head(top_n)

    results = scores.merge(
        movies,
        on="movie_id",
        how="left"
    )

    results["explanation"] = results[
        "neighbor_count"
    ].apply(
        lambda count:
        f"Recommended based on ratings from {count} similar users."
    )

    results = results[
        [
            "movie_id",
            "title",
            "predicted_rating",
            "neighbor_count",
            "explanation"
        ]
    ]

    print("\n" + "=" * 60)
    print("USER-BASED COLLABORATIVE FILTERING")
    print("=" * 60)

    print("Target user:", user_id)
    print("Similar users found:", len(similar_users))

    print("\nTOP RECOMMENDATIONS")
    print(
        results[
            ["title", "predicted_rating", "neighbor_count"]
        ].to_string(index=False)
    )

    return results


if __name__ == "__main__":
    recommendations = recommend_for_user(
        user_id=1,
        top_n=10,
        neighbors=20
    )

    if not recommendations.empty:
        output_file = (
            OUTPUT_DIR / "collaborative_recommendations.csv"
        )

        recommendations.to_csv(
            output_file,
            index=False
        )

        print("\nRecommendations saved to:", output_file)

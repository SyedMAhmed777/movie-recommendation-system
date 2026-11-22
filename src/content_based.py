
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MOVIES_FILE = DATA_DIR / "u.item"

# MovieLens genre columns
GENRES = [
    "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

MOVIE_COLUMNS = [
    "movie_id", "title", "release_date",
    "video_release_date", "imdb_url"
] + GENRES


def load_movies():
    """Load movie information from MovieLens."""
    movies = pd.read_csv(
        MOVIES_FILE,
        sep="|",
        names=MOVIE_COLUMNS,
        encoding="latin-1"
    )

    movies = movies.dropna(
        subset=["movie_id", "title"]
    ).drop_duplicates(subset=["movie_id"])

    return movies.reset_index(drop=True)


def recommend_movies(movie_title, top_n=5):
    """Recommend movies using genre-based cosine similarity."""

    movies = load_movies()

    # Convert movie genres into numeric feature vectors
    genre_matrix = movies[GENRES].fillna(0).astype(float)

    # Calculate similarities between movies
    similarity_matrix = cosine_similarity(genre_matrix)

    # Find the selected movie
    matches = movies[
        movies["title"].str.contains(
            movie_title,
            case=False,
            regex=False,
            na=False
        )
    ]

    if matches.empty:
        print(f"\nMovie not found: {movie_title}")
        return pd.DataFrame()

    selected_index = matches.index[0]
    selected_title = movies.loc[selected_index, "title"]

    # Rank movies by similarity
    similarity_scores = list(
        enumerate(similarity_matrix[selected_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda item: item[1],
        reverse=True
    )

    # Exclude the selected movie itself
    similarity_scores = [
        item for item in similarity_scores
        if item[0] != selected_index
    ][:top_n]

    recommendations = []

    for movie_index, score in similarity_scores:
        movie = movies.iloc[movie_index]

        shared_genres = [
            genre for genre in GENRES
            if movies.loc[selected_index, genre] == 1
            and movie[genre] == 1
        ]

        recommendations.append({
            "movie_id": int(movie["movie_id"]),
            "title": movie["title"],
            "similarity": round(float(score), 4),
            "shared_genres": ", ".join(shared_genres),
            "explanation": (
                "Recommended because it shares these genres: "
                + ", ".join(shared_genres)
                if shared_genres
                else "Recommended based on genre similarity."
            )
        })

    result = pd.DataFrame(recommendations)

    print("\n" + "=" * 60)
    print("CONTENT-BASED MOVIE RECOMMENDATIONS")
    print("=" * 60)
    print("Selected movie:", selected_title)

    print("\nTOP RECOMMENDATIONS")
    print(
        result[
            ["title", "similarity", "shared_genres"]
        ].to_string(index=False)
    )

    return result


if __name__ == "__main__":
    recommendations = recommend_movies(
        "Star Wars (1977)",
        top_n=5
    )

    if not recommendations.empty:
        output_dir = BASE_DIR / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "content_recommendations.csv"
        recommendations.to_csv(output_file, index=False)

        print("\nRecommendations saved to:", output_file)

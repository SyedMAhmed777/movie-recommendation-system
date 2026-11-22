from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RATINGS_FILE = DATA_DIR / "u.data"
MOVIES_FILE = DATA_DIR / "u.item"

# Load ratings dataset
ratings = pd.read_csv(
    RATINGS_FILE,
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# MovieLens 100K has 19 genre indicator columns
genres = [
    "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

movie_columns = [
    "movie_id", "title", "release_date",
    "video_release_date", "imdb_url"
] + genres

# Load movie information
movies = pd.read_csv(
    MOVIES_FILE,
    sep="|",
    names=movie_columns,
    encoding="latin-1"
)

print("=" * 50)
print("MOVIE RECOMMENDATION SYSTEM")
print("=" * 50)

print("\nDATASET SUMMARY")
print("-" * 50)

print("Total ratings:", len(ratings))
print("Total users:", ratings["user_id"].nunique())
print("Total movies:", movies["movie_id"].nunique())

print("\nFIRST 5 RATINGS")
print(ratings.head())

print("\nFIRST 5 MOVIES")
print(movies[["movie_id", "title"]].head())

print("\nRATING DISTRIBUTION")
print(ratings["rating"].value_counts().sort_index())

print("\nDataset loaded successfully!")


# ==================================================
# STEP 6: DATA CLEANING AND EXPLORATORY ANALYSIS
# ==================================================

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 50)
print("STEP 6: DATA CLEANING AND ANALYSIS")
print("=" * 50)

# 1. Check for missing values
print("\nMISSING VALUES IN RATINGS")
print(ratings.isnull().sum())

print("\nMISSING VALUES IN IMPORTANT MOVIE COLUMNS")
print(movies[["movie_id", "title"]].isnull().sum())

# 2. Check for duplicate user-movie ratings
duplicate_ratings = ratings.duplicated(
    subset=["user_id", "movie_id"]
).sum()

print("\nDuplicate user-movie ratings:", duplicate_ratings)

# 3. Remove invalid and duplicate records
ratings_clean = ratings.dropna(
    subset=["user_id", "movie_id", "rating"]
).copy()

ratings_clean = ratings_clean[
    ratings_clean["rating"].between(1, 5)
]

ratings_clean = ratings_clean.drop_duplicates(
    subset=["user_id", "movie_id"],
    keep="last"
)

movies_clean = movies.dropna(
    subset=["movie_id", "title"]
).drop_duplicates(
    subset=["movie_id"]
).copy()

# 4. Combine ratings with movie titles
movie_ratings = ratings_clean.merge(
    movies_clean[["movie_id", "title"]],
    on="movie_id",
    how="inner"
)

print("\nCleaned ratings:", len(ratings_clean))
print("Matched ratings:", len(movie_ratings))

# 5. Calculate average rating and rating count
movie_stats = movie_ratings.groupby(
    ["movie_id", "title"]
).agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

# 6. Find the most rated movies
most_rated = movie_stats.sort_values(
    by="rating_count",
    ascending=False
).head(10)

print("\nTOP 10 MOST RATED MOVIES")
print(
    most_rated[
        ["title", "rating_count", "average_rating"]
    ].to_string(index=False)
)

# 7. Find highly rated movies with enough reviews
top_rated = movie_stats[
    movie_stats["rating_count"] >= 50
].sort_values(
    by="average_rating",
    ascending=False
).head(10)

print("\nTOP 10 HIGHEST RATED MOVIES (MINIMUM 50 RATINGS)")
print(
    top_rated[
        ["title", "average_rating", "rating_count"]
    ].to_string(index=False)
)

# 8. Analyze user activity
user_activity = ratings_clean.groupby(
    "user_id"
).agg(
    total_ratings=("rating", "count"),
    average_rating=("rating", "mean")
).reset_index()

print("\nUSER ACTIVITY SUMMARY")
print(user_activity["total_ratings"].describe())

# 9. Save analysis results
most_rated.to_csv(
    OUTPUT_DIR / "most_rated_movies.csv",
    index=False
)

top_rated.to_csv(
    OUTPUT_DIR / "top_rated_movies.csv",
    index=False
)

movie_stats.to_csv(
    OUTPUT_DIR / "movie_statistics.csv",
    index=False
)

user_activity.to_csv(
    OUTPUT_DIR / "user_activity.csv",
    index=False
)

print("\nAnalysis files saved in outputs folder.")
print("Step 6 completed successfully!")

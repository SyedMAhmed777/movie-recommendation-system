
from pathlib import Path
import sys
from html import escape

import pandas as pd
import streamlit as st

# ==================================================
# PATHS AND IMPORTS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

sys.path.insert(0, str(SRC_DIR))

from content_based import load_movies, recommend_movies
from collaborative import recommend_for_user


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# MODERN DARK THEME
# ==================================================

st.markdown("""
<style>
    :root {
        --bg: #0B0D14;
        --surface: #141824;
        --border: #272D3B;
        --text: #F4F5FA;
        --muted: #959CAD;
        --accent: #8B7CF6;
    }

    .stApp {
        background: #0B0D14;
        color: #F4F5FA;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: #10131D;
        border-right: 1px solid #272D3B;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    h1, h2, h3 {
        color: #F4F5FA !important;
        letter-spacing: -0.5px;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 750 !important;
    }

    h3 {
        font-weight: 650 !important;
    }

    p {
        color: #A8AFBF;
    }

    div.stButton > button[kind="primary"] {
        background: #7968ED;
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        min-height: 46px;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #9183FA;
        color: white;
    }

    div.stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #353C4D;
        background: #191E2B;
        color: #E4E6F0;
    }

    div[data-testid="stMetric"] {
        background: #141824;
        border: 1px solid #272D3B;
        padding: 16px 20px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #959CAD;
    }

    div[data-testid="stMetricValue"] {
        color: #F4F5FA;
    }

    div[data-baseweb="select"] > div {
        background: #171B28;
        border: 1px solid #32394A;
        border-radius: 10px;
    }

    .brand {
        font-size: 23px;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #F4F5FA;
        margin-bottom: 4px;
    }

    .brand span {
        color: #9B8CFF;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #858D9E;
        margin-bottom: 25px;
    }

    .page-subtitle {
        font-size: 15px;
        color: #959CAD;
        margin-top: -8px;
        margin-bottom: 24px;
    }

    .section-heading {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.4px;
        color: #F4F5FA;
        margin-bottom: 5px;
    }

    .section-description {
        font-size: 14px;
        color: #959CAD;
        margin-bottom: 20px;
    }

    .movie-card {
        background: #151A27;
        border: 1px solid #2A3040;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 14px;
        min-height: 155px;
    }

    .movie-card:hover {
        border-color: #5D568D;
    }

    .movie-title {
        color: #F4F5FA;
        font-size: 16px;
        font-weight: 700;
        line-height: 1.45;
        margin-bottom: 12px;
    }

    .movie-score {
        color: #A99DFF;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .movie-description {
        color: #959CAD;
        font-size: 13px;
        line-height: 1.6;
    }

    .empty-state {
        background: #141824;
        border: 1px dashed #343B4C;
        border-radius: 14px;
        padding: 35px;
        text-align: center;
        color: #959CAD;
    }

    hr {
        border-color: #272D3B !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem;
        }

        h1 {
            font-size: 1.8rem !important;
        }

        .movie-card {
            min-height: auto;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def get_movies():
    return load_movies()


@st.cache_data
def get_ratings():
    return pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )


@st.cache_data
def get_evaluation():
    path = OUTPUT_DIR / "evaluation_results.csv"

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


# ==================================================
# RECOMMENDATION CARDS
# ==================================================

def display_recommendations(results, score_column, score_label):

    if results.empty:
        st.warning("No recommendations found.")
        return

    columns = st.columns(2, gap="medium")

    for index, (_, movie) in enumerate(results.iterrows()):

        title = escape(str(movie["title"]))
        explanation = escape(str(movie["explanation"]))
        score = float(movie[score_column])

        with columns[index % 2]:
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">{title}</div>
                    <div class="movie-score">
                        {escape(score_label)}: {score:.2f}
                    </div>
                    <div class="movie-description">
                        {explanation}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">Cine<span>Match</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">Movie discovery, made smarter.</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Explore",
        [
            "Discover Movies",
            "For You",
            "Model Insights"
        ],
        label_visibility="collapsed"
    )


# ==================================================
# HEADER
# ==================================================

st.title("Discover your next favorite movie")

st.markdown(
    '<div class="page-subtitle">'
    'Smart recommendations powered by movie genres and viewing preferences.'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# DATASET METRICS
# ==================================================

movies = get_movies()
ratings = get_ratings()

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.metric("Movies", f"{movies['movie_id'].nunique():,}")

with col2:
    st.metric("Users", f"{ratings['user_id'].nunique():,}")

with col3:
    st.metric("Ratings", f"{len(ratings):,}")

st.write("")


# ==================================================
# PAGE 1: CONTENT-BASED RECOMMENDATIONS
# ==================================================

if page == "Discover Movies":

    st.markdown(
        '<div class="section-heading">Explore similar movies</div>'
        '<div class="section-description">'
        'Pick a movie you enjoy and discover titles with similar genres.'
        '</div>',
        unsafe_allow_html=True
    )

    titles = sorted(
        movies["title"].dropna().unique().tolist()
    )

    default_movie = "Star Wars (1977)"

    default_index = (
        titles.index(default_movie)
        if default_movie in titles
        else 0
    )

    selected_movie = st.selectbox(
        "Choose a movie",
        titles,
        index=default_index
    )

    top_n = st.slider(
        "Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    generate = st.button(
        "Find similar movies",
        type="primary",
        use_container_width=True
    )

    if generate:

        with st.spinner("Finding similar movies..."):
            results = recommend_movies(
                selected_movie,
                top_n=top_n
            )

        st.session_state["content_results"] = results
        st.session_state["content_movie"] = selected_movie

    if "content_results" in st.session_state:

        results = st.session_state["content_results"]

        st.write("")
        st.markdown("### Recommended for you")

        display_recommendations(
            results,
            "similarity",
            "Genre similarity"
        )

        if not results.empty:
            st.download_button(
                "Export recommendations",
                data=results.to_csv(index=False),
                file_name="content_recommendations.csv",
                mime="text/csv"
            )

    else:
        st.write("")
        st.markdown(
            '<div class="empty-state">'
            'Select a movie and click "Find similar movies" to get started.'
            '</div>',
            unsafe_allow_html=True
        )


# ==================================================
# PAGE 2: COLLABORATIVE FILTERING
# ==================================================

elif page == "For You":

    st.markdown(
        '<div class="section-heading">Personalized picks</div>'
        '<div class="section-description">'
        'Find movies based on the preferences of similar users.'
        '</div>',
        unsafe_allow_html=True
    )

    user_id = st.number_input(
        "MovieLens User ID",
        min_value=1,
        max_value=int(ratings["user_id"].max()),
        value=1,
        step=1
    )

    col_a, col_b = st.columns(2)

    with col_a:
        top_n = st.slider(
            "Recommendations",
            min_value=3,
            max_value=15,
            value=10,
            key="collab_count"
        )

    with col_b:
        neighbors = st.slider(
            "Similar users",
            min_value=5,
            max_value=50,
            value=20
        )

    generate = st.button(
        "Find my recommendations",
        type="primary",
        use_container_width=True
    )

    if generate:

        with st.spinner("Finding personalized recommendations..."):
            results = recommend_for_user(
                user_id=int(user_id),
                top_n=top_n,
                neighbors=neighbors
            )

        st.session_state["collab_results"] = results

    if "collab_results" in st.session_state:

        results = st.session_state["collab_results"]

        st.write("")
        st.markdown("### Recommended for you")

        display_recommendations(
            results,
            "predicted_rating",
            "Predicted rating"
        )

        if not results.empty:
            st.download_button(
                "Export recommendations",
                data=results.to_csv(index=False),
                file_name="collaborative_recommendations.csv",
                mime="text/csv"
            )

    else:
        st.write("")
        st.markdown(
            '<div class="empty-state">'
            'Choose a user ID to discover personalized recommendations.'
            '</div>',
            unsafe_allow_html=True
        )


# ==================================================
# PAGE 3: MODEL INSIGHTS
# ==================================================

elif page == "Model Insights":

    st.markdown(
        '<div class="section-heading">Model performance</div>'
        '<div class="section-description">'
        'How well does collaborative filtering predict movie ratings?'
        '</div>',
        unsafe_allow_html=True
    )

    evaluation = get_evaluation()

    if evaluation.empty:
        st.warning(
            "Evaluation results are missing. "
            "Run python src/evaluate.py first."
        )

    else:

        values = dict(
            zip(
                evaluation["metric"],
                evaluation["value"]
            )
        )

        baseline = float(
            values.get("Baseline_RMSE", 0)
        )

        collaborative = float(
            values.get("Collaborative_Filtering_RMSE", 0)
        )

        improvement = float(
            values.get("RMSE_Improvement_Percent", 0)
        )

        precision = float(
            values.get("Precision@10_heldout", 0)
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Baseline RMSE",
            f"{baseline:.4f}"
        )

        c2.metric(
            "Collaborative RMSE",
            f"{collaborative:.4f}"
        )

        c3.metric(
            "RMSE improvement",
            f"{improvement:.2f}%"
        )

        st.write("")

        st.markdown("### Prediction accuracy")

        comparison = pd.DataFrame({
            "Model": [
                "User-average baseline",
                "Collaborative filtering"
            ],
            "RMSE": [
                baseline,
                collaborative
            ]
        })

        st.bar_chart(
            comparison.set_index("Model"),
            color="#8B7CF6"
        )

        st.write("")

        c4, c5 = st.columns(2)

        c4.metric(
            "Held-out Precision@10",
            f"{precision * 100:.2f}%"
        )

        c5.metric(
            "Users evaluated",
            int(values.get("Users_Evaluated", 0))
        )

        st.caption(
            "Lower RMSE is better. Precision@10 is measured "
            "over held-out candidate movies rather than "
            "the full movie catalog."
        )

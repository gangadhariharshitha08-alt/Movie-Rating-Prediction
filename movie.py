
import streamlit as st
import pandas as pd
import joblib
import base64

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Movie Rating Prediction",
    page_icon="🎬",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_model():
    return joblib.load("movie_rating_model.pkl")

# ==========================
# BACKGROUND IMAGE FUNCTION
# ==========================

def set_bg(image_file):

    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .main-title {{
            text-align: center;
            color: white;
            font-size: 60px;
            font-weight: bold;
            margin-top: 120px;
            text-shadow: 3px 3px 12px black;
        }}

        .quote {{
            text-align: center;
            color: white;
            font-size: 25px;
            margin-top: 20px;
            text-shadow: 2px 2px 8px black;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# ==========================
# SESSION STATE
# ==========================

if "page" not in st.session_state:
    st.session_state.page = "home"

# ==========================
# HOME PAGE
# ==========================

if st.session_state.page == "home":

    set_bg("theater.jpg")

    st.markdown(
        """
        <div class="main-title">
        🎬 Movie Rating Prediction System
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="quote">
        Lights. Camera. Prediction.<br><br>
        Discover how your movie might be rated using Machine Learning.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3,1,3])

    with col2:
        if st.button("start prediction"):
            st.session_state.page = "predict"
            st.rerun()

# ==========================
# PREDICTION PAGE
# ==========================

else:

    model = load_model()

    st.title("🎬 Movie Rating Predictor")

    st.write(
        "Enter movie details and predict the IMDb rating."
    )

    genre = st.text_input(
        "Genre",
        placeholder="Action, Drama, Comedy..."
    )

    director = st.text_input(
        "Director",
        placeholder="Director Name"
    )

    year = st.number_input(
        "Release Year",
        min_value=1900,
        max_value=2035,
        value=2024
    )

    duration = st.number_input(
        "Duration (Minutes)",
        min_value=1,
        value=120
    )

    votes = st.number_input(
        "Votes",
        min_value=0,
        value=1000
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⭐ Predict Rating"):

        input_data = pd.DataFrame({
            "Genre": [genre],
            "Director": [director],
            "Year": [year],
            "Duration": [duration],
            "Votes": [votes]
        })

        prediction = model.predict(input_data)

        st.success(
            f"⭐ Predicted IMDb Rating: {prediction[0]:.2f}"
        )

    st.markdown("---")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

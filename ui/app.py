import streamlit as st
from pathlib import Path

PAGE_ROUTES = {
    "home": "home",
    "experiments": "experiments-explorer",
    "leaderboard": "model-leaderboard",
    "forecast": "forecast-horizon",
}

st.set_page_config(
    page_title="EX-Dash: Exchange rate forecasting dashboard",
    page_icon="📈",
    layout="wide",
)


def load_css():
    css_path = Path(__file__).parent / "styles.css"
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_home():
    st.markdown(
        """
        <div class="market-bg" aria-hidden="true">
            <svg class="market-wave center-wave" viewBox="0 0 1400 220" preserveAspectRatio="none">
                <polyline points="0,120 120,88 240,142 360,84 490,136 620,78 760,146 900,86 1040,138 1180,82 1310,128 1400,96"></polyline>
                <circle cx="360" cy="84" r="4"></circle><circle cx="760" cy="146" r="4"></circle><circle cx="1180" cy="82" r="4"></circle>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h1 class="hero-title">EX-Dash: Exchange rate forecasting dashboard</h1>
        <div class="home-card">
            <p>
                This project was a classic time-series forecasting project focused on exchange-rate prediction.
                The methodology combines data preparation, rolling-window evaluation, and comparative benchmarking
                across transformer-based models including Autoformer, Informer, and baseline Transformer variants.
                The strongest models showed more stable multi-step forecasts and tighter uncertainty behavior
                across multiple prediction horizons.
            </p>
            <p class="cta">Check the results out yourself.</p>
            <div class="home-links">
                <a href="/experiments-explorer" target="_self">Experiments Explorer</a>
                <a href="/model-leaderboard" target="_self">Model Leaderboard</a>
                <a href="/forecast-horizon" target="_self">Forecast Horizon</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


load_css()

if hasattr(st, "navigation") and hasattr(st, "Page"):
    pages = [
        st.Page(render_home, title="Home", url_path=PAGE_ROUTES["home"]),
        st.Page(
            "pages/1_experiments_explorer.py",
            title="Experiments Explorer",
            url_path=PAGE_ROUTES["experiments"],
        ),
        st.Page(
            "pages/2_model_leaderboard.py",
            title="Model Leaderboard",
            url_path=PAGE_ROUTES["leaderboard"],
        ),
        st.Page(
            "pages/3_forecast_horizon.py",
            title="Forecast Horizon",
            url_path=PAGE_ROUTES["forecast"],
        ),
    ]
    nav = st.navigation(pages, position="sidebar")
    nav.run()
else:
    render_home()

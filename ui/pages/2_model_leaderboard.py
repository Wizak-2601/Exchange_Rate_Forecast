import streamlit as st
from utils.load_data import load_experiments
import plotly.express as px
st.title("Model Leaderboard")

experiments = load_experiments()

best_model = experiments.loc[experiments["model_smape"].idxmin()]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Best Model",
        value=best_model["model_type"]
    )

with col2:
    st.metric(
        label="Best sMAPE",
        value=f"{best_model['model_smape']:.4f}"
    )

with col3:
    st.metric(
        label="Experiments Run",
        value=len(experiments)
    )


fig = px.bar(
    experiments,
    x="model_type",
    y="model_smape",
    color="model_type",
    template="plotly_dark"
)

fig.update_layout(
    title="Model Performance Comparison",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

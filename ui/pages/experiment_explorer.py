import streamlit as st
import plotly.express as px
from utils.load_data import load_experiments

st.title("Architecture Experiment Explorer")

experiments = load_experiments()

model = st.selectbox(
    "Select Model",
    experiments["model_type"].unique()
)

filtered = experiments[
    experiments["model_type"] == model
]

fig = px.scatter(
    filtered,
    x="seq_len",
    y="model_smape",
    color="pred_len",
    size="n_heads",
    hover_data=["dropout","d_model"],
    title="Sequence Length vs Forecast Error"
)

fig.update_layout(template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

fig2 = px.line(
    experiments,
    x="pred_len",
    y="model_smape",
    color="model_type",
    markers=True,
    title="Forecast Horizon vs Error"
)

fig2.update_layout(template="plotly_dark")

st.plotly_chart(fig2, use_container_width=True)

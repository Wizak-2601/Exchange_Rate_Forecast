import streamlit as st
import plotly.express as px
from utils.load_data import load_final

st.title("Model Leaderboard")

final_results = load_final()

st.subheader("Final Test Performance")

st.dataframe(
    final_results.sort_values("test_smape"),
    use_container_width=True
)

fig = px.bar(
    final_results,
    x="model_type",
    y="test_smape",
    color="model_type",
    text="test_smape",
    title="Final Model Performance (Test sMAPE)"
)

fig.update_layout(
    template="plotly_dark",
    title_x=0.5
)

st.plotly_chart(fig, use_container_width=True)

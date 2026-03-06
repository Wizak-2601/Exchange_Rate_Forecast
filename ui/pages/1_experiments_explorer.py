import streamlit as st
from utils.load_data import load_experiments
import plotly.express as px
st.title("Experiments Explorer")

experiments = load_experiments()

st.dataframe(
    experiments,
    use_container_width=True
)
model_filter = st.selectbox(
    "Select Model",
    experiments["model_type"].unique()
)

filtered = experiments[
    experiments["model_type"] == model_filter
]

st.dataframe(filtered)


heatmap = experiments.pivot_table(
    index="seq_len",
    columns="pred_len",
    values="model_smape"
)

fig = px.imshow(
    heatmap,
    color_continuous_scale="viridis",
    template="plotly_dark"
)

st.plotly_chart(fig)

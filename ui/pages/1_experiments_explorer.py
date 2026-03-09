import streamlit as st
import pandas as pd
import plotly.express as px
from utils.load_data import load_baselines, load_experiments

st.title("Experiments Explorer")

experiments = load_experiments().copy()
baselines = load_baselines().copy()

# Include transformer runs from baseline results so they appear in the table/charts.
transformer_rows = baselines[
    baselines["model_type"].str.lower() == "transformer"
].copy()

common_cols = experiments.columns.tolist()
for col in common_cols:
    if col not in transformer_rows.columns:
        transformer_rows[col] = pd.NA
transformer_rows = transformer_rows[common_cols]

experiments_all = pd.concat(
    [experiments, transformer_rows], ignore_index=True
).dropna(subset=["model_type", "seq_len", "pred_len", "model_smape"])

experiments_all["seq_len"] = experiments_all["seq_len"].astype(int)
experiments_all["pred_len"] = experiments_all["pred_len"].astype(int)

st.subheader("Experiment Results Table")

st.dataframe(
    experiments_all.sort_values(
        ["model_type", "pred_len", "seq_len", "model_smape"]
    ),
    use_container_width=True
)

model_filter = st.selectbox(
    "Select Model",
    sorted(experiments_all["model_type"].unique())
)

filtered = experiments_all[
    experiments_all["model_type"] == model_filter
]

st.subheader("Filtered Table")
st.dataframe(filtered)

st.subheader("heatmaps")
heatmap_choice = st.selectbox(
    "Heatmap Model",
    ["Best Model"] + sorted(experiments_all["model_type"].unique())
)

if heatmap_choice == "Best Model":
    selected_model = experiments_all.loc[
        experiments_all["model_smape"].idxmin(), "model_type"
    ]
else:
    selected_model = heatmap_choice

heatmap_source = experiments_all[
    experiments_all["model_type"] == selected_model
]

heatmap = heatmap_source.pivot_table(
    index="seq_len",
    columns="pred_len",
    values="model_smape",
    aggfunc="min"
)

fig = px.imshow(
    heatmap,
    color_continuous_scale="viridis",
    template="plotly_dark",
    labels={"x": "Prediction Horizon", "y": "Sequence Length", "color": "sMAPE"},
    title=f"{selected_model.capitalize()} Heatmap (lower is better)"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("sMAPE vs Prediction Horizon (All Models)")
horizon_plot_data = (
    experiments_all
    .groupby(["model_type", "pred_len"], as_index=False)["model_smape"]
    .min()
    .sort_values(["model_type", "pred_len"])
)

line_fig = px.line(
    horizon_plot_data,
    x="pred_len",
    y="model_smape",
    color="model_type",
    markers=True,
    template="plotly_dark",
    labels={
        "pred_len": "Prediction Horizon",
        "model_smape": "Best sMAPE",
        "model_type": "Model",
    },
    title="Best sMAPE by Prediction Horizon"
)

line_fig.update_layout(hovermode="x unified")
st.plotly_chart(line_fig, use_container_width=True)

import pandas as pd
import streamlit as st
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]

def load_experiments():
    path=PROJECT_ROOT/"notebooks"/"results"/"experiment_results.csv"
    return pd.read_csv(path)

def load_baselines():
    path=PROJECT_ROOT/"notebooks"/"results"/"baseline_results.csv"
    return pd.read_csv(path)
def load_final():
    path=PROJECT_ROOT/"notebooks"/"results"/"final_model_comparison_table.csv"
    return pd.read_csv(path)

@st.cache_data
def load_forecast(horizon=199, max_points=2000):

    autoformer=pd.read_csv(PROJECT_ROOT/"notebooks"/"results"/"autoformer_forecast.csv")
    informer=pd.read_csv(PROJECT_ROOT/"notebooks"/"results"/"informer_forecast.csv")
    transformer=pd.read_csv(PROJECT_ROOT/"notebooks"/"results"/"transformer_forecast.csv")
    df = pd.DataFrame({
    "time": autoformer["time"],
    "actual": autoformer["actual"],
    "autoformer": autoformer["autoformer"],
    "informer": informer["informer"],
    "transformer": transformer["transformer"],
    "autoformer_upper": autoformer["autoformer_upper"],
    "autoformer_lower": autoformer["autoformer_lower"],
    "informer_upper":informer["informer_upper"],
    "informer_lower":informer["informer_lower"],
    "transformer_upper":transformer["transformer_upper"],
    "transformer_lower":transformer["transformer_lower"]
})

    # Apply horizon-aware filtering first so toggle has real impact.
    # Use a fixed visualization cycle (366) so 24/96/199/366 are distinct views.
    horizon = int(horizon)
    cycle_window = 366
    if horizon <= cycle_window:
        df = df[df["time"] % cycle_window < horizon]

    # Downsample for UI performance.
    if len(df) > max_points:
        step = len(df) // max_points
        df = df.iloc[::step]

    df = df.reset_index(drop=True)
    return df

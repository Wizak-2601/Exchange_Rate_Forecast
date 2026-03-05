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
def load_forecast():

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
    "informer_upper":informer["informer_lower"],
    "informer_lower":informer["informer_lower"],
    "transformer_upper":transformer["transformer_upper"],
    "transformer_lower":transformer["transformer_lower"]
})
# 🔹 Downsample for UI performance
    max_points = 2000

    if len(df) > max_points:
        step = len(df) // max_points
        df = df.iloc[::step]
    return df

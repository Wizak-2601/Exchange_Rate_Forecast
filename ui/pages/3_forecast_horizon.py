import streamlit as st
import plotly.graph_objects as go
from utils.load_data import load_forecast

st.title("Forecast Horizon")

horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    ["24", "96", "199", "366"]
)

models = ["autoformer","informer","transformer"]

selected_models = st.sidebar.multiselect(
    "Select Models",
    models,
    default=["autoformer"]
)

forecast = load_forecast(horizon=int(horizon))

fig = go.Figure()

fig.add_trace(
    go.Scattergl(
        x=forecast["time"],
        y=forecast["actual"],
        mode="lines",
        name="Actual",
        line=dict(width=3,color="white")
    )
)

for model in selected_models:

    fig.add_trace(
        go.Scattergl(
            x=forecast["time"],
            y=forecast[model],
            mode="lines",
            name=model.capitalize(),
            line=dict(width=2)
        )
    )

    upper = f"{model}_upper"
    lower = f"{model}_lower"

    if upper in forecast.columns:

        fig.add_trace(
            go.Scattergl(
                x=forecast["time"],
                y=forecast[upper],
                mode="lines",
                line=dict(width=0),
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast["time"],
                y=forecast[lower],
                fill="tonexty",
                mode="lines",
                name=f"{model} uncertainty",
                opacity=0.2
            )
        )

fig.update_layout(
    template="plotly_dark",
    title=f"Forecast vs Actual (Horizon {horizon})",
    xaxis_title="Time",
    yaxis_title="Exchange Rate",
    hovermode="x unified",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Uncertainty Calibration")

calibration_model = st.selectbox(
    "Calibration Model",
    models,
    index=models.index(selected_models[0]) if selected_models else 0
)

upper_col = f"{calibration_model}_upper"
lower_col = f"{calibration_model}_lower"

calibration_df = forecast.dropna(
    subset=["actual", calibration_model, upper_col, lower_col]
).copy()

calibration_df["error"] = abs(
    calibration_df["actual"] - calibration_df[calibration_model]
)
calibration_df["interval_width"] = (
    calibration_df[upper_col] - calibration_df[lower_col]
)

fig2 = go.Figure()

fig2.add_trace(
    go.Scattergl(
        x=calibration_df["interval_width"],
        y=calibration_df["error"],
        mode="markers",
        name=calibration_model.capitalize()
    )
)

fig2.update_layout(
    template="plotly_dark",
    xaxis_title="Prediction Interval Width",
    yaxis_title="Absolute Error",
    title=f"Uncertainty Calibration: {calibration_model.capitalize()}"
)

st.plotly_chart(fig2, use_container_width=True)

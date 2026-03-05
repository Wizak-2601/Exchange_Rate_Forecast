import streamlit as st
import plotly.graph_objects as go
from utils.load_data import load_forecast

st.title("Forecast Visualization")

forecast = load_forecast()




horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    ["24", "96", "199"]
)

models = ["autoformer","informer","transformer"]

selected_models = st.sidebar.multiselect(
    "Select Models",
    models,
    default=["autoformer"]
)

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
    title="Forecast vs Actual",
    xaxis_title="Time",
    yaxis_title="Exchange Rate",
    hovermode="x unified",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Uncertainty Calibration")

forecast["error"] = abs(
    forecast["actual"] - forecast["autoformer"]
)

fig2 = go.Figure()

fig2.add_trace(
    go.Scattergl(
        x=forecast["autoformer_upper"] - forecast["autoformer_lower"],
        y=forecast["error"],
        mode="markers"
    )
)

fig2.update_layout(
    template="plotly_dark",
    xaxis_title="Prediction Interval Width",
    yaxis_title="Absolute Error"
)

st.plotly_chart(fig2, use_container_width=True)

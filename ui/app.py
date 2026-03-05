import streamlit as st

st.set_page_config(
    page_title="Exchange Rate Forecast Dashboard",
    layout="wide"
)

st.title("Exchange Rate Forecasting Dashboard")

st.markdown("""
This dashboard visualizes forecasting experiments using:

• Transformer  
• Informer  
• Autoformer  

Navigate using the sidebar to explore experiments and forecasts.
""")

st.info("Use the sidebar to explore different sections of the project.")

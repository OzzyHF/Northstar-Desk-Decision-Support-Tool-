import streamlit as st

st.set_page_config(layout="wide")


THEME_COLOUR = "#707275"


st.markdown(f"<h1 style='color:{THEME_COLOUR};'>Hackathon</h1>", unsafe_allow_html=True)
st.markdown(f"<h2 style='color:{THEME_COLOUR};'>Model Card</h2>", unsafe_allow_html=True)
st.write('This page provides key model information from our model')

model_type = 'Logistic Regression'
mse = 0.00
mae = 0.00
r2 = 0.00

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model Type", model_type)
col2.metric("Mean Squared Error", f"{mse:.1f}%")
col3.metric("Mean Absolute Error", f"{mae:.1f}%")
col4.metric("R2", f"{r2:.1f}%")


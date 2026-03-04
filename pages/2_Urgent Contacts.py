import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")


THEME_COLOUR = "#597AAC"



path = "results/predictions.csv"
data = pd.read_csv(path)


data["predicted_priority"] = data["predicted_priority"].astype(int)
st.markdown(f"<h1 style='color:{THEME_COLOUR};'>Hackathon - Top 10 Urgent Contacts</h1>", unsafe_allow_html=True)
st.write('This page provides a list of those enquiriers that require urgent contact.') 

top10 = data.sort_values("predicted_priority", ascending=False).head(10)


st.dataframe(top10)
st.table()
st.markdown(
    "[View Project on GitHub](https://github.com/OzzyHF/Northstar-Desk-Decision-Support-Tool-)",
    unsafe_allow_html=True
)

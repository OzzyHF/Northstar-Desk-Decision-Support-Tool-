import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")


THEME_COLOUR = "#707275"

path = r"C:\Users\barnyrumbold\OneDrive - Kidney Research UK\Desktop\Hackathon\Northstar-Desk-Decision-Support-Tool-\Data\clean.csv"
data = pd.read_csv(path)

st.markdown(f"<h1 style='color:{THEME_COLOUR};'>Hackathon</h1>", unsafe_allow_html=True)
st.write('This page provides a list of those enquiriers that require urgent contact.') 

top10 = data.sort_values("priority", ascending=False).head(10)

st.markdown(f"<h2 style='color:{THEME_COLOUR};'>Top 10 Urgent Contacts</h2>", unsafe_allow_html=True)
st.dataframe(top10)
st.table()

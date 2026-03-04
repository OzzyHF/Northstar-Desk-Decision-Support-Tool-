import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")


THEME_COLOUR = "#597AAC"

path = r"C:\Users\barnyrumbold\OneDrive - Kidney Research UK\Desktop\Hackathon\Northstar-Desk-Decision-Support-Tool-\Data\clean.csv"
data = pd.read_csv(path)

st.markdown(f"<h1 style='color:{THEME_COLOUR};'>Hackathon - Top 10 Urgent Contacts</h1>", unsafe_allow_html=True)
st.write('This page provides a list of those enquiriers that require urgent contact.') 

top10 = data.sort_values("priority", ascending=False).head(10)


st.dataframe(top10)
st.table()
st.markdown(
    "[View Project on GitHub](https://github.com/OzzyHF/Northstar-Desk-Decision-Support-Tool-)",
    unsafe_allow_html=True
)

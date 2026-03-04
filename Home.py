import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
st.set_page_config(layout="wide")
# Load your logo
THEME_COLOUR = "#597AAC"
st.markdown(f"<h1 style='color:{THEME_COLOUR};'>Hackathon</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.write('This app provides a decision support tool for Northstar Desk a UK-based subscription software company.') 
    logo = Image.open("images/ChatGPT Image Mar 3, 2026, 12_31_48 PM.png")  
    st.image(logo, width=450)  
with col2:

    
    st.markdown(f"<h2 style='color:{THEME_COLOUR};'>Problem</h2>", unsafe_allow_html=True)
    st.write('At Northstar Desk, over 100 new cases arrive every month with little indication of priority levels. The team must quickly judge urgency from short summaries, which can lead to delays or missed high-risk issues.')
    st.markdown(f"<h2 style='color:{THEME_COLOUR};'>Solution</h2>", unsafe_allow_html=True)
    st.write('We have developed a simple interactive tool that supports triage decisions. When an analyst open the tool, it provides a list with potential contacts with those at priority level at the top. The analyst makes the final decision — now backed by historical patterns.')
    st.markdown(f"<h2 style='color:{THEME_COLOUR};'>Impact</h2>", unsafe_allow_html=True)
    st.write('Faster, more consistent triage, better workload focus, and reduced risk of urgent cases being overlooked.')

    st.markdown(f"<h2 style='color:{THEME_COLOUR};'>Pages</h2>", unsafe_allow_html=True)
    st.write('Page 1 - Shows an overview of analytics within the data set.')
    st.write('Page 2 - Shows a table with those individuals most important to contact.')
    st.write('Page 3 - Shows the model card, an overview of the machine learning model and performance metrics that influence the score.')
    st.write('Page 4 - Shows an overview of the data cleaning steps we followed prior to training the model.')

st.markdown(
    "[View Project on GitHub](https://github.com/OzzyHF/Northstar-Desk-Decision-Support-Tool-)",
    unsafe_allow_html=True
)


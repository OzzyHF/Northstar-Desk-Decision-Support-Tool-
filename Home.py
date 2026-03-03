import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# Load your logo
THEME_COLOUR = "#707275"
st.markdown(f"<h1 style='color:{THEME_COLOUR};'>Hackathon</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    logo = Image.open(r"C:\Users\barnyrumbold\OneDrive - Kidney Research UK\Desktop\Hackathon\Northstar-Desk-Decision-Support-Tool-\ChatGPT Image Mar 3, 2026, 12_31_48 PM.png")  
    st.image(logo, width=450)  
with col2:

    st.write('This app provides a decision support tool for Northstar Desk a UK-based subscription software company.') 
    st.write('The app identifies uses machine learning models to identify which customer interactions are most urgent to support timely customer service.') 
    st.write('A score is then created from the information provided and then a list is provided of those individuals with the highest score (the most urgent)') 
    st.write('Page 1 - Shows an overview of analytics within the data set.')
    st.write('Page 2 - Shows a table with those individuals most important to contact.')
    st.write('Page 3 - Shows the model card, an overview of the machine learning model and performance metrics that influence the score.')
    st.write('Page 4 - Shows an overview of the data cleaning steps we followed prior to training the model.')




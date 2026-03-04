import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(layout="wide")


THEME_COLOUR = "#597AAC"


st.markdown(f"<h1 style='color:{THEME_COLOUR};'>Hackathon - Data Cleaning Steps</h1>", unsafe_allow_html=True)
st.write('This page provides an overview of the data, and data manipulation prior to training.')


col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<h2 style='color:{THEME_COLOUR};'>1. Correlation Matrix</h2>", unsafe_allow_html=True)
    path = "Data/clean.csv"
    d = pd.read_csv(path)
    num_df = d.select_dtypes(include="number")
    corr = num_df.corr(method="pearson")
    corr.round(2)
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        annot_kws={"size": 3, "color": THEME_COLOUR},  # annotation numbers themed
        ax=ax,
    )

    # Title styling
    ax.set_title(
        "Correlation Matrix (Numeric Features)",
        fontsize=9,
        color=THEME_COLOUR
    )

    # Tick labels styling
    ax.tick_params(axis='both', labelsize=6, colors=THEME_COLOUR)

    # Rotate x labels slightly if cramped
    plt.xticks(rotation=45, ha="right", color=THEME_COLOUR)
    plt.yticks(color=THEME_COLOUR)

    # Make axis spines theme colour
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME_COLOUR)

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6, colors=THEME_COLOUR)

    plt.tight_layout()

    st.pyplot(fig)
    
    
    # 2. Initial Data

    st.markdown(f"<h2 style='color:{THEME_COLOUR};'>2. Data Split</h2>", unsafe_allow_html=True)


    st.write("**Target Variable (n=1):**")
    st.write("- `priority`")

    st.write("**Features (n=7):**")
    st.write("- `case_summary` (primary)")
    st.write("- `created_at`")
    st.write("- `channel`")
    st.write("- `case_type`")
    st.write("- `customer_tenure_months`")
    st.write("- `plan_tier`")
    st.write("- `category`")

with col2:

    # 3. Cleaning (Before Split)
 
    st.markdown(f"<h2 style='color:{THEME_COLOUR};'>3. Cleaning (Before Train-Test Split)</h2>", unsafe_allow_html=True)
    st.write("Cleaning steps applied to columns before splitting the dataset:")

    st.write("**case_summary**: Fix column-shift rows, convert to lowercase, strip whitespace")
    st.write("**created_at**: Parse to datetime")
    st.write("**category**: Standardise casing")
    st.write("**plan_tier**: Standardise casing")
    st.write("**channel**: Confirm casing is consistent")
    st.write("**case_type**: Confirm casing is consistent")
    st.write("**customer_tenure_months**: Check for nulls and outliers")
    st.write("**priority**: Standardise casing, confirm no unexpected values")


    # 4. Train–Test Split

    st.markdown(f"<h2 style='color:{THEME_COLOUR};'>4. Train–Test Split</h2>", unsafe_allow_html=True)
    st.write("The dataset was split into training and test sets, stratified by `priority` to preserve class distribution.")
    st.write("- Training set: 80%")
    st.write("- Test set: 20%")


    # 5. Preprocessing (After Split)

    st.markdown(f"<h2 style='color:{THEME_COLOUR};'>5. Preprocessing (After Split)</h2>", unsafe_allow_html=True)
    st.write("Preprocessing steps were fitted on the training set only and then applied to the test set.")

    st.write("**case_summary**: TF-IDF vectorization")
    st.write("**channel**: One-hot encoding")
    st.write("**case_type**: One-hot encoding")
    st.write("**category**: One-hot encoding")
    st.write("**plan_tier**: One-hot encoding")
    st.write("**customer_tenure_months**: Bin numerical values, then one-hot encode")
    st.write("**created_at**: Extract hour + day_of_week, then one-hot encode")
    st.write("**priority**: Ordinal encoding: Low=0, Medium=1, High=2, Urgent=3")
st.markdown(
    "[View Project on GitHub](https://github.com/OzzyHF/Northstar-Desk-Decision-Support-Tool-)",
    unsafe_allow_html=True
)

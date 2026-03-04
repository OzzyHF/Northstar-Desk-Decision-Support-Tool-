import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

st.set_page_config(layout="wide")


THEME_COLOUR = "#597AAC"



path = "Data\clean.csv"
data = pd.read_csv(path)
INPUT_DIR  = os.path.join("Data", "processed")
SPLITS_DIR = os.path.join("Data", "splits")
OUTPUT_DIR = "models"

# Load preprocessed data
X_train = joblib.load(os.path.join(INPUT_DIR, "X_train_processed.pkl"))
X_test  = joblib.load(os.path.join(INPUT_DIR, "X_test_processed.pkl"))
y_train = pd.read_csv(os.path.join(SPLITS_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(SPLITS_DIR, "y_test.csv")).squeeze()

print(f"Training features: {X_train.shape[1]}")

# Train Logistic Regression with balanced class weights
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High", "Urgent"]))


st.markdown(f"<h1 style='color:{THEME_COLOUR};'>Hackathon - Top 10 Urgent Contacts</h1>", unsafe_allow_html=True)
st.write('This page provides a list of those enquiriers that require urgent contact.') 

top10 = data.sort_values("priority", ascending=False).head(10)


st.dataframe(top10)
st.table()
st.markdown(
    "[View Project on GitHub](https://github.com/OzzyHF/Northstar-Desk-Decision-Support-Tool-)",
    unsafe_allow_html=True
)

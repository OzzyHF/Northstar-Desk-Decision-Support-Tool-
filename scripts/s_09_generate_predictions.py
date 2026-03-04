import os
import joblib
import pandas as pd
import numpy as np

# --- File paths ---
MODEL_PATH = os.path.join("models", "lr_model.pkl")
PREPROCESSOR_PATH = os.path.join("Data", "processed", "preprocessor.pkl")
X_TEST_PATH = os.path.join("Data", "processed", "X_test_processed.pkl")
Y_TEST_PATH = os.path.join("Data", "splits", "y_test.csv")
X_TEST_RAW_PATH = os.path.join("Data", "splits", "X_test.csv")
OUTPUT_PATH = os.path.join("results", "predictions.csv")

# --- Load the model, preprocessor, and test data ---
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
X_test = joblib.load(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()
X_test_raw = pd.read_csv(X_TEST_RAW_PATH)

# --- Run predictions ---
y_pred = model.predict(X_test)

# predict_proba gives a probability for each class (Low, Medium, High, Urgent)
# confidence is the highest probability — how sure the model is about its pick
y_proba = model.predict_proba(X_test)
confidence = y_proba.max(axis=1)

# --- Build the output table ---
labels = ["Low", "Medium", "High", "Urgent"]

preds_df = pd.DataFrame({
    "case_id": X_test_raw["case_id"],
    "actual_priority": y_test.values,
    "predicted_priority": y_pred,
    "confidence": confidence.round(3),
    "prob_Low": y_proba[:, 0].round(3),
    "prob_Medium": y_proba[:, 1].round(3),
    "prob_High": y_proba[:, 2].round(3),
    "prob_Urgent": y_proba[:, 3].round(3),
})

# --- Extract feature names from the preprocessor ---
feature_names = preprocessor.get_feature_names_out()

# --- Build coefficients table ---
# model.coef_ has shape (n_classes, n_features)
coef_df = pd.DataFrame(
    model.coef_,
    index=[f"coef_{lbl}" for lbl in labels],
    columns=feature_names,
)
# Add intercepts as a column
coef_df.insert(0, "intercept", model.intercept_)
# Transpose so each row is a feature
coef_df = coef_df.T.reset_index()
coef_df.columns = ["feature", "coef_Low", "coef_Medium", "coef_High", "coef_Urgent"]

# --- Save to CSV ---
os.makedirs("results", exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    preds_df.to_csv(f, index=False)
    f.write("\n")  # blank separator line
    coef_df.to_csv(f, index=False)

print(f"Done — saved {len(preds_df)} predictions + {len(coef_df)} coefficients to {OUTPUT_PATH}")

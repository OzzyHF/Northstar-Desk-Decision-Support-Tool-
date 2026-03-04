import os
import joblib
import pandas as pd

# --- File paths ---
MODEL_PATH = os.path.join("models", "lr_model.pkl")
X_TEST_PATH = os.path.join("Data", "processed", "X_test_processed.pkl")
Y_TEST_PATH = os.path.join("Data", "splits", "y_test.csv")
X_TEST_RAW_PATH = os.path.join("Data", "splits", "X_test.csv")
OUTPUT_PATH = os.path.join("results", "predictions.csv")

# --- Load the model and test data ---
model = joblib.load(MODEL_PATH)
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

# --- Save to CSV ---
os.makedirs("results", exist_ok=True)
preds_df.to_csv(OUTPUT_PATH, index=False)

print(f"Done — saved {len(preds_df)} predictions to {OUTPUT_PATH}")

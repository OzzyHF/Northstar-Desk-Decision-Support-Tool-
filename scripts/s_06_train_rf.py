import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

INPUT_DIR  = os.path.join("Data", "processed")
SPLITS_DIR = os.path.join("Data", "splits")
OUTPUT_DIR = "models"

# Load full preprocessed data (including TF-IDF)
X_train = joblib.load(os.path.join(INPUT_DIR, "X_train_processed.pkl"))
X_test  = joblib.load(os.path.join(INPUT_DIR, "X_test_processed.pkl"))
y_train = pd.read_csv(os.path.join(SPLITS_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(SPLITS_DIR, "y_test.csv")).squeeze()

print(f"Training features: {X_train.shape[1]}")

# Best params from Optuna search (run on structured features, applied to full feature set)
best_params = {
    "n_estimators":      413,
    "max_depth":         18,
    "min_samples_split": 19,
    "min_samples_leaf":  1,
    "max_features":      "sqrt",
    "class_weight":      "balanced",
    "bootstrap":         True,
    "random_state":      42,
}

best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High", "Urgent"]))

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "rf_model.pkl"))
print(f"Saved to {OUTPUT_DIR}/rf_model.pkl")

import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report

INPUT_DIR  = os.path.join("Data", "processed")
SPLITS_DIR = os.path.join("Data", "splits")
MODEL_DIR  = "models"
OUTPUT_DIR = "results"

# Load test data
X_test = joblib.load(os.path.join(INPUT_DIR, "X_test_processed.pkl"))
y_test = pd.read_csv(os.path.join(SPLITS_DIR, "y_test.csv")).squeeze()

# Models to compare
model_configs = {
    "Decision Tree": os.path.join(MODEL_DIR, "dt_model.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "rf_model.pkl"),
}

rows = []

for name, model_path in model_configs.items():
    model  = joblib.load(model_path)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=["Low", "Medium", "High", "Urgent"],
        output_dict=True
    )

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High", "Urgent"]))

    rows.append({
        "Model":          name,
        "Macro F1":       round(report["macro avg"]["f1-score"], 3),
        "Weighted F1":    round(report["weighted avg"]["f1-score"], 3),
        "Urgent Recall":  round(report["Urgent"]["recall"], 3),
        "Urgent F1":      round(report["Urgent"]["f1-score"], 3),
    })

summary = pd.DataFrame(rows)

print("\n=== Model Comparison ===")
print(summary.to_string(index=False))

os.makedirs(OUTPUT_DIR, exist_ok=True)
summary.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
print(f"\nSaved to {OUTPUT_DIR}/model_comparison.csv")

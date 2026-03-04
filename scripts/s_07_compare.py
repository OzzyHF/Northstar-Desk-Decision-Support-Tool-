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
    "Logistic Regression": os.path.join(MODEL_DIR, "lr_model.pkl"),
    "Decision Tree":       os.path.join(MODEL_DIR, "dt_model.pkl"),
    "Random Forest":       os.path.join(MODEL_DIR, "rf_model.pkl"),
}

target_names = ["Low", "Medium", "High", "Urgent"]
summary_rows = []
detail_rows = []

for name, model_path in model_configs.items():
    model  = joblib.load(model_path)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        output_dict=True
    )

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Per-class detail rows
    for cls in target_names:
        detail_rows.append({
            "Model":     name,
            "Class":     cls,
            "Precision": round(report[cls]["precision"], 3),
            "Recall":    round(report[cls]["recall"], 3),
            "F1-Score":  round(report[cls]["f1-score"], 3),
            "Support":   int(report[cls]["support"]),
        })

    # Summary row
    summary_rows.append({
        "Model":          name,
        "Accuracy":       round(report["accuracy"], 3),
        "Macro F1":       round(report["macro avg"]["f1-score"], 3),
        "Weighted F1":    round(report["weighted avg"]["f1-score"], 3),
        "Urgent Recall":  round(report["Urgent"]["recall"], 3),
        "Urgent F1":      round(report["Urgent"]["f1-score"], 3),
    })

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Per-class results
detail_df = pd.DataFrame(detail_rows)
detail_df.to_csv(os.path.join(OUTPUT_DIR, "classification_reports.csv"), index=False)
print(f"\nSaved per-class results to {OUTPUT_DIR}/classification_reports.csv")

# Summary comparison
summary_df = pd.DataFrame(summary_rows)
print("\n=== Model Comparison ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
print(f"Saved to {OUTPUT_DIR}/model_comparison.csv")

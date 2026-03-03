import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = os.path.join("Data", "merged", "model_data.csv")
OUTPUT_DIR = os.path.join("Data", "splits")

df = pd.read_csv(INPUT)
print(f"Loaded: {df.shape[0]} rows")

priority_order = {"Low": 0, "Medium": 1, "High": 2, "Urgent": 3}
df["priority_enc"] = df["priority"].map(priority_order)

FEATURES = ["case_id", "case_summary", "created_at", "channel",
            "case_type", "category", "plan_tier", "customer_tenure_months"]

X = df[FEATURES]
y = df["priority_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

print(f"Train: {X_train.shape[0]} rows")
print(f"Test:  {X_test.shape[0]} rows")
print()
print("Train distribution:")
print(y_train.value_counts().sort_index())
print()
print("Test distribution:")
print(y_test.value_counts().sort_index())
print()
print(f"Saved to {OUTPUT_DIR}/")

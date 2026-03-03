import os
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

INPUT_DIR = os.path.join("Data", "splits")
OUTPUT_DIR = os.path.join("Data", "processed")

# Load splits
X_train = pd.read_csv(os.path.join(INPUT_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(INPUT_DIR, "X_test.csv"))

# Extract datetime features before dropping created_at
X_train["hour"] = pd.to_datetime(X_train["created_at"]).dt.hour
X_train["day_of_week"] = pd.to_datetime(X_train["created_at"]).dt.day_name()
X_test["hour"] = pd.to_datetime(X_test["created_at"]).dt.hour
X_test["day_of_week"] = pd.to_datetime(X_test["created_at"]).dt.day_name()

# Drop columns not used as features
X_train = X_train.drop(columns=["case_id", "created_at"])
X_test = X_test.drop(columns=["case_id", "created_at"])

# Define feature groups
text_feature = "case_summary"
cat_features = ["channel", "case_type", "category", "plan_tier", "day_of_week", "hour"]
tenure_feature = ["customer_tenure_months"]

# Build preprocessor — fit on train only
preprocessor = ColumnTransformer(transformers=[
    ("text",   TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),                   text_feature),
    ("cat",    OneHotEncoder(handle_unknown="ignore"),                                    cat_features),
    ("tenure", KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile"),         tenure_feature),
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump(preprocessor,       os.path.join(OUTPUT_DIR, "preprocessor.pkl"))
joblib.dump(X_train_processed,  os.path.join(OUTPUT_DIR, "X_train_processed.pkl"))
joblib.dump(X_test_processed,   os.path.join(OUTPUT_DIR, "X_test_processed.pkl"))

print(f"Train processed shape: {X_train_processed.shape}")
print(f"Test processed shape:  {X_test_processed.shape}")
print()
print(f"Saved to {OUTPUT_DIR}/")

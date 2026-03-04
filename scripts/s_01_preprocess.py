"""
Preprocess data for model training.

Creates:
- Data/splits/X_train.csv, X_test.csv, y_train.csv, y_test.csv
- Data/processed/X_train_processed.pkl, X_test_processed.pkl
- Data/processed/preprocessor.pkl
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

INPUT_FILE = os.path.join("Data", "clean.csv")
SPLITS_DIR = os.path.join("Data", "splits")
PROCESSED_DIR = os.path.join("Data", "processed")

os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows")

# Filter out rows with missing age_band (as done in notebook)
df = df[df["age_band"].notna() & df["age_band"].astype(str).str.strip().ne("")]
df = df.drop_duplicates()
print(f"After filtering: {len(df)} rows")

# Define features
text_col = "case_summary"
cat_cols = ["channel", "case_type", "category", "plan_tier"]
num_cols = ["customer_tenure_months"]

# Encode priority labels
priority_map = {"Low": 0, "Medium": 1, "High": 2, "Urgent": 3}
df["priority_encoded"] = df["priority"].map(priority_map)
df = df.dropna(subset=["priority_encoded"])
df["priority_encoded"] = df["priority_encoded"].astype(int)

# Build X and y
feature_cols = ["case_id", "created_at", text_col] + cat_cols + num_cols
X = df[feature_cols].copy()
y = df["priority_encoded"]

# Fill missing text
X[text_col] = X[text_col].fillna("")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save raw splits
X_train.to_csv(os.path.join(SPLITS_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(SPLITS_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(SPLITS_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(SPLITS_DIR, "y_test.csv"), index=False)

print(f"Saved raw splits: train={len(X_train)}, test={len(X_test)}")

# Build preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=500), text_col),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
        ("tenure", KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile"), num_cols),
    ],
    remainder="drop"
)

# Fit and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed features: {X_train_processed.shape[1]} columns")

# Save processed data
joblib.dump(X_train_processed, os.path.join(PROCESSED_DIR, "X_train_processed.pkl"))
joblib.dump(X_test_processed, os.path.join(PROCESSED_DIR, "X_test_processed.pkl"))
joblib.dump(preprocessor, os.path.join(PROCESSED_DIR, "preprocessor.pkl"))

print(f"""
Done.
  Raw splits saved to:       {SPLITS_DIR}/
  Processed data saved to:   {PROCESSED_DIR}/

  Files created:
    - X_train.csv, X_test.csv, y_train.csv, y_test.csv
    - X_train_processed.pkl, X_test_processed.pkl
    - preprocessor.pkl
""")

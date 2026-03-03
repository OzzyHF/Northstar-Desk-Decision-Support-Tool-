import os
import joblib
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

optuna.logging.set_verbosity(optuna.logging.WARNING)

INPUT_DIR = os.path.join("Data", "processed")
SPLITS_DIR = os.path.join("Data", "splits")
OUTPUT_DIR = "models"

# Load preprocessed data
X_train = joblib.load(os.path.join(INPUT_DIR, "X_train_processed.pkl"))
X_test  = joblib.load(os.path.join(INPUT_DIR, "X_test_processed.pkl"))
y_train = pd.read_csv(os.path.join(SPLITS_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(SPLITS_DIR, "y_test.csv")).squeeze()

base_model = DecisionTreeClassifier(random_state=42)

# Optuna objective — tunes hyperparameters via 5-fold CV on train set
def objective(trial):
    params = {
        "criterion":         trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "splitter":          trial.suggest_categorical("splitter", ["best", "random"]),
        "max_depth":         trial.suggest_int("max_depth", 2, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 30),
        "max_features":      trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        "class_weight":      trial.suggest_categorical("class_weight", [None, "balanced"]),
        "ccp_alpha":         trial.suggest_float("ccp_alpha", 0.0, 0.02),
    }
    model = clone(base_model).set_params(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, scoring="f1_macro", cv=cv, n_jobs=-1)
    return scores.mean()

print("Tuning Decision Tree with Optuna (50 trials)...")
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50)

print(f"Best CV f1_macro: {round(study.best_value, 4)}")
print(f"Best params: {study.best_params}")

# Refit best model on full training data
best_model = clone(base_model).set_params(**study.best_params)
best_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High", "Urgent"]))

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "dt_model.pkl"))
print(f"Saved to {OUTPUT_DIR}/dt_model.pkl")

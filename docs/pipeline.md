# Priority Predictor — Pipeline Plan

---

## 1. Data Analysis

| Notebook | Purpose |
|---|---|
| `eda_initial.ipynb` | Initial exploratory analysis — understand features, distributions, and confirm what to include in the model |
| `eda_general.ipynb` | Full picture of the company — volume, SLA, escalation, CSAT — can feed into the dashboard |

---

## 2. Data for Model

> **Note:** This is the minimum feature set. Further exploration may identify additional useful features.

| Role | Column |
|---|---|
| **Target** | `priority` |
| **Feature** | `case_summary` _(primary)_ |
| **Feature** | `created_at` |
| **Feature** | `channel` |
| **Feature** | `case_type` |
| **Feature** | `customer_tenure_months` |
| **Feature** | `plan_tier` |

---

## 3. Cleaning _(Before Split)_

| Column | Action |
|---|---|
| `case_summary` | Fix column-shift rows, lowercase, strip whitespace |
| `created_at` | Parse to datetime |
| `category` | Standardise casing |
| `plan_tier` | Standardise casing |
| `channel` | Confirm casing is consistent |
| `case_type` | Confirm casing is consistent |
| `customer_tenure_months` | Check for nulls and outliers |
| `priority` | Standardise casing, confirm no unexpected values |

---

## 4. Train–Test Split

Stratified by `priority` to preserve class distribution in both sets.

| Set | Size |
|---|---|
| Train | 80% |
| Test | 20% |

---

## 5. Preprocessing _(After Split)_

> Fit on training set only, then applied to test set.

| Column | Preprocessing |
|---|---|
| `case_summary` | TF-IDF (`max_features=5000`, `ngram_range=(1,2)`) |
| `channel` | One-hot encode |
| `case_type` | One-hot encode |
| `category` | One-hot encode |
| `plan_tier` | One-hot encode |
| `customer_tenure_months` | Bin into bands → one-hot encode |
| `created_at` | Extract `hour` + `day_of_week` → one-hot encode |
| `priority` | Ordinal encode: Low=0, Medium=1, High=2, Urgent=3 |

---

## 6. Models

> Moderate class imbalance in target — all models use `class_weight='balanced'`.

| # | Model |
|---|---|
| 1 | Logistic Regression |
| 2 | Decision Tree |
| 3 | Random Forest |

---

## 7. Pipeline

1. Load and validate cleaned dataset
2. Feature engineering
3. Train/test split (stratified by `priority`)
4. Preprocessing (fit on train, apply to test)
5. Train all models
6. Evaluate all models
7. Select final model

---

## 8. Evaluation

| Metric | Why |
|---|---|
| Weighted F1 | Accounts for class imbalance across all 4 classes |
| Urgent recall | Most important class to get right — high stakes |
| Per-class F1 | See where each model struggles |
| Confusion matrix | Understand the shape of errors |

---

## 9. Streamlit App

Interactive prototype demonstrating the decision-support workflow.

---

## 10. Outputs

| Output | Description |
|---|---|
| **Working Streamlit prototype** | Interactive app demonstrating the decision-support workflow |
| **Technical summary** | Data prep, modelling choices and rationale, risks, limitations, interpretability choices |
| **User-facing story** | Who uses it, what problem it solves, how they'd use it day-to-day |
| **Roadmap** | What you'd do next with more time — data, modelling, UX, governance, monitoring |

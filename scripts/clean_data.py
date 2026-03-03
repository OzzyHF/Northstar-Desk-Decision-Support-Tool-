import csv
import os
import pandas as pd

FILES = [
    "Q1-Jan.csv", "Q1-Feb.csv", "Q1-Mar.csv",
    "Q2-Apr.csv", "Q2-May.csv", "Q2-June.csv",
    "Q3-July.csv", "Q3-Aug.csv", "Q3-Sep.csv",
    "Q4-Oct.csv", "Q4-Nov.csv", "Q4-Dec.csv",
]

# 1. Load
# Uses csv reader instead of pd.read_csv to fix a column-shift bug in Q1-Feb.csv
# where 27 rows are missing the gender value, causing all columns after it to shift left
frames = []
for filename in FILES:
    filepath = os.path.join("../Data/Raw Data", filename)
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if len(row) == len(headers) - 1:
                row.insert(20, "")  # insert blank gender to fix the shift
            rows.append(row)
    frames.append(pd.DataFrame(rows, columns=headers))

df = pd.concat(frames, ignore_index=True)
initial_rows = len(df)

# 2. Deduplicate — files are in order so last = most recent
df = df.drop_duplicates(subset="case_id", keep="last")
df = df.reset_index(drop=True)

# 3. Clean
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["priority"] = df["priority"].str.strip().str.title()
df["case_summary"] = df["case_summary"].str.strip().str.lower()
df["customer_tenure_months"] = pd.to_numeric(df["customer_tenure_months"], errors="coerce")

for col in ["channel", "case_type", "category", "plan_tier"]:
    df[col] = df[col].str.strip().str.lower()

# 4. Save
os.makedirs("Data/merged", exist_ok=True)

model_cols = ["case_id", "case_summary", "created_at", "channel", "case_type",
              "category", "plan_tier", "customer_tenure_months", "priority"]

df[model_cols].to_csv("Data/merged/model_data.csv", index=False)
df.to_csv("Data/merged/analysis_data.csv", index=False)

print(f"""
Done.
  Files loaded:      {len(FILES)} monthly CSVs
  Initial rows:      {initial_rows}
  After dedup:       {len(df)} unique cases
  Rows removed:      {initial_rows - len(df)}

  Saved:
    Data/merged/model_data.csv    ({len(df)} rows x {len(model_cols)} cols)
    Data/merged/analysis_data.csv ({len(df)} rows x {df.shape[1]} cols)
""")

# Design Decisions

---

## Deduplication approach

**Decision:** Deduplicate by `case_id`, keeping the row from the latest source file — not `drop_duplicates()`.

**Why we changed it:**

The original cleaning notebook uses `drop_duplicates()`, which only removes rows that are completely identical across every column. This is the wrong approach for this dataset.

The raw data is structured as monthly snapshots — each CSV file (`Q1-Jan.csv`, `Q1-Feb.csv`, etc.) is a point-in-time extract of all open or recently closed cases. A case that was opened in January and closed in March will appear in the January, February, and March files. Across those files the rows are *not* identical — the `snapshot_at` timestamp differs, and fields like `status`, `resolution_code`, and `resolution_time_hours` may have updated as the case progressed.

`drop_duplicates()` will not catch these — it sees them as different rows and keeps all of them, leaving the same case in the dataset multiple times under different states.

**Evidence of the problem:** After `drop_duplicates()`, the DQ summary shows `snapshot_at` has only 1 distinct value across 1,703 rows. This strongly suggests the dedup accidentally retained rows from only a single monthly snapshot rather than the full dataset.

**New approach:** Track the source filename for every row on load. Group by `case_id` and keep the row from the latest file. This gives one row per case in its most complete, final state.

---

## Future improvement: extract structured features from `case_summary`

Currently `case_summary` is fed into the model as raw TF-IDF (5000 features) but contributes minimal signal. A better approach would be to extract structured features from the text — urgency keywords ("down", "broken", "critical"), sentiment score (via VADER or TextBlob), text length, and punctuation indicators (exclamation marks, question marks) — and feed those as simple numeric columns instead.

---

## Model selection: why latency was not a factor

**Decision:** Model selected on Macro F1 and Urgent Recall only. Latency was not considered.

**Why:** The dataset contains 1,730 cases across 12 months — approximately 144 tickets per month, or roughly 5 per day. At that volume the model predicts one ticket at a time when an agent opens it. Even a Random Forest with 413 trees returns a prediction in under 200ms, which is invisible to the user.

Latency would be a factor if the tool needed to score thousands of tickets per second or had a hard API response-time SLA. Neither applies here.

**Selected model:** Random Forest — Macro F1: 0.646, Urgent Recall: 0.70 vs Decision Tree Macro F1: 0.582, Urgent Recall: 0.50.

---

## Column-shift fix (Q1-Feb.csv)

**Decision:** For the 27 rows in `Q1-Feb.csv` that have 24 values instead of 25, insert a blank at position 20 (`gender`) before parsing to restore correct column alignment.

**Why:** The `gender` field is completely absent in these rows — no value, not even an empty placeholder — causing every column from `case_summary` onwards to shift one position to the left, putting the case summary text in the gender column and corrupting the row. All 12 file headers are identical and correct; this is a data entry issue isolated to 27 rows in one file.

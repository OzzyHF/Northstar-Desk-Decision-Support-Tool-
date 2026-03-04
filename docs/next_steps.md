# Next Steps

---

## 1. Pipeline (scripts)

| Script | Status | What it does |
|---|---|---|
| `s_04_train_lr.py` | Empty | Train Logistic Regression with `class_weight='balanced'`, save `models/lr_model.pkl` |
| `s_07_compare.py` | Needs update | Add LR to comparison table once s_04 is complete |
| `s_08_final_model.py` | Empty | Read `results/model_comparison.csv`, select RF as best model, copy to `models/final_model.pkl`, log reason |
| `s_09_evaluate.py` | Empty | Load `final_model.pkl`, print classification report + confusion matrix, run SHAP summary, save all outputs to `results/` |
| `s_10_pipeline.py` | Empty | Orchestrator — runs s_01 through s_09 in order using `subprocess`, prints pass/fail for each step |

---

## 2. Dashboard fixes

### Broken across all pages
- All 4 pages load data from a hardcoded Windows path (`C:\Users\barnyrumbold\...`) — replace with relative path `Data/merged/analysis_data.csv`

### Page 2 — Urgent Contacts
- `sort_values("priority")` sorts alphabetically, not by urgency — replace with ordinal sort (Low → Medium → High → Urgent)
- `st.table()` called with no arguments — remove it
- Filter to `priority == "Urgent"` rather than showing all cases sorted

### Page 3 — Model Card
- Metrics shown (MSE, MAE, R²) are regression metrics — replace with classification metrics: Macro F1, Weighted F1, Urgent Recall, Urgent F1
- Model name hardcoded as "Logistic Regression" — update to "Random Forest"
- Load actual values from `results/model_comparison.csv` instead of hardcoded zeros

### New Page — Triage Input Form (most important — core of the brief)
- User enters: case summary, channel, case type, category, plan tier, customer tenure
- App runs the RF model on the inputs and returns: predicted priority + confidence score
- This is what makes the tool interactive and demonstrates the decision-support workflow

### New Page — SHAP / Model Explainability
- Load `models/final_model.pkl` and `Data/processed/preprocessor.pkl`
- SHAP summary plot — which features drive priority across all predictions
- Optional: force plot for a single selected ticket showing why it got its predicted priority

---

## 3. General analysis (Page 1 additions + new page)

### Add to Page 1 — Analytics Dashboard
- Priority distribution chart — how cases break down across Low / Medium / High / Urgent
- Monthly case volume trend — cases per month from `created_at`
- SLA breach rate by priority — `resolution_time_hours > sla_target_hours`
- Escalation rate by plan tier — do Enterprise customers escalate more?

### New Page — SLA & Performance
- Resolution time vs SLA target by priority and team
- Sentiment breakdown (Positive / Neutral / Negative) and its relationship to priority
- CSAT score by priority — note ~43% of CSAT values are missing, flag this on the chart
- Top tags by volume

---

## 4. Hackathon deliverables (from brief)

### Technical summary
- Write a short document covering: data prep decisions, modelling choices, why RF was selected over DT, risks and limitations (synthetic data, class imbalance, TF-IDF signal weakness), interpretability approach (SHAP)
- Can be a page in the app or a standalone markdown file

### User-facing story
- One page: who is the user (frontline support agent / team lead), what problem it solves (agents waste time manually triaging — model flags Urgent cases instantly), how they'd use it day-to-day (paste in a new case, get a priority suggestion before opening the queue)

### Roadmap
- What you'd do next: extract structured features from case_summary (urgency keywords, sentiment score, text length), retrain on real labelled data, add monitoring for model drift, governance review before using for actual decisioning

---

## 5. Presentation outline (30 minutes)

| Slot | Duration | Content |
|---|---|---|
| 1. Context & problem | 3 min | Northstar Desk background — 1,730 cases, ~144/month, manual triage is slow and inconsistent. We want to help agents prioritise faster. |
| 2. User story | 3 min | Who uses the tool: a frontline support agent opens the queue each morning. The tool flags what needs attention first and explains why. |
| 3. Data walkthrough | 4 min | 12 monthly CSVs merged and cleaned — deduplication by case_id, column-shift fix in Q1-Feb, 25 fields including structured triage fields and free text. Show the analytics dashboard. |
| 4. Modelling approach | 5 min | Three models compared (LR, DT, RF). Preprocessing pipeline — TF-IDF, OHE, binning. Class imbalance handled with `class_weight='balanced'`. Optuna hyperparameter tuning. Show model comparison results. |
| 5. Live demo | 8 min | Walk through the app: analytics dashboard → urgent contacts → triage input form (enter a case, get a priority prediction) → model card → SHAP explainability. |
| 6. Limitations & risks | 3 min | Synthetic data (performance will differ on real data). TF-IDF adds noise — case_summary needs better feature engineering. Model should support decisions, not replace them. |
| 7. Roadmap | 4 min | Extract urgency signals from text, retrain on real data, add confidence thresholds, monitoring for drift, governance sign-off before production use. |

**Total: 30 minutes**

---

## Summary — what's left

| Area | Items remaining |
|---|---|
| Pipeline | 4 empty scripts + 1 update (s_04, s_07, s_08, s_09, s_10) |
| Dashboard fixes | Broken paths (all pages), Page 2 sort + filter, Page 3 metrics |
| Dashboard new | Triage input form, SHAP page, SLA & Performance page |
| Analysis | Priority distribution, volume trend, SLA breach, escalation, sentiment, CSAT, tags |
| Hackathon deliverables | Technical summary, user story, roadmap |
| Presentation | 30-min outline above — needs slides or talking points written |

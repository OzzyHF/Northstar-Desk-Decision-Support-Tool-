# Northstar Desk Decision Support Tool

A machine learning-powered decision support tool for triaging customer support cases at Northstar Desk, a UK-based subscription software company.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Problem

At Northstar Desk, over 100 new support cases arrive every month with little indication of priority levels. The team must quickly judge urgency from short summaries, which can lead to delays or missed high-risk issues.

## Solution

We developed an interactive tool that supports triage decisions using machine learning. When an analyst opens the tool, it provides:

- A prioritized list of cases with urgent contacts at the top
- Predictions backed by historical patterns
- Explainable AI insights showing *why* a case is classified as urgent

The analyst makes the final decision — now backed by data.

## Impact

- Faster, more consistent triage
- Better workload focus
- Reduced risk of urgent cases being overlooked

## Features

### Pages

| Page | Description |
|------|-------------|
| **Home** | Overview of the tool, problem statement, and solution |
| **Analytics Dashboard** | Key metrics and visualizations from the case data |
| **Urgent Contacts** | Prioritized table of cases requiring immediate attention |
| **Model Card** | Model comparison, performance metrics, and selection rationale |
| **Data Cleaning Steps** | Overview of data preprocessing pipeline |
| **Explanation** | Select any case and see detailed prediction explanations |
| **New Case Prediction** | Enter new case details and get instant predictions |

### Models

Three classification models were trained and compared:

| Model | Accuracy | Macro F1 | Urgent Recall |
|-------|----------|----------|---------------|
| Logistic Regression | 0.66 | 0.65 | **0.68** |
| Random Forest | 0.67 | 0.65 | 0.65 |
| Decision Tree | 0.61 | 0.60 | 0.68 |

**Logistic Regression** was selected as the default model because it has the highest **Urgent Recall** — for a triage tool, catching urgent cases is the most important metric.

### Explainability

The tool provides transparent explanations using:

- **SHAP values** for feature importance
- **Semantic keyword detection** (urgency, financial, account, technical)
- **LLM-generated explanations** via Groq API (Llama 3.1)
- **Template fallback** when LLM is unavailable

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/OzzyHF/Northstar-Desk-Decision-Support-Tool-.git
cd Northstar-Desk-Decision-Support-Tool-
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Set up Groq API for LLM explanations:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a free API key at [console.groq.com](https://console.groq.com)

### Running the App

```bash
streamlit run Home.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
├── Home.py                     # Main Streamlit entry point
├── pages/
│   ├── 1_Analytics Dashboard.py
│   ├── 2_Urgent Contacts.py
│   ├── 3_Model Card.py
│   ├── 4_Data Cleaning Steps.py
│   ├── 5_Explanation.py
│   └── 6_New Case Prediction.py
├── explainability/             # ML explainability module
│   ├── core.py                 # Main ExplainabilityCore class
│   ├── shap_wrapper.py         # SHAP explainer wrapper
│   ├── feature_grouper.py      # Semantic keyword grouping
│   ├── groq_generator.py       # Groq LLM integration
│   ├── llm_generator.py        # Local LLM fallback
│   ├── structured_output.py    # JSON output builder
│   ├── prompts.py              # Anti-hallucination prompts
│   └── config.py               # Configuration settings
├── scripts/                    # Training pipeline scripts
│   ├── s_01_preprocess.py      # Data preprocessing
│   ├── s_04_train_lr.py        # Logistic Regression training
│   ├── s_05_train_dt.py        # Decision Tree training (Optuna)
│   ├── s_06_train_rf.py        # Random Forest training
│   └── s_07_compare.py         # Model comparison
├── models/                     # Trained model files (.pkl)
├── Data/
│   ├── Raw Data/               # Original CSV files
│   ├── clean.csv               # Cleaned dataset
│   ├── splits/                 # Train/test splits
│   └── processed/              # Preprocessed data & preprocessor
├── results/                    # Model comparison results
├── notebooks/                  # Jupyter notebooks for exploration
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── .github/workflows/          # GitHub Actions CI/CD
├── requirements.txt            # Production dependencies
└── requirements-local.txt      # Local development dependencies
```

## Data

The tool is trained on synthetic customer support case data with the following features:

| Feature | Description |
|---------|-------------|
| `case_summary` | Text description of the customer issue |
| `channel` | Contact method (email, phone, webchat, mobile_app) |
| `case_type` | Type of case (incident, service_request, question, problem) |
| `category` | Category (billing, technical, account, etc.) |
| `plan_tier` | Customer subscription tier (free, standard, pro, enterprise) |
| `customer_tenure_months` | How long the customer has been subscribed |
| `priority` | Target variable (Low, Medium, High, Urgent) |

## Training Pipeline

To retrain the models:

```bash
# 1. Preprocess data
python scripts/s_01_preprocess.py

# 2. Train models
python scripts/s_04_train_lr.py
python scripts/s_05_train_dt.py
python scripts/s_06_train_rf.py

# 3. Compare models
python scripts/s_07_compare.py
```

## Deployment

### Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file to `Home.py`
5. Add secrets in Advanced Settings:
   ```toml
   GROQ_API_KEY = "your_api_key_here"
   ```

### GitHub Actions

The repository includes a CI/CD workflow (`.github/workflows/northstar-desk-support.yml`) that:

- Validates code with ruff
- Checks model and data files exist
- Tests imports across Python 3.10, 3.11, 3.12
- Supports manual model selection

## Technologies

- **Frontend**: Streamlit
- **ML**: scikit-learn, SHAP, Optuna
- **Data**: pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **LLM**: Groq API (Llama 3.1)
- **CI/CD**: GitHub Actions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built during a hackathon focused on AI-powered decision support tools
- Uses synthetic data representing a fictional company (Northstar Desk)

## Contact

- GitHub: [@OzzyHF](https://github.com/OzzyHF)
- Project Link: [https://github.com/OzzyHF/Northstar-Desk-Decision-Support-Tool-](https://github.com/OzzyHF/Northstar-Desk-Decision-Support-Tool-)

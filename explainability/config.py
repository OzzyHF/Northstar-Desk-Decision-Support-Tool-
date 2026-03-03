"""
Configuration for explainability module.

Defines semantic keyword groups for TF-IDF aggregation and
human-readable explanations for each feature type.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Priority levels and their ordinal values
PRIORITY_LEVELS = ["Low", "Medium", "High", "Urgent"]
PRIORITY_ORDER = {p: i for i, p in enumerate(PRIORITY_LEVELS)}

# Semantic groups for TF-IDF token aggregation
SEMANTIC_GROUPS = {
    "urgency_keywords": [
        "crash", "urgent", "broken", "down", "emergency", "critical",
        "asap", "immediately", "outage", "unavailable", "dead", "stuck",
        "blocked", "frozen", "unresponsive", "failing", "failed"
    ],
    "financial_keywords": [
        "invoice", "payment", "billing", "refund", "charge", "subscription",
        "vat", "receipt", "pricing", "cost", "fee", "credit", "debit",
        "transaction", "upgrade", "downgrade", "renewal", "cancel"
    ],
    "account_keywords": [
        "login", "password", "access", "permission", "user", "account",
        "authentication", "sso", "2fa", "mfa", "locked", "reset",
        "credentials", "sign", "register", "activate", "deactivate"
    ],
    "technical_keywords": [
        "error", "bug", "api", "sync", "integration", "export", "import",
        "data", "report", "dashboard", "connection", "timeout", "slow",
        "performance", "loading", "display", "render", "configuration"
    ],
}

# Feature explanations for structured output
FEATURE_EXPLANATIONS = {
    # Channel explanations
    "channel": {
        "email": "Email channel indicates a non-time-sensitive contact method",
        "webchat": "Webchat indicates the customer expects faster responses",
        "phone": "Phone contact typically signals higher urgency",
        "mobile_app": "Mobile app submissions may indicate on-the-go issues",
    },
    # Case type explanations
    "case_type": {
        "incident": "Incidents typically require faster response times",
        "service_request": "Service requests are generally planned activities",
        "question": "Questions usually have lower urgency",
        "problem": "Problems may indicate systemic issues needing attention",
    },
    # Category explanations
    "category": {
        "billing_payments": "Billing issues can block business operations",
        "account_admin": "Account administration is usually routine",
        "technical_support": "Technical issues may impact productivity",
        "product_feedback": "Feedback is typically non-urgent",
        "feature_request": "Feature requests are low priority by nature",
        "general_enquiry": "General enquiries are typically informational",
    },
    # Plan tier explanations
    "plan_tier": {
        "enterprise": "Enterprise customers have SLA commitments",
        "pro": "Pro tier customers expect priority support",
        "standard": "Standard tier follows normal queue priority",
        "free": "Free tier has lowest support priority",
    },
    # Customer tenure impact
    "customer_tenure_months": {
        "high": "Long-tenured customers are high-value relationships",
        "medium": "Established customer with moderate history",
        "low": "New customer, building relationship",
    },
}

# Thresholds for tenure categorization
TENURE_THRESHOLDS = {
    "low": 6,      # < 6 months
    "medium": 24,  # 6-24 months
    "high": 24,    # > 24 months
}

# LLM generation settings
LLM_CONFIG = {
    "n_ctx": 2048,         # Context window
    "n_threads": 4,        # CPU threads
    "temperature": 0.1,    # Low for factual output
    "max_tokens": 512,     # Max response length
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}

# SHAP configuration
SHAP_CONFIG = {
    "max_samples": 100,    # Background samples for KernelExplainer
    "check_additivity": False,  # Disable for speed
}

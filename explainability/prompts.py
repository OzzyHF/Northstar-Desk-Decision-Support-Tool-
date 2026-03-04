"""
Anti-hallucination prompt templates for LLM explanation generation.

These prompts are designed to:
1. Ground the LLM strictly in provided data
2. Forbid speculation or inference beyond the data
3. Produce consistent, factual explanations
"""

SYSTEM_PROMPT = """You are an explanation assistant for a customer support priority classification system.

CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:
1. ONLY use information explicitly provided in the data below
2. NEVER invent, assume, or speculate about information not provided
3. NEVER mention specific customer names, dates, or details not in the data
4. If a value is missing or unclear, say "information not available"
5. Use professional, neutral language suitable for business reports
6. Keep explanations concise (2-4 sentences for summary)

Your task is to explain WHY the model predicted a specific priority level based on the contributing factors provided."""


EXPLANATION_PROMPT_TEMPLATE = """Based on the following model output, write a brief natural language explanation of why this case was classified with this priority level.

=== MODEL OUTPUT DATA (GROUND TRUTH - USE ONLY THIS) ===
{context}
=== END OF DATA ===

Write a 2-4 sentence explanation that:
1. States the predicted priority and confidence level
2. Lists the top 2-3 factors that most strongly influenced this prediction
3. Mentions any text patterns that contributed (if significant)

DO NOT include:
- Information not present in the data above
- Speculation about the customer or their situation
- Recommendations or next steps
- Phrases like "it seems" or "probably" or "likely"

Explanation:"""


SUMMARY_PROMPT_TEMPLATE = """Create a one-sentence summary of why case {case_id} was classified as {priority} priority.

Key factors:
{factors}

Write exactly ONE sentence that captures the main reason for this classification.
Use only the factors provided above. Do not add any information.

Summary:"""


FACTOR_LIST_TEMPLATE = """Factors contributing to {priority} priority:
{positive_factors}

{negative_intro}
{negative_factors}"""


def build_explanation_prompt(context: str) -> str:
    """
    Build the full explanation prompt with context.

    Args:
        context: Formatted context string from StructuredOutputBuilder

    Returns:
        Complete prompt string
    """
    return EXPLANATION_PROMPT_TEMPLATE.format(context=context)


def build_summary_prompt(
    case_id: str,
    priority: str,
    top_factors: list,
) -> str:
    """
    Build a summary prompt for one-liner explanation.

    Args:
        case_id: Case identifier
        priority: Predicted priority level
        top_factors: List of top contributing factors

    Returns:
        Complete summary prompt string
    """
    factors = "\n".join(
        f"- {f['feature']}={f['value']}: {f['explanation']}"
        for f in top_factors[:3]
    )

    return SUMMARY_PROMPT_TEMPLATE.format(
        case_id=case_id,
        priority=priority,
        factors=factors,
    )


def build_factor_explanation(
    priority: str,
    positive_factors: list,
    negative_factors: list,
) -> str:
    """
    Build a factor list for display without LLM.

    Args:
        priority: Predicted priority
        positive_factors: Factors increasing priority
        negative_factors: Factors decreasing priority

    Returns:
        Formatted factor list string
    """
    pos_lines = []
    for f in positive_factors[:4]:
        pos_lines.append(
            f"• {f['feature'].replace('_', ' ').title()}: "
            f"{f['value']} (+{f['contribution']:.2f})"
        )
        pos_lines.append(f"  → {f['explanation']}")

    neg_lines = []
    for f in negative_factors[:2]:
        neg_lines.append(
            f"• {f['feature'].replace('_', ' ').title()}: "
            f"{f['value']} ({f['contribution']:.2f})"
        )
        neg_lines.append(f"  → {f['explanation']}")

    negative_intro = "\nCounterbalancing factors:" if neg_lines else ""

    return FACTOR_LIST_TEMPLATE.format(
        priority=priority,
        positive_factors="\n".join(pos_lines),
        negative_intro=negative_intro,
        negative_factors="\n".join(neg_lines) if neg_lines else "",
    ).strip()


# Validation patterns to detect hallucination
HALLUCINATION_INDICATORS = [
    "it seems",
    "probably",
    "likely that",
    "might be",
    "could be",
    "appears to",
    "suggests that",
    "we can assume",
    "it's possible",
    "based on my knowledge",
    "generally speaking",
    "in my experience",
    "typically",
    "usually",
]


def check_for_hallucination(generated_text: str, source_data: dict) -> dict:
    """
    Check generated text for potential hallucination.

    Args:
        generated_text: LLM-generated explanation
        source_data: Original structured output used as source

    Returns:
        Dict with is_valid bool and list of warnings
    """
    text_lower = generated_text.lower()
    warnings = []

    # Check for speculative language
    for indicator in HALLUCINATION_INDICATORS:
        if indicator in text_lower:
            warnings.append(f"Speculative language detected: '{indicator}'")

    # Check that mentioned priority matches
    pred_priority = source_data.get("prediction", {}).get("priority", "")
    if pred_priority and pred_priority.lower() not in text_lower:
        warnings.append(f"Generated text doesn't mention predicted priority: {pred_priority}")

    # Check case ID if mentioned
    case_id = source_data.get("case_id", "")
    if case_id:
        # Allow either full ID or partial
        if "nd-" in text_lower and case_id.lower() not in text_lower:
            warnings.append("Case ID mentioned but doesn't match source data")

    # Check that at least one feature from source is mentioned
    features_mentioned = False
    positive = source_data.get("feature_contributions", {}).get("positive", [])
    for contrib in positive[:3]:
        feature = contrib.get("feature", "").replace("_", " ")
        if feature.lower() in text_lower:
            features_mentioned = True
            break

    if positive and not features_mentioned:
        warnings.append("No source features mentioned in explanation")

    return {
        "is_valid": len(warnings) == 0,
        "warnings": warnings,
    }

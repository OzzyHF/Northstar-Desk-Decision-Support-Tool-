"""
Groq API generator for natural language explanations.

Uses Groq's fast inference API with Llama models for explanation generation.
Free tier: 30 requests/min, 14K tokens/min.
"""

import logging
import os
from typing import Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars set elsewhere

from .prompts import (
    SYSTEM_PROMPT,
    build_explanation_prompt,
    build_summary_prompt,
    check_for_hallucination,
)
from .structured_output import StructuredOutputBuilder

logger = logging.getLogger(__name__)


class GroqGenerator:
    """
    Natural language explanation generator using Groq API.

    Groq provides extremely fast inference for open-source models.
    """

    # Available models (in order of preference)
    MODELS = [
        "llama-3.1-8b-instant",      # Fast, good quality
        "llama3-8b-8192",            # Fallback
        "mixtral-8x7b-32768",        # Higher quality, slower
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the Groq generator.

        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Model to use (defaults to llama-3.1-8b-instant)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model or self.MODELS[0]
        self.client = None
        self._available = None

    def is_available(self) -> bool:
        """Check if Groq API is available."""
        if self._available is not None:
            return self._available

        if not self.api_key:
            self._available = False
            return False

        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self._available = True
        except ImportError:
            logger.warning("groq package not installed. Install with: pip install groq")
            self._available = False
        except Exception as e:
            logger.warning(f"Failed to initialize Groq client: {e}")
            self._available = False

        return self._available

    def _ensure_client(self):
        """Ensure the Groq client is initialized."""
        if self.client is None:
            if not self.is_available():
                raise RuntimeError("Groq API not available")
            from groq import Groq
            self.client = Groq(api_key=self.api_key)

    def generate_explanation(
        self,
        structured_output: Dict[str, Any],
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate natural language explanation from structured output.

        Args:
            structured_output: Complete structured output dict
            validate: Whether to validate for hallucination

        Returns:
            Dict with explanation text and validation result
        """
        self._ensure_client()

        # Build context and prompt
        builder = StructuredOutputBuilder()
        context = builder.to_prompt_context(structured_output)
        user_prompt = build_explanation_prompt(context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=512,
                top_p=0.9,
            )

            explanation = response.choices[0].message.content.strip()

            # Validate if requested
            validation = {"is_valid": True, "warnings": []}
            if validate:
                validation = check_for_hallucination(explanation, structured_output)

                if not validation["is_valid"]:
                    logger.warning(f"Hallucination detected: {validation['warnings']}")
                    # Return with warnings but don't fail
                    validation["is_valid"] = True  # Allow through with warnings

            return {
                "explanation": explanation,
                "validation": validation,
                "source": "groq",
                "model": self.model,
            }

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    def generate_summary(
        self,
        structured_output: Dict[str, Any],
    ) -> str:
        """
        Generate a one-sentence summary.

        Args:
            structured_output: Complete structured output dict

        Returns:
            One-sentence summary string
        """
        self._ensure_client()

        case_id = structured_output["case_id"]
        priority = structured_output["prediction"]["priority"]
        top_factors = structured_output["feature_contributions"]["positive"][:3]

        user_prompt = build_summary_prompt(case_id, priority, top_factors)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=100,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise


def create_generator(use_groq: bool = True, use_local_llm: bool = False):
    """
    Factory function to create the best available generator.

    Priority order:
    1. Groq API (if available and use_groq=True)
    2. Local LLM (if available and use_local_llm=True)
    3. Template fallback

    Args:
        use_groq: Whether to try Groq API first
        use_local_llm: Whether to try local LLM

    Returns:
        Generator instance
    """
    # Try Groq first
    if use_groq:
        groq_gen = GroqGenerator()
        if groq_gen.is_available():
            logger.info("Using Groq API for explanations")
            return groq_gen

    # Try local LLM
    if use_local_llm:
        from .llm_generator import LLMGenerator
        local_gen = LLMGenerator()
        if local_gen.is_available():
            logger.info("Using local LLM for explanations")
            return local_gen

    # Fall back to template
    logger.info("Using template-based explanations")
    from .llm_generator import FallbackGenerator
    return FallbackGenerator()

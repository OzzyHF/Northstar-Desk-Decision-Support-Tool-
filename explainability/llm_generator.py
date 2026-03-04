"""
LLM generator for natural language explanations.

Uses Mistral 7B Instruct via llama-cpp-python for efficient CPU inference.
Includes validation to prevent hallucination.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .config import MODELS_DIR, MODEL_FILENAME, LLM_CONFIG
from .prompts import (
    SYSTEM_PROMPT,
    build_explanation_prompt,
    build_summary_prompt,
    check_for_hallucination,
)
from .structured_output import StructuredOutputBuilder

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    Natural language explanation generator using local LLM.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the LLM generator.

        Args:
            model_path: Path to GGUF model file
            config: LLM configuration overrides
        """
        self.model_path = model_path or (MODELS_DIR / MODEL_FILENAME)
        self.config = {**LLM_CONFIG, **(config or {})}
        self.llm = None
        self._model_available = None

    def is_available(self) -> bool:
        """Check if the LLM model is available."""
        if self._model_available is not None:
            return self._model_available

        self._model_available = self.model_path.exists()
        return self._model_available

    def load_model(self):
        """Load the LLM model into memory."""
        if self.llm is not None:
            return

        if not self.is_available():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Download with: huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF "
                f"{MODEL_FILENAME} --local-dir {MODELS_DIR}"
            )

        try:
            from llama_cpp import Llama

            logger.info(f"Loading model from {self.model_path}")
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.config["n_ctx"],
                n_threads=self.config["n_threads"],
                verbose=False,
            )
            logger.info("Model loaded successfully")

        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )

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
        if not self.is_available():
            return self._generate_fallback_explanation(structured_output)

        self.load_model()

        # Build context and prompt
        builder = StructuredOutputBuilder()
        context = builder.to_prompt_context(structured_output)
        prompt = build_explanation_prompt(context)

        # Generate with Mistral chat format
        full_prompt = self._format_mistral_prompt(SYSTEM_PROMPT, prompt)

        response = self.llm(
            full_prompt,
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            repeat_penalty=self.config["repeat_penalty"],
            stop=["</s>", "[INST]", "\n\n\n"],
        )

        explanation = response["choices"][0]["text"].strip()

        # Validate if requested
        validation = {"is_valid": True, "warnings": []}
        if validate:
            validation = check_for_hallucination(explanation, structured_output)

            # Regenerate with stricter prompt if validation fails
            if not validation["is_valid"]:
                logger.warning(f"Hallucination detected: {validation['warnings']}")
                return self._generate_fallback_explanation(
                    structured_output,
                    validation_warnings=validation["warnings"],
                )

        return {
            "explanation": explanation,
            "validation": validation,
            "source": "llm",
        }

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
        if not self.is_available():
            return self._generate_fallback_summary(structured_output)

        self.load_model()

        case_id = structured_output["case_id"]
        priority = structured_output["prediction"]["priority"]
        top_factors = structured_output["feature_contributions"]["positive"][:3]

        prompt = build_summary_prompt(case_id, priority, top_factors)
        full_prompt = self._format_mistral_prompt(SYSTEM_PROMPT, prompt)

        response = self.llm(
            full_prompt,
            max_tokens=100,
            temperature=0.1,
            stop=["</s>", "[INST]", "\n"],
        )

        return response["choices"][0]["text"].strip()

    def _format_mistral_prompt(self, system: str, user: str) -> str:
        """Format prompt for Mistral Instruct model."""
        return f"<s>[INST] {system}\n\n{user} [/INST]"

    def _generate_fallback_explanation(
        self,
        structured_output: Dict[str, Any],
        validation_warnings: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Generate template-based explanation when LLM unavailable or fails validation.

        Args:
            structured_output: Complete structured output dict
            validation_warnings: Warnings from failed validation

        Returns:
            Dict with explanation and metadata
        """
        pred = structured_output["prediction"]
        contribs = structured_output["feature_contributions"]

        priority = pred["priority"]
        confidence = pred["confidence"] * 100

        # Build explanation from template
        lines = [
            f"This case has been classified as **{priority}** priority "
            f"with {confidence:.0f}% confidence."
        ]

        # Add top positive factors
        if contribs["positive"]:
            factors = contribs["positive"][:3]
            factor_parts = []
            for f in factors:
                factor_parts.append(
                    f"{f['feature'].replace('_', ' ')} ({f['value']})"
                )

            lines.append(
                f"The main contributing factors are: {', '.join(factor_parts)}."
            )

            # Add explanation for top factor
            top = factors[0]
            lines.append(top["explanation"] + ".")

        # Add text analysis if significant
        text_analysis = structured_output.get("text_analysis", {})
        semantic = text_analysis.get("semantic_groups", {})
        significant_themes = [
            k for k, v in semantic.items()
            if v > 0.05
        ]
        if significant_themes:
            theme_str = significant_themes[0].replace("_", " ")
            lines.append(
                f"The case text shows {theme_str} patterns."
            )

        explanation = " ".join(lines)

        return {
            "explanation": explanation,
            "validation": {
                "is_valid": True,
                "warnings": validation_warnings or [],
            },
            "source": "template",
        }

    def _generate_fallback_summary(
        self,
        structured_output: Dict[str, Any],
    ) -> str:
        """Generate template-based summary."""
        pred = structured_output["prediction"]
        contribs = structured_output["feature_contributions"]

        priority = pred["priority"]

        if contribs["positive"]:
            top = contribs["positive"][0]
            return (
                f"Classified as {priority} priority primarily due to "
                f"{top['feature'].replace('_', ' ')} ({top['value']})."
            )

        return f"Classified as {priority} priority based on overall case characteristics."


class FallbackGenerator:
    """
    Simple template-based generator that requires no LLM.

    Use this when:
    - LLM model is not downloaded
    - Need guaranteed fast response
    - Want deterministic output
    """

    def __init__(self):
        """Initialize fallback generator."""
        pass

    def generate(self, structured_output: Dict[str, Any]) -> str:
        """
        Generate explanation from structured output.

        Args:
            structured_output: Complete structured output dict

        Returns:
            Explanation string
        """
        generator = LLMGenerator()
        result = generator._generate_fallback_explanation(structured_output)
        return result["explanation"]

    def generate_summary(self, structured_output: Dict[str, Any]) -> str:
        """
        Generate one-line summary.

        Args:
            structured_output: Complete structured output dict

        Returns:
            Summary string
        """
        generator = LLMGenerator()
        return generator._generate_fallback_summary(structured_output)

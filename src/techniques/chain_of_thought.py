"""Chain-of-thought prompt technique"""

from typing import Dict, Any, Optional
from .base import PromptTechnique


class ChainOfThoughtPrompt(PromptTechnique):
    """Prompt requesting step-by-step reasoning"""

    def format_prompt(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format with CoT instructions"""
        return f"""Answer this question by thinking step by step.

Question: {question}

Let's approach this step by step:
1."""

    def get_name(self) -> str:
        return "chain_of_thought"

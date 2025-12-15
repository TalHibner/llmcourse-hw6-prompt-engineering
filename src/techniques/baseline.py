"""Baseline prompt technique"""

from typing import Dict, Any, Optional
from .base import PromptTechnique


class BaselinePrompt(PromptTechnique):
    """Simple baseline prompting without enhancements"""

    def format_prompt(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format as simple direct question"""
        return f"Answer this question: {question}"

    def get_name(self) -> str:
        return "baseline"

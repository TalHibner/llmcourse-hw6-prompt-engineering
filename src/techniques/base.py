"""Base interface for prompt techniques"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class PromptTechnique(ABC):
    """Abstract base class for prompt techniques"""

    @abstractmethod
    def format_prompt(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format the prompt using this technique

        Args:
            question: The question to answer
            context: Optional context (e.g., examples for few-shot)

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return technique name"""
        pass

"""Base interface for LLM providers"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt

        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model identifier"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured"""
        pass

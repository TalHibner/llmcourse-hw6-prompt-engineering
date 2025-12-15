"""Standard improved prompt technique"""

from typing import Dict, Any, Optional
from .base import PromptTechnique


class StandardPrompt(PromptTechnique):
    """Enhanced prompt with role and clear structure"""

    def format_prompt(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format with role and structure"""
        return f"""You are an expert assistant. Please provide a clear and concise answer.

Question: {question}

Answer:"""

    def get_name(self) -> str:
        return "standard_improved"

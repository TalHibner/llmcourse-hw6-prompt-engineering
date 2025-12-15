"""Prompt technique implementations"""

from .base import PromptTechnique
from .baseline import BaselinePrompt
from .standard import StandardPrompt
from .few_shot import FewShotPrompt
from .chain_of_thought import ChainOfThoughtPrompt

__all__ = [
    "PromptTechnique",
    "BaselinePrompt",
    "StandardPrompt",
    "FewShotPrompt",
    "ChainOfThoughtPrompt",
]

"""Few-shot learning prompt technique"""

from typing import Dict, Any, Optional, List
from .base import PromptTechnique


class FewShotPrompt(PromptTechnique):
    """Prompt with examples (2-3 shot learning)"""

    def format_prompt(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format with examples"""
        if context and "examples" in context:
            examples = context["examples"]
            examples_text = "\n\n".join([
                f"Q: {ex['question']}\nA: {ex['answer']}"
                for ex in examples[:3]  # Use up to 3 examples
            ])

            return f"""Here are some examples:

{examples_text}

Now answer this question:
Q: {question}
A:"""
        else:
            # Fallback to standard if no examples provided
            return f"""Answer the following question clearly and concisely.

Q: {question}
A:"""

    def get_name(self) -> str:
        return "few_shot"

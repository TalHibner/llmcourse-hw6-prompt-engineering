"""Generate chain-of-thought reasoning dataset"""

from typing import List, Dict
from .base import DatasetExample, Dataset


class CoTDatasetGenerator:
    """Generate chain-of-thought reasoning examples"""

    @staticmethod
    def generate() -> Dataset:
        """Generate complete CoT dataset

        Returns:
            Dataset with multi-step reasoning problems
        """
        examples = []
        examples.extend(CoTDatasetGenerator._math_problems())
        examples.extend(CoTDatasetGenerator._logic_problems())

        return Dataset(
            name="chain_of_thought",
            description="Multi-step reasoning problems",
            examples=examples
        )

    @staticmethod
    def _math_problems() -> List[DatasetExample]:
        """Generate math reasoning problems"""
        problems = [
            {
                "question": "If John has 8 apples and eats 2, then buys 5 more, how many does he have?",
                "expected_answer": "11",
                "steps": ["Start with 8", "Eat 2: 8-2=6", "Buy 5: 6+5=11"]
            },
            {
                "question": "Sarah has 15 cookies. She gives half to her friend. How many does she have left?",
                "expected_answer": "7 or 8",
                "steps": ["Start with 15", "Half of 15 is 7.5", "Round to 7 or 8"]
            },
            {
                "question": "A store sells apples for $2 each. If you buy 4 apples, how much do you pay?",
                "expected_answer": "8",
                "steps": ["$2 per apple", "4 apples", "2 × 4 = $8"]
            },
            {
                "question": "Tom has 20 marbles. He loses 3, then wins 7 more. How many does he have?",
                "expected_answer": "24",
                "steps": ["Start: 20", "Lose 3: 20-3=17", "Win 7: 17+7=24"]
            },
            {
                "question": "A train travels 60 miles in 1 hour. How far does it travel in 3 hours?",
                "expected_answer": "180",
                "steps": ["60 miles per hour", "3 hours", "60 × 3 = 180 miles"]
            },
        ]

        return [
            DatasetExample(
                id=f"cot_math_{i:03d}",
                question=prob["question"],
                expected_answer=prob["expected_answer"],
                category="math",
                metadata={
                    "difficulty": "medium",
                    "steps": len(prob["steps"]),
                    "reasoning_steps": prob["steps"]
                }
            )
            for i, prob in enumerate(problems)
        ]

    @staticmethod
    def _logic_problems() -> List[DatasetExample]:
        """Generate logic reasoning problems"""
        problems = [
            {
                "question": "If all cats are animals, and Fluffy is a cat, is Fluffy an animal?",
                "expected_answer": "yes",
                "steps": ["All cats are animals", "Fluffy is a cat", "Therefore, Fluffy is an animal"]
            },
            {
                "question": "It's raining and I have an umbrella. Will I get wet if I go outside with my umbrella?",
                "expected_answer": "no or less likely",
                "steps": ["It's raining", "Umbrella protects from rain", "Won't get wet (or less wet)"]
            },
            {
                "question": "If today is Monday, what day was it 2 days ago?",
                "expected_answer": "Saturday",
                "steps": ["Today is Monday", "1 day ago was Sunday", "2 days ago was Saturday"]
            },
        ]

        return [
            DatasetExample(
                id=f"cot_logic_{i:03d}",
                question=prob["question"],
                expected_answer=prob["expected_answer"],
                category="logic",
                metadata={
                    "difficulty": "easy",
                    "steps": len(prob["steps"]),
                    "reasoning_steps": prob["steps"]
                }
            )
            for i, prob in enumerate(problems)
        ]

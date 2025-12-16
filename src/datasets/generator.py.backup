"""Dataset generation"""

import json
from pathlib import Path
from typing import List, Dict, Any
from .base import DatasetExample, Dataset


class DatasetGenerator:
    """Generate datasets for experiments"""

    @staticmethod
    def generate_sentiment_analysis() -> Dataset:
        """Generate sentiment analysis dataset"""
        examples = []

        # Positive examples
        positive_texts = [
            "I love this product",
            "This is amazing",
            "Excellent service",
            "Best experience ever",
            "Wonderful quality",
            "Highly recommend this",
            "Fantastic results",
            "Absolutely brilliant",
            "Very impressed with this",
            "Outstanding performance",
            "This exceeded my expectations",
            "Truly exceptional",
            "Love it so much",
            "Perfect in every way",
            "Could not be happier",
        ]

        for i, text in enumerate(positive_texts):
            examples.append(DatasetExample(
                id=f"sent_pos_{i:03d}",
                question=f"What is the sentiment of: '{text}'?",
                expected_answer="positive",
                category="positive",
                metadata={"text_length": "short", "difficulty": "easy"}
            ))

        # Negative examples
        negative_texts = [
            "I hate this",
            "This is terrible",
            "Awful experience",
            "Worst product ever",
            "Very disappointed",
            "Complete waste of money",
            "Do not recommend",
            "Horrible quality",
            "Extremely frustrating",
            "Total disaster",
            "Really bad service",
            "Not worth it at all",
            "Deeply unsatisfied",
            "This is broken",
            "Regret this purchase",
        ]

        for i, text in enumerate(negative_texts):
            examples.append(DatasetExample(
                id=f"sent_neg_{i:03d}",
                question=f"What is the sentiment of: '{text}'?",
                expected_answer="negative",
                category="negative",
                metadata={"text_length": "short", "difficulty": "easy"}
            ))

        # Neutral examples
        neutral_texts = [
            "The product arrived",
            "It is blue",
            "This costs $10",
            "The size is medium",
            "It has three buttons",
            "Made in China",
            "Released in 2020",
            "The box contains items",
            "Standard features included",
            "Shipped on Monday",
        ]

        for i, text in enumerate(neutral_texts):
            examples.append(DatasetExample(
                id=f"sent_neu_{i:03d}",
                question=f"What is the sentiment of: '{text}'?",
                expected_answer="neutral",
                category="neutral",
                metadata={"text_length": "short", "difficulty": "easy"}
            ))

        return Dataset(
            name="sentiment_analysis",
            description="Simple Q&A for sentiment classification",
            examples=examples
        )

    @staticmethod
    def generate_chain_of_thought() -> Dataset:
        """Generate chain-of-thought reasoning dataset"""
        examples = []

        # Math problems
        math_problems = [
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

        for i, prob in enumerate(math_problems):
            examples.append(DatasetExample(
                id=f"cot_math_{i:03d}",
                question=prob["question"],
                expected_answer=prob["expected_answer"],
                category="math",
                metadata={
                    "difficulty": "medium",
                    "steps": len(prob["steps"]),
                    "reasoning_steps": prob["steps"]
                }
            ))

        # Logic problems
        logic_problems = [
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

        for i, prob in enumerate(logic_problems):
            examples.append(DatasetExample(
                id=f"cot_logic_{i:03d}",
                question=prob["question"],
                expected_answer=prob["expected_answer"],
                category="logic",
                metadata={
                    "difficulty": "easy",
                    "steps": len(prob["steps"]),
                    "reasoning_steps": prob["steps"]
                }
            ))

        return Dataset(
            name="chain_of_thought",
            description="Multi-step reasoning problems",
            examples=examples
        )

    @staticmethod
    def save_dataset(dataset: Dataset, output_path: Path):
        """Save dataset to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dataset_name": dataset.name,
            "description": dataset.description,
            "num_examples": len(dataset.examples),
            "examples": [
                {
                    "id": ex.id,
                    "question": ex.question,
                    "expected_answer": ex.expected_answer,
                    "category": ex.category,
                    "metadata": ex.metadata
                }
                for ex in dataset.examples
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_dataset(input_path: Path) -> Dataset:
        """Load dataset from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)

        examples = [
            DatasetExample(
                id=ex["id"],
                question=ex["question"],
                expected_answer=ex["expected_answer"],
                category=ex["category"],
                metadata=ex.get("metadata", {})
            )
            for ex in data["examples"]
        ]

        return Dataset(
            name=data["dataset_name"],
            description=data["description"],
            examples=examples
        )

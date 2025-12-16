"""Generate sentiment analysis dataset"""

from typing import List
from .base import DatasetExample, Dataset


class SentimentDatasetGenerator:
    """Generate sentiment classification examples"""

    @staticmethod
    def generate() -> Dataset:
        """Generate complete sentiment analysis dataset

        Returns:
            Dataset with sentiment classification examples
        """
        examples = []
        examples.extend(SentimentDatasetGenerator._positive_examples())
        examples.extend(SentimentDatasetGenerator._negative_examples())
        examples.extend(SentimentDatasetGenerator._neutral_examples())

        return Dataset(
            name="sentiment_analysis",
            description="Simple Q&A for sentiment classification",
            examples=examples
        )

    @staticmethod
    def _positive_examples() -> List[DatasetExample]:
        """Generate positive sentiment examples"""
        texts = [
            "I love this product", "This is amazing", "Excellent service",
            "Best experience ever", "Wonderful quality", "Highly recommend this",
            "Fantastic results", "Absolutely brilliant", "Very impressed with this",
            "Outstanding performance", "This exceeded my expectations",
            "Truly exceptional", "Love it so much", "Perfect in every way",
            "Could not be happier"
        ]

        return [
            DatasetExample(
                id=f"sent_pos_{i:03d}",
                question=f"What is the sentiment of: '{text}'?",
                expected_answer="positive",
                category="positive",
                metadata={"text_length": "short", "difficulty": "easy"}
            )
            for i, text in enumerate(texts)
        ]

    @staticmethod
    def _negative_examples() -> List[DatasetExample]:
        """Generate negative sentiment examples"""
        texts = [
            "I hate this", "This is terrible", "Awful experience",
            "Worst product ever", "Very disappointed", "Complete waste of money",
            "Do not recommend", "Horrible quality", "Extremely frustrating",
            "Total disaster", "Really bad service", "Not worth it at all",
            "Deeply unsatisfied", "This is broken", "Regret this purchase"
        ]

        return [
            DatasetExample(
                id=f"sent_neg_{i:03d}",
                question=f"What is the sentiment of: '{text}'?",
                expected_answer="negative",
                category="negative",
                metadata={"text_length": "short", "difficulty": "easy"}
            )
            for i, text in enumerate(texts)
        ]

    @staticmethod
    def _neutral_examples() -> List[DatasetExample]:
        """Generate neutral sentiment examples"""
        texts = [
            "The product arrived", "It is blue", "This costs $10",
            "The size is medium", "It has three buttons", "Made in China",
            "Released in 2020", "The box contains items",
            "Standard features included", "Shipped on Monday"
        ]

        return [
            DatasetExample(
                id=f"sent_neu_{i:03d}",
                question=f"What is the sentiment of: '{text}'?",
                expected_answer="neutral",
                category="neutral",
                metadata={"text_length": "short", "difficulty": "easy"}
            )
            for i, text in enumerate(texts)
        ]

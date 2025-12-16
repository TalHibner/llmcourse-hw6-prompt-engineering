"""Dataset generation coordinator - delegates to specialized generators"""

from pathlib import Path
from typing import List
from .base import Dataset
from .sentiment_dataset import SentimentDatasetGenerator
from .cot_dataset import CoTDatasetGenerator
from .dataset_io import DatasetIO


class DatasetGenerator:
    """Coordinate dataset generation across different types"""

    @staticmethod
    def generate_sentiment_analysis() -> Dataset:
        """Generate sentiment analysis dataset

        Returns:
            Sentiment classification dataset
        """
        return SentimentDatasetGenerator.generate()

    @staticmethod
    def generate_chain_of_thought() -> Dataset:
        """Generate chain-of-thought reasoning dataset

        Returns:
            Multi-step reasoning dataset
        """
        return CoTDatasetGenerator.generate()

    @staticmethod
    def generate_all() -> List[Dataset]:
        """Generate all available datasets

        Returns:
            List of all generated datasets
        """
        return [
            DatasetGenerator.generate_sentiment_analysis(),
            DatasetGenerator.generate_chain_of_thought()
        ]

    @staticmethod
    def save_dataset(dataset: Dataset, output_path: Path) -> None:
        """Save dataset to JSON file

        Args:
            dataset: Dataset to save
            output_path: Path where to save
        """
        DatasetIO.save_dataset(dataset, output_path)

    @staticmethod
    def load_dataset(input_path: Path) -> Dataset:
        """Load dataset from JSON file

        Args:
            input_path: Path to JSON file

        Returns:
            Loaded dataset
        """
        return DatasetIO.load_dataset(input_path)

    @staticmethod
    def save_all(datasets: List[Dataset], output_dir: Path) -> None:
        """Save multiple datasets

        Args:
            datasets: List of datasets
            output_dir: Output directory
        """
        DatasetIO.save_all(datasets, output_dir)

    @staticmethod
    def load_all(input_dir: Path) -> List[Dataset]:
        """Load all datasets from directory

        Args:
            input_dir: Directory with JSON files

        Returns:
            List of datasets
        """
        return DatasetIO.load_all(input_dir)

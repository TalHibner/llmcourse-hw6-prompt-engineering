"""Dataset input/output operations"""

import json
from pathlib import Path
from typing import List
from .base import DatasetExample, Dataset


class DatasetIO:
    """Handle dataset serialization and deserialization"""

    @staticmethod
    def save_dataset(dataset: Dataset, output_path: Path) -> None:
        """Save dataset to JSON file

        Args:
            dataset: Dataset to save
            output_path: Path where to save the file
        """
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

        print(f"✓ Saved dataset: {output_path} ({len(dataset.examples)} examples)")

    @staticmethod
    def load_dataset(input_path: Path) -> Dataset:
        """Load dataset from JSON file

        Args:
            input_path: Path to JSON file

        Returns:
            Loaded Dataset object
        """
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

    @staticmethod
    def save_all(datasets: List[Dataset], output_dir: Path) -> None:
        """Save multiple datasets to directory

        Args:
            datasets: List of datasets to save
            output_dir: Directory where to save datasets
        """
        for dataset in datasets:
            output_path = output_dir / f"{dataset.name}.json"
            DatasetIO.save_dataset(dataset, output_path)

    @staticmethod
    def load_all(input_dir: Path) -> List[Dataset]:
        """Load all datasets from directory

        Args:
            input_dir: Directory containing dataset JSON files

        Returns:
            List of loaded datasets
        """
        datasets = []
        for json_file in input_dir.glob("*.json"):
            datasets.append(DatasetIO.load_dataset(json_file))
        return datasets

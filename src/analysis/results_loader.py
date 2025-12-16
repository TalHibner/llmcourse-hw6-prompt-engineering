"""Load experimental results from JSON files"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


class ResultsLoader:
    """Load and organize experiment results from JSON files"""

    def __init__(self, results_dir: Path = Path("results/experiments")):
        """Initialize loader with results directory

        Args:
            results_dir: Path to directory containing result JSON files
        """
        self.results_dir = results_dir
        self.results_by_dataset = defaultdict(dict)

    def load_all_results(self) -> bool:
        """Load all experiment results from directory

        Returns:
            True if results were found and loaded, False otherwise
        """
        result_files = list(self.results_dir.glob("*.json"))

        if not result_files:
            print(f"No result files found in {self.results_dir}")
            return False

        for result_file in result_files:
            self._load_single_result(result_file)

        return True

    def _load_single_result(self, result_file: Path) -> None:
        """Load a single result file and add to dataset

        Args:
            result_file: Path to JSON result file
        """
        with open(result_file, 'r') as f:
            data = json.load(f)

        dataset = data['configuration']['dataset']
        technique = data['configuration']['technique']

        # Extract similarity scores (filter out errors with 0.0 score)
        similarities = [
            r['similarity_score']
            for r in data['results']
            if r['similarity_score'] > 0.0  # Exclude errors
        ]

        # Store results
        self.results_by_dataset[dataset][technique] = {
            'scores': similarities,
            'results': data['results'],
            'configuration': data['configuration']
        }

    def get_dataset_techniques(self, dataset: str) -> Dict:
        """Get all techniques for a specific dataset

        Args:
            dataset: Dataset name

        Returns:
            Dictionary mapping technique names to their results
        """
        return self.results_by_dataset.get(dataset, {})

    def get_technique_scores(self, dataset: str, technique: str) -> List[float]:
        """Get similarity scores for a specific technique

        Args:
            dataset: Dataset name
            technique: Technique name

        Returns:
            List of similarity scores
        """
        if dataset in self.results_by_dataset:
            if technique in self.results_by_dataset[dataset]:
                return self.results_by_dataset[dataset][technique]['scores']
        return []

    def get_example_output(self, dataset: str, technique: str, idx: int = 0) -> Dict:
        """Get example output for a technique

        Args:
            dataset: Dataset name
            technique: Technique name
            idx: Index of example to retrieve

        Returns:
            Example result dict or None if not found
        """
        if dataset not in self.results_by_dataset:
            return None
        if technique not in self.results_by_dataset[dataset]:
            return None

        results = self.results_by_dataset[dataset][technique]['results']
        if idx >= len(results):
            return None

        return results[idx]

    def get_all_datasets(self) -> List[str]:
        """Get list of all loaded datasets

        Returns:
            List of dataset names
        """
        return list(self.results_by_dataset.keys())

    def has_baseline(self, dataset: str) -> bool:
        """Check if dataset has baseline results

        Args:
            dataset: Dataset name

        Returns:
            True if baseline exists for this dataset
        """
        return 'baseline' in self.results_by_dataset.get(dataset, {})

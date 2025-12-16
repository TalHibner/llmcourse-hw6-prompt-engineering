"""Statistical calculations for experimental results"""

import numpy as np
from typing import Dict, List


class StatisticsCalculator:
    """Calculate descriptive statistics for experiment results"""

    @staticmethod
    def calculate_statistics(scores: List[float]) -> Dict[str, float]:
        """Calculate statistics for a set of scores

        Args:
            scores: List of similarity scores

        Returns:
            Dictionary with mean, std, min, max, count
        """
        if not scores:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }

        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'count': len(scores)
        }

    @staticmethod
    def calculate_improvement(baseline_mean: float, technique_mean: float) -> float:
        """Calculate percentage improvement over baseline

        Args:
            baseline_mean: Baseline technique mean score
            technique_mean: New technique mean score

        Returns:
            Percentage improvement (can be negative)
        """
        if baseline_mean == 0:
            return 0.0
        return ((technique_mean - baseline_mean) / baseline_mean) * 100

    @staticmethod
    def find_best_technique(techniques: Dict, baseline_mean: float = None) -> tuple:
        """Find technique with highest mean score

        Args:
            techniques: Dict mapping technique names to their data
            baseline_mean: Optional baseline mean for comparison

        Returns:
            Tuple of (technique_name, mean_score, improvement_pct)
        """
        best_technique = None
        best_mean = -1.0
        improvement = 0.0

        for technique, data in techniques.items():
            stats = StatisticsCalculator.calculate_statistics(data['scores'])
            if stats['mean'] > best_mean:
                best_mean = stats['mean']
                best_technique = technique

        if baseline_mean and best_mean > baseline_mean:
            improvement = StatisticsCalculator.calculate_improvement(
                baseline_mean, best_mean
            )

        return best_technique, best_mean, improvement

    @staticmethod
    def find_most_consistent(techniques: Dict) -> tuple:
        """Find technique with lowest standard deviation

        Args:
            techniques: Dict mapping technique names to their data

        Returns:
            Tuple of (technique_name, std_dev)
        """
        min_std = float('inf')
        most_consistent = None

        for technique, data in techniques.items():
            stats = StatisticsCalculator.calculate_statistics(data['scores'])
            if stats['std'] < min_std:
                min_std = stats['std']
                most_consistent = technique

        return most_consistent, min_std

    @staticmethod
    def aggregate_scores(techniques: Dict, exclude: List[str] = None) -> List[float]:
        """Aggregate scores from multiple techniques

        Args:
            techniques: Dict mapping technique names to their data
            exclude: List of technique names to exclude

        Returns:
            Combined list of all scores
        """
        exclude = exclude or []
        all_scores = []

        for technique, data in techniques.items():
            if technique not in exclude:
                all_scores.extend(data['scores'])

        return all_scores

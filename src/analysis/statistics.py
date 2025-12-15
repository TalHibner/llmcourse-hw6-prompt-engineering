"""Statistical analysis"""

import numpy as np
from typing import List, Dict, Any
from scipy import stats


class StatisticalAnalyzer:
    """Perform statistical analysis on results"""

    @staticmethod
    def calculate_metrics(similarities: List[float]) -> Dict[str, float]:
        """Calculate statistical metrics

        Args:
            similarities: List of similarity scores

        Returns:
            Dictionary of metrics
        """
        return {
            'mean': float(np.mean(similarities)),
            'median': float(np.median(similarities)),
            'std': float(np.std(similarities)),
            'var': float(np.var(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'q1': float(np.percentile(similarities, 25)),
            'q3': float(np.percentile(similarities, 75)),
        }

    @staticmethod
    def compare_techniques(
        baseline: List[float],
        improved: List[float]
    ) -> Dict[str, Any]:
        """Statistical comparison between techniques

        Args:
            baseline: Baseline similarity scores
            improved: Improved technique scores

        Returns:
            Comparison statistics
        """
        # T-test
        t_stat, p_value = stats.ttest_ind(baseline, improved)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(baseline)**2 + np.std(improved)**2) / 2)
        cohens_d = (np.mean(improved) - np.mean(baseline)) / pooled_std if pooled_std > 0 else 0

        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': p_value < 0.05,
            'improvement_pct': float(
                (np.mean(improved) - np.mean(baseline)) / np.mean(baseline) * 100
            ) if np.mean(baseline) > 0 else 0
        }

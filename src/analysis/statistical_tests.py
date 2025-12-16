"""Statistical significance testing"""

import numpy as np
from typing import List, Tuple, Dict, Any
from scipy import stats as scipy_stats


class StatisticalTests:
    """Perform statistical significance tests"""

    @staticmethod
    def t_test(scores1: List[float], scores2: List[float]) -> Tuple[float, bool]:
        """Perform independent samples t-test

        Args:
            scores1: First group of scores
            scores2: Second group of scores

        Returns:
            Tuple of (p_value, is_significant) where significant means p < 0.05
        """
        if len(scores1) < 2 or len(scores2) < 2:
            return 1.0, False

        statistic, p_value = scipy_stats.ttest_ind(scores1, scores2)
        significant = p_value < 0.05

        return float(p_value), significant

    @staticmethod
    def cohens_d(scores1: List[float], scores2: List[float]) -> float:
        """Calculate Cohen's d effect size

        Args:
            scores1: First group of scores (typically baseline)
            scores2: Second group of scores (typically improved technique)

        Returns:
            Cohen's d value (0.2=small, 0.5=medium, 0.8=large effect)
        """
        if not scores1 or not scores2:
            return 0.0

        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        std1, std2 = np.std(scores1, ddof=1), np.std(scores2, ddof=1)
        n1, n2 = len(scores1), len(scores2)

        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        )

        if pooled_std == 0:
            return 0.0

        return float((mean2 - mean1) / pooled_std)

    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size

        Args:
            d: Cohen's d value

        Returns:
            Interpretation string (small/medium/large)
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    @staticmethod
    def compare_to_baseline(
        baseline_scores: List[float],
        technique_scores: List[float],
        technique_name: str
    ) -> Dict[str, Any]:
        """Complete statistical comparison to baseline

        Args:
            baseline_scores: Baseline technique scores
            technique_scores: New technique scores
            technique_name: Name of the technique being compared

        Returns:
            Dictionary with p_value, significant, cohens_d, interpretation
        """
        p_value, significant = StatisticalTests.t_test(
            baseline_scores, technique_scores
        )
        cohens = StatisticalTests.cohens_d(baseline_scores, technique_scores)
        interpretation = StatisticalTests.interpret_cohens_d(cohens)

        return {
            'technique': technique_name,
            'p_value': p_value,
            'significant': significant,
            'cohens_d': cohens,
            'effect_size': interpretation
        }

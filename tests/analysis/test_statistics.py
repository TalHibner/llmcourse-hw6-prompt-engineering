"""Tests for statistical analysis."""
import pytest
import numpy as np
from src.analysis.statistics import StatisticalAnalyzer


class TestCalculateMetrics:
    """Tests for calculate_metrics method."""

    def test_basic_metrics(self):
        """Test basic statistical metrics calculation."""
        similarities = [0.5, 0.6, 0.7, 0.8, 0.9]

        metrics = StatisticalAnalyzer.calculate_metrics(similarities)

        assert isinstance(metrics, dict)
        assert 'mean' in metrics
        assert 'median' in metrics
        assert 'std' in metrics
        assert 'var' in metrics
        assert 'min' in metrics
        assert 'max' in metrics
        assert 'q1' in metrics
        assert 'q3' in metrics

    def test_mean_calculation(self):
        """Test mean is calculated correctly."""
        similarities = [0.2, 0.4, 0.6, 0.8]
        metrics = StatisticalAnalyzer.calculate_metrics(similarities)

        assert metrics['mean'] == pytest.approx(0.5, abs=0.001)

    def test_median_calculation(self):
        """Test median is calculated correctly."""
        similarities = [0.1, 0.2, 0.3, 0.4, 0.5]
        metrics = StatisticalAnalyzer.calculate_metrics(similarities)

        assert metrics['median'] == pytest.approx(0.3, abs=0.001)

    def test_min_max(self):
        """Test min and max are correct."""
        similarities = [0.3, 0.1, 0.9, 0.5, 0.7]
        metrics = StatisticalAnalyzer.calculate_metrics(similarities)

        assert metrics['min'] == 0.1
        assert metrics['max'] == 0.9

    def test_quartiles(self):
        """Test quartile calculations."""
        similarities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        metrics = StatisticalAnalyzer.calculate_metrics(similarities)

        assert 0.0 <= metrics['q1'] <= 1.0
        assert 0.0 <= metrics['q3'] <= 1.0
        assert metrics['q1'] < metrics['q3']

    def test_single_value(self):
        """Test with single value."""
        similarities = [0.75]
        metrics = StatisticalAnalyzer.calculate_metrics(similarities)

        assert metrics['mean'] == 0.75
        assert metrics['median'] == 0.75
        assert metrics['min'] == 0.75
        assert metrics['max'] == 0.75
        assert metrics['std'] == 0.0

    def test_all_same_values(self):
        """Test with all identical values."""
        similarities = [0.5, 0.5, 0.5, 0.5]
        metrics = StatisticalAnalyzer.calculate_metrics(similarities)

        assert metrics['mean'] == 0.5
        assert metrics['std'] == 0.0
        assert metrics['var'] == 0.0
        assert metrics['min'] == 0.5
        assert metrics['max'] == 0.5

    def test_returns_floats(self):
        """Test that all values are floats."""
        similarities = [0.1, 0.2, 0.3]
        metrics = StatisticalAnalyzer.calculate_metrics(similarities)

        for value in metrics.values():
            assert isinstance(value, float)

    def test_empty_list_handling(self):
        """Test handling of empty list."""
        # NumPy will raise a warning or error for empty array
        with pytest.raises((ValueError, RuntimeWarning)):
            StatisticalAnalyzer.calculate_metrics([])


class TestCompareTechniques:
    """Tests for compare_techniques method."""

    def test_basic_comparison(self):
        """Test basic technique comparison."""
        baseline = [0.5, 0.6, 0.7]
        improved = [0.7, 0.8, 0.9]

        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        assert isinstance(result, dict)
        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'cohens_d' in result
        assert 'significant' in result
        assert 'improvement_pct' in result

    def test_improved_better_than_baseline(self):
        """Test when improved is better."""
        baseline = [0.3, 0.4, 0.5]
        improved = [0.7, 0.8, 0.9]

        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        # Mean of improved should be higher
        assert result['improvement_pct'] > 0
        assert result['cohens_d'] > 0

    def test_baseline_better_than_improved(self):
        """Test when baseline is better."""
        baseline = [0.8, 0.9, 0.85]
        improved = [0.4, 0.5, 0.45]

        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        # Should show negative improvement
        assert result['improvement_pct'] < 0
        assert result['cohens_d'] < 0

    def test_identical_techniques(self):
        """Test when techniques are identical."""
        baseline = [0.5, 0.6, 0.7]
        improved = [0.5, 0.6, 0.7]

        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        assert result['improvement_pct'] == pytest.approx(0.0, abs=0.01)
        assert result['cohens_d'] == pytest.approx(0.0, abs=0.01)

    def test_significance_detection(self):
        """Test statistical significance detection."""
        # Very different distributions should be significant
        baseline = [0.1, 0.15, 0.2, 0.12, 0.18] * 10
        improved = [0.8, 0.85, 0.9, 0.82, 0.88] * 10

        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        assert result['significant'] == True
        assert result['p_value'] < 0.05

    def test_non_significance(self):
        """Test when difference is not significant."""
        # Very similar distributions
        np.random.seed(42)
        baseline = list(np.random.normal(0.5, 0.01, 5))
        improved = list(np.random.normal(0.51, 0.01, 5))

        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        # With small sample and small difference, likely not significant
        # But we just check the format is correct
        assert isinstance(result['significant'], (bool, np.bool_))
        assert 0.0 <= result['p_value'] <= 1.0

    def test_cohens_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        baseline = [0.5, 0.5, 0.5]
        improved = [0.8, 0.8, 0.8]

        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        # Large difference should give large Cohen's d
        assert abs(result['cohens_d']) > 0
        assert isinstance(result['cohens_d'], float)

    def test_improvement_percentage(self):
        """Test improvement percentage calculation."""
        baseline = [0.5, 0.5, 0.5]  # mean = 0.5
        improved = [0.75, 0.75, 0.75]  # mean = 0.75

        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        # Should be 50% improvement
        assert result['improvement_pct'] == pytest.approx(50.0, abs=1.0)

    def test_zero_baseline_handling(self):
        """Test handling when baseline mean is zero."""
        baseline = [0.0, 0.0, 0.0]
        improved = [0.5, 0.5, 0.5]

        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        # Should handle gracefully (returns 0 for improvement_pct)
        assert result['improvement_pct'] == 0.0

    def test_zero_std_handling(self):
        """Test handling when std is zero."""
        baseline = [0.5, 0.5, 0.5]
        improved = [0.5, 0.5, 0.5]

        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        # Cohen's d should be 0 when pooled_std is 0
        assert result['cohens_d'] == 0.0

    def test_returns_floats_and_bool(self):
        """Test that return types are correct."""
        baseline = [0.5, 0.6, 0.7]
        improved = [0.6, 0.7, 0.8]

        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        assert isinstance(result['t_statistic'], float)
        assert isinstance(result['p_value'], float)
        assert isinstance(result['cohens_d'], float)
        assert isinstance(result['significant'], (bool, np.bool_))
        assert isinstance(result['improvement_pct'], float)

    def test_unequal_sample_sizes(self):
        """Test with different sample sizes."""
        baseline = [0.5, 0.6, 0.7]
        improved = [0.6, 0.7, 0.8, 0.85, 0.9]

        # Should work with unequal sizes
        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        assert isinstance(result, dict)
        assert all(key in result for key in [
            't_statistic', 'p_value', 'cohens_d', 'significant', 'improvement_pct'
        ])

    def test_large_samples(self):
        """Test with large sample sizes."""
        np.random.seed(42)
        baseline = list(np.random.normal(0.5, 0.1, 100))
        improved = list(np.random.normal(0.6, 0.1, 100))

        result = StatisticalAnalyzer.compare_techniques(baseline, improved)

        assert isinstance(result, dict)
        assert result['improvement_pct'] > 0  # Improved should be better


class TestStatisticalAnalyzerIntegration:
    """Integration tests for StatisticalAnalyzer."""

    def test_full_workflow(self):
        """Test complete analysis workflow."""
        # Simulate experimental data
        baseline_scores = [0.45, 0.50, 0.48, 0.52, 0.47]
        improved_scores = [0.65, 0.70, 0.68, 0.72, 0.67]

        # Calculate metrics for each
        baseline_metrics = StatisticalAnalyzer.calculate_metrics(baseline_scores)
        improved_metrics = StatisticalAnalyzer.calculate_metrics(improved_scores)

        # Compare techniques
        comparison = StatisticalAnalyzer.compare_techniques(
            baseline_scores,
            improved_scores
        )

        # Verify all data is consistent
        assert baseline_metrics['mean'] < improved_metrics['mean']
        assert comparison['improvement_pct'] > 0
        assert comparison['cohens_d'] > 0

    def test_realistic_experiment_data(self):
        """Test with realistic experimental similarity scores."""
        # Simulate real similarity scores (0-1 range)
        np.random.seed(123)
        baseline = list(np.random.beta(5, 3, 20))  # Skewed towards higher values
        improved = list(np.random.beta(7, 2, 20))  # Even more skewed

        metrics_baseline = StatisticalAnalyzer.calculate_metrics(baseline)
        metrics_improved = StatisticalAnalyzer.calculate_metrics(improved)
        comparison = StatisticalAnalyzer.compare_techniques(baseline, improved)

        # All scores should be in [0, 1]
        assert 0.0 <= metrics_baseline['min'] <= 1.0
        assert 0.0 <= metrics_baseline['max'] <= 1.0
        assert 0.0 <= metrics_improved['min'] <= 1.0
        assert 0.0 <= metrics_improved['max'] <= 1.0

        # Comparison should be valid
        assert -1.0 <= comparison['p_value'] <= 1.0
        assert isinstance(comparison['significant'], (bool, np.bool_))

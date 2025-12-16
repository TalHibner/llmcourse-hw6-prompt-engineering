"""Tests for insights generation"""

import pytest
from src.analysis.insights_generator import InsightsGenerator


@pytest.fixture
def sample_technique_data():
    """Sample technique results for testing"""
    return {
        'baseline': {
            'scores': [0.60, 0.62, 0.58, 0.61, 0.59]
        },
        'standard': {
            'scores': [0.70, 0.72, 0.68, 0.71, 0.69]
        },
        'few_shot': {
            'scores': [0.85, 0.87, 0.83, 0.86, 0.84]
        },
        'chain_of_thought': {
            'scores': [0.78, 0.80, 0.76, 0.79, 0.77]
        }
    }


@pytest.fixture
def sample_all_results(sample_technique_data):
    """Sample results across multiple datasets"""
    return {
        'sentiment_analysis': sample_technique_data,
        'chain_of_thought': {
            'baseline': {'scores': [0.55, 0.57, 0.53]},
            'few_shot': {'scores': [0.75, 0.77, 0.73]},
            'chain_of_thought': {'scores': [0.88, 0.90, 0.86]}
        }
    }


class TestGenerateDatasetInsights:
    """Test dataset-specific insight generation"""

    def test_generate_insights_with_best_technique(self, sample_technique_data):
        """Test insights generation identifies best technique"""
        insights = InsightsGenerator.generate_dataset_insights(
            'test_dataset',
            sample_technique_data
        )

        assert isinstance(insights, str)
        assert 'Few Shot' in insights
        assert 'improvement' in insights.lower()
        assert '%' in insights

    def test_generate_insights_calculates_correct_improvement(self, sample_technique_data):
        """Test that improvement percentage is calculated correctly"""
        insights = InsightsGenerator.generate_dataset_insights(
            'test_dataset',
            sample_technique_data
        )

        # Few-shot (mean ~0.85) vs baseline (mean ~0.60)
        # Should show ~40% improvement
        assert '41' in insights or '42' in insights  # Approximately 41-42%

    def test_generate_insights_identifies_most_consistent(self, sample_technique_data):
        """Test insights identifies most consistent technique"""
        insights = InsightsGenerator.generate_dataset_insights(
            'test_dataset',
            sample_technique_data
        )

        assert 'consistent' in insights.lower()
        assert 'std=' in insights

    def test_generate_insights_with_no_baseline(self):
        """Test handling when baseline is missing"""
        techniques = {
            'few_shot': {'scores': [0.8, 0.9]},
            'standard': {'scores': [0.7, 0.8]}
        }

        insights = InsightsGenerator.generate_dataset_insights(
            'test_dataset',
            techniques
        )

        assert 'Insufficient data' in insights

    def test_generate_insights_with_poor_performance(self):
        """Test detection of techniques performing worse than baseline"""
        techniques = {
            'baseline': {'scores': [0.80, 0.82, 0.81]},
            'poor_technique': {'scores': [0.70, 0.72, 0.71]}  # >5% worse
        }

        insights = InsightsGenerator.generate_dataset_insights(
            'test_dataset',
            techniques
        )

        assert '⚠️' in insights
        assert 'worse' in insights.lower()

    def test_generate_insights_with_no_improvement(self):
        """Test when no technique outperforms baseline"""
        techniques = {
            'baseline': {'scores': [0.85, 0.87, 0.86]},
            'worse_technique': {'scores': [0.80, 0.82, 0.81]}
        }

        insights = InsightsGenerator.generate_dataset_insights(
            'test_dataset',
            techniques
        )

        assert 'worse' in insights.lower() or 'No technique' in insights

    def test_generate_insights_formats_technique_names(self):
        """Test that technique names are properly formatted"""
        techniques = {
            'baseline': {'scores': [0.6]},
            'chain_of_thought': {'scores': [0.8]}
        }

        insights = InsightsGenerator.generate_dataset_insights(
            'test_dataset',
            techniques
        )

        # Should convert 'chain_of_thought' to 'Chain Of Thought'
        assert 'Chain Of Thought' in insights

    def test_generate_insights_with_single_technique(self):
        """Test insights with only baseline and one other technique"""
        techniques = {
            'baseline': {'scores': [0.60, 0.62, 0.61]},
            'standard': {'scores': [0.75, 0.77, 0.76]}
        }

        insights = InsightsGenerator.generate_dataset_insights(
            'test_dataset',
            techniques
        )

        assert 'Standard' in insights
        assert 'improvement' in insights.lower()

    def test_generate_insights_with_identical_performance(self):
        """Test when all techniques perform identically"""
        techniques = {
            'baseline': {'scores': [0.70, 0.70, 0.70]},
            'standard': {'scores': [0.70, 0.70, 0.70]},
            'few_shot': {'scores': [0.70, 0.70, 0.70]}
        }

        insights = InsightsGenerator.generate_dataset_insights(
            'test_dataset',
            techniques
        )

        # Should still generate some insights
        assert isinstance(insights, str)
        assert len(insights) > 0

    def test_generate_insights_returns_bulleted_list(self, sample_technique_data):
        """Test that insights are returned as bulleted markdown list"""
        insights = InsightsGenerator.generate_dataset_insights(
            'test_dataset',
            sample_technique_data
        )

        # Should contain bullet points
        assert '- ' in insights
        lines = insights.split('\n')
        assert all(line.startswith('- ') for line in lines if line)


class TestGenerateOverallInsights:
    """Test overall insights across datasets"""

    def test_generate_overall_insights_with_improvement(self, sample_all_results):
        """Test overall insights when techniques show improvement"""
        insights = InsightsGenerator.generate_overall_insights(sample_all_results)

        assert isinstance(insights, list)
        assert len(insights) > 0
        assert any('Overall Improvement' in insight for insight in insights)
        assert any('%' in insight for insight in insights)

    def test_generate_overall_insights_calculates_aggregate(self, sample_all_results):
        """Test that overall insights aggregate across all datasets"""
        insights = InsightsGenerator.generate_overall_insights(sample_all_results)

        insight_text = ' '.join(insights)
        assert 'improvement' in insight_text.lower()

    def test_generate_overall_insights_with_no_baseline(self):
        """Test handling when no baseline data exists"""
        results = {
            'dataset1': {
                'few_shot': {'scores': [0.8, 0.9]},
                'standard': {'scores': [0.7, 0.8]}
            }
        }

        insights = InsightsGenerator.generate_overall_insights(results)

        # Should return empty or minimal insights
        assert isinstance(insights, list)

    def test_generate_overall_insights_with_no_improvement(self):
        """Test when techniques don't improve over baseline"""
        results = {
            'dataset1': {
                'baseline': {'scores': [0.85, 0.87, 0.86]},
                'poor_technique': {'scores': [0.70, 0.72, 0.71]}
            }
        }

        insights = InsightsGenerator.generate_overall_insights(results)

        insight_text = ' '.join(insights)
        assert 'did not show' in insight_text or 'improvement' in insight_text.lower()

    def test_generate_overall_insights_with_empty_results(self):
        """Test handling of empty results"""
        insights = InsightsGenerator.generate_overall_insights({})

        assert isinstance(insights, list)

    def test_generate_overall_insights_with_multiple_datasets(self):
        """Test aggregation across multiple datasets"""
        results = {
            'dataset1': {
                'baseline': {'scores': [0.6, 0.6]},
                'improved': {'scores': [0.8, 0.8]}
            },
            'dataset2': {
                'baseline': {'scores': [0.7, 0.7]},
                'improved': {'scores': [0.9, 0.9]}
            }
        }

        insights = InsightsGenerator.generate_overall_insights(results)

        assert len(insights) > 0
        assert any('improvement' in insight.lower() for insight in insights)

    def test_generate_overall_insights_mixed_performance(self):
        """Test insights when some datasets improve, others don't"""
        results = {
            'good_dataset': {
                'baseline': {'scores': [0.5, 0.5]},
                'improved': {'scores': [0.9, 0.9]}
            },
            'poor_dataset': {
                'baseline': {'scores': [0.8, 0.8]},
                'worse': {'scores': [0.6, 0.6]}
            }
        }

        insights = InsightsGenerator.generate_overall_insights(results)

        # Should still calculate overall trend
        assert isinstance(insights, list)


class TestGenerateRecommendations:
    """Test recommendation generation"""

    def test_generate_recommendations_identifies_best(self, sample_all_results):
        """Test that recommendations identify best overall technique"""
        recommendations = InsightsGenerator.generate_recommendations(sample_all_results)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_generate_recommendations_excludes_baseline(self, sample_all_results):
        """Test that baseline is not recommended"""
        recommendations = InsightsGenerator.generate_recommendations(sample_all_results)

        recommendations_text = ' '.join(recommendations)
        assert 'Baseline' not in recommendations_text

    def test_generate_recommendations_formats_names(self):
        """Test that technique names are properly formatted"""
        results = {
            'dataset1': {
                'baseline': {'scores': [0.6]},
                'chain_of_thought': {'scores': [0.9]}
            }
        }

        recommendations = InsightsGenerator.generate_recommendations(results)

        recommendations_text = ' '.join(recommendations)
        assert 'Chain Of Thought' in recommendations_text

    def test_generate_recommendations_with_no_techniques(self):
        """Test handling when only baseline exists"""
        results = {
            'dataset1': {
                'baseline': {'scores': [0.7, 0.8, 0.75]}
            }
        }

        recommendations = InsightsGenerator.generate_recommendations(results)

        # Should return empty or no recommendation
        assert isinstance(recommendations, list)

    def test_generate_recommendations_with_tie(self):
        """Test when multiple techniques perform equally well"""
        results = {
            'dataset1': {
                'baseline': {'scores': [0.6]},
                'technique1': {'scores': [0.9]},
                'technique2': {'scores': [0.9]}
            }
        }

        recommendations = InsightsGenerator.generate_recommendations(results)

        # Should recommend one of them
        assert len(recommendations) > 0

    def test_generate_recommendations_aggregates_across_datasets(self):
        """Test that recommendations consider all datasets"""
        results = {
            'dataset1': {
                'baseline': {'scores': [0.5]},
                'good_technique': {'scores': [0.8]}
            },
            'dataset2': {
                'baseline': {'scores': [0.6]},
                'good_technique': {'scores': [0.9]}
            }
        }

        recommendations = InsightsGenerator.generate_recommendations(results)

        # Should aggregate scores across both datasets
        assert len(recommendations) > 0
        assert 'Good Technique' in ' '.join(recommendations)

    def test_generate_recommendations_includes_production_mention(self, sample_all_results):
        """Test that recommendations mention production use"""
        recommendations = InsightsGenerator.generate_recommendations(sample_all_results)

        recommendations_text = ' '.join(recommendations)
        assert 'production' in recommendations_text.lower()

    def test_generate_recommendations_with_empty_results(self):
        """Test handling of empty results"""
        recommendations = InsightsGenerator.generate_recommendations({})

        assert isinstance(recommendations, list)

    def test_generate_recommendations_with_single_dataset(self):
        """Test recommendations with only one dataset"""
        results = {
            'only_dataset': {
                'baseline': {'scores': [0.6, 0.6]},
                'best_technique': {'scores': [0.9, 0.9]}
            }
        }

        recommendations = InsightsGenerator.generate_recommendations(results)

        assert len(recommendations) > 0
        assert 'Best Technique' in ' '.join(recommendations)


class TestInsightsIntegration:
    """Integration tests for insights generation"""

    def test_full_insights_pipeline(self, sample_all_results):
        """Test complete insights generation pipeline"""
        # Generate all types of insights
        dataset_insights = {}
        for dataset, techniques in sample_all_results.items():
            dataset_insights[dataset] = InsightsGenerator.generate_dataset_insights(
                dataset, techniques
            )

        overall_insights = InsightsGenerator.generate_overall_insights(sample_all_results)
        recommendations = InsightsGenerator.generate_recommendations(sample_all_results)

        # Verify all components generated
        assert len(dataset_insights) == 2
        assert all(isinstance(v, str) for v in dataset_insights.values())
        assert isinstance(overall_insights, list)
        assert isinstance(recommendations, list)

    def test_insights_consistency(self, sample_technique_data):
        """Test that multiple calls produce same insights"""
        insights1 = InsightsGenerator.generate_dataset_insights(
            'test', sample_technique_data
        )
        insights2 = InsightsGenerator.generate_dataset_insights(
            'test', sample_technique_data
        )

        assert insights1 == insights2

    def test_all_methods_are_static(self):
        """Test that all methods can be called without instantiation"""
        # Should not need to create instance
        techniques = {
            'baseline': {'scores': [0.6]},
            'improved': {'scores': [0.8]}
        }
        results = {'dataset': techniques}

        # All should work as class methods
        InsightsGenerator.generate_dataset_insights('test', techniques)
        InsightsGenerator.generate_overall_insights(results)
        InsightsGenerator.generate_recommendations(results)

        # Should succeed without errors

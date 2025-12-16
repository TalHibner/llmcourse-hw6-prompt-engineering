"""Tests for results formatting"""

import pytest
from src.analysis.results_formatter import ResultsFormatter


@pytest.fixture
def sample_techniques():
    """Sample technique results"""
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
def sample_example():
    """Sample example result"""
    return {
        'question': 'What is the capital of France?',
        'expected_answer': 'Paris',
        'llm_response': 'The capital of France is Paris',
        'similarity_score': 0.9245
    }


class TestFormatDatasetName:
    """Test dataset name formatting"""

    def test_format_simple_name(self):
        """Test formatting simple dataset name"""
        result = ResultsFormatter.format_dataset_name('sentiment_analysis')
        assert result == 'Sentiment Analysis'

    def test_format_multi_word_name(self):
        """Test formatting multi-word dataset name"""
        result = ResultsFormatter.format_dataset_name('chain_of_thought')
        assert result == 'Chain Of Thought'

    def test_format_already_formatted(self):
        """Test formatting already formatted name"""
        result = ResultsFormatter.format_dataset_name('Simple')
        assert result == 'Simple'

    def test_format_lowercase(self):
        """Test formatting lowercase name"""
        result = ResultsFormatter.format_dataset_name('test')
        assert result == 'Test'

    def test_format_with_numbers(self):
        """Test formatting name with numbers"""
        result = ResultsFormatter.format_dataset_name('dataset_v2')
        assert result == 'Dataset V2'


class TestFormatTechniqueName:
    """Test technique name formatting"""

    def test_format_baseline(self):
        """Test formatting baseline technique"""
        result = ResultsFormatter.format_technique_name('baseline')
        assert result == 'Baseline'

    def test_format_chain_of_thought(self):
        """Test formatting chain_of_thought technique"""
        result = ResultsFormatter.format_technique_name('chain_of_thought')
        assert result == 'Chain Of Thought'

    def test_format_few_shot(self):
        """Test formatting few_shot technique"""
        result = ResultsFormatter.format_technique_name('few_shot')
        assert result == 'Few Shot'

    def test_format_standard_improved(self):
        """Test formatting standard_improved technique"""
        result = ResultsFormatter.format_technique_name('standard_improved')
        assert result == 'Standard Improved'

    def test_format_removes_underscores(self):
        """Test that underscores are removed"""
        result = ResultsFormatter.format_technique_name('multi_word_technique')
        assert '_' not in result


class TestGeneratePerformanceTable:
    """Test performance table generation"""

    def test_generate_table_structure(self, sample_techniques):
        """Test that table has correct markdown structure"""
        table = ResultsFormatter.generate_performance_table(
            'test_dataset',
            sample_techniques
        )

        assert '|' in table
        assert 'Technique' in table
        assert 'Mean Similarity' in table
        assert '|-' in table  # Header separator

    def test_generate_table_includes_all_techniques(self, sample_techniques):
        """Test that all techniques are included in table"""
        table = ResultsFormatter.generate_performance_table(
            'test_dataset',
            sample_techniques
        )

        assert 'Baseline' in table
        assert 'Standard' in table
        assert 'Few Shot' in table
        assert 'Chain Of Thought' in table

    def test_generate_table_shows_statistics(self, sample_techniques):
        """Test that table shows statistical values"""
        table = ResultsFormatter.generate_performance_table(
            'test_dataset',
            sample_techniques
        )

        # Should show mean, std, min, max
        assert '0.60' in table or '0.6000' in table  # Baseline mean
        assert 'Std Dev' in table
        assert 'Min' in table
        assert 'Max' in table

    def test_generate_table_shows_improvements(self, sample_techniques):
        """Test that table shows improvement percentages"""
        table = ResultsFormatter.generate_performance_table(
            'test_dataset',
            sample_techniques
        )

        # Few-shot should show improvement
        assert '+' in table  # Positive improvement indicator
        assert '%' in table

    def test_generate_table_baseline_no_improvement(self, sample_techniques):
        """Test that baseline shows no improvement value"""
        table = ResultsFormatter.generate_performance_table(
            'test_dataset',
            sample_techniques
        )

        lines = table.split('\n')
        # Find data lines (not header or separator)
        baseline_line = [l for l in lines if '| Baseline |' in l][0]
        # Baseline improvement column should show "-"
        assert ' - |' in baseline_line

    def test_generate_table_without_baseline(self):
        """Test table generation without baseline"""
        techniques = {
            'few_shot': {'scores': [0.8, 0.9]},
            'standard': {'scores': [0.7, 0.8]}
        }

        table = ResultsFormatter.generate_performance_table(
            'test_dataset',
            techniques
        )

        # Should still generate table
        assert 'Technique' in table
        assert 'Few Shot' in table

    def test_generate_table_technique_order(self, sample_techniques):
        """Test that techniques appear in correct order"""
        table = ResultsFormatter.generate_performance_table(
            'test_dataset',
            sample_techniques
        )

        lines = table.split('\n')
        technique_lines = [l for l in lines if '|' in l and 'Technique' not in l and '---' not in l]

        # Baseline should be first
        assert 'Baseline' in technique_lines[0]

    def test_generate_table_with_single_technique(self):
        """Test table with only one technique"""
        techniques = {
            'baseline': {'scores': [0.7, 0.75, 0.72]}
        }

        table = ResultsFormatter.generate_performance_table(
            'test_dataset',
            techniques
        )

        assert 'Baseline' in table
        assert '0.7' in table

    def test_generate_table_sample_size(self, sample_techniques):
        """Test that table shows correct sample sizes"""
        table = ResultsFormatter.generate_performance_table(
            'test_dataset',
            sample_techniques
        )

        # Each technique has 5 samples
        assert '5' in table

    def test_generate_table_precision(self, sample_techniques):
        """Test that numbers are formatted with correct precision"""
        table = ResultsFormatter.generate_performance_table(
            'test_dataset',
            sample_techniques
        )

        # Should have 4 decimal places for mean/std/min/max
        # Count dots to verify decimal formatting
        assert table.count('.') > 10  # Multiple decimal values


class TestGenerateStatisticalSignificance:
    """Test statistical significance section generation"""

    def test_generate_significance_basic(self, sample_techniques):
        """Test basic significance section generation"""
        text = ResultsFormatter.generate_statistical_significance(
            'test_dataset',
            sample_techniques
        )

        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_significance_includes_comparisons(self, sample_techniques):
        """Test that all techniques are compared to baseline"""
        text = ResultsFormatter.generate_statistical_significance(
            'test_dataset',
            sample_techniques
        )

        assert 'Standard vs Baseline' in text
        assert 'Few Shot vs Baseline' in text
        assert 'Chain Of Thought vs Baseline' in text

    def test_generate_significance_shows_pvalues(self, sample_techniques):
        """Test that p-values are shown"""
        text = ResultsFormatter.generate_statistical_significance(
            'test_dataset',
            sample_techniques
        )

        assert 'p-value' in text.lower()
        assert 'significant' in text.lower()

    def test_generate_significance_shows_cohens_d(self, sample_techniques):
        """Test that Cohen's d is shown"""
        text = ResultsFormatter.generate_statistical_significance(
            'test_dataset',
            sample_techniques
        )

        assert "Cohen's d" in text
        assert 'effect' in text.lower()

    def test_generate_significance_without_baseline(self):
        """Test significance generation without baseline"""
        techniques = {
            'few_shot': {'scores': [0.8, 0.9]},
            'standard': {'scores': [0.7, 0.8]}
        }

        text = ResultsFormatter.generate_statistical_significance(
            'test_dataset',
            techniques
        )

        assert 'No baseline' in text

    def test_generate_significance_symbols(self, sample_techniques):
        """Test that significance uses correct symbols"""
        text = ResultsFormatter.generate_statistical_significance(
            'test_dataset',
            sample_techniques
        )

        # Should have checkmarks or crosses
        assert '✓' in text or '✗' in text

    def test_generate_significance_with_subset(self):
        """Test significance with only some techniques"""
        techniques = {
            'baseline': {'scores': [0.6, 0.6, 0.6]},
            'few_shot': {'scores': [0.9, 0.9, 0.9]}
        }

        text = ResultsFormatter.generate_statistical_significance(
            'test_dataset',
            techniques
        )

        assert 'Few Shot vs Baseline' in text
        assert 'Standard vs Baseline' not in text

    def test_generate_significance_effect_size_labels(self, sample_techniques):
        """Test that effect size interpretations are included"""
        text = ResultsFormatter.generate_statistical_significance(
            'test_dataset',
            sample_techniques
        )

        # Should include effect size labels (negligible, small, medium, large)
        assert any(word in text.lower() for word in ['negligible', 'small', 'medium', 'large', 'effect'])


class TestFormatExampleOutput:
    """Test example output formatting"""

    def test_format_example_basic(self, sample_example):
        """Test basic example formatting"""
        text = ResultsFormatter.format_example_output(sample_example)

        assert isinstance(text, str)
        assert len(text) > 0

    def test_format_example_includes_question(self, sample_example):
        """Test that formatted output includes question"""
        text = ResultsFormatter.format_example_output(sample_example)

        assert 'Question' in text
        assert sample_example['question'] in text

    def test_format_example_includes_expected_answer(self, sample_example):
        """Test that formatted output includes expected answer"""
        text = ResultsFormatter.format_example_output(sample_example)

        assert 'Expected Answer' in text
        assert sample_example['expected_answer'] in text

    def test_format_example_includes_llm_response(self, sample_example):
        """Test that formatted output includes LLM response"""
        text = ResultsFormatter.format_example_output(sample_example)

        assert 'LLM Response' in text
        assert sample_example['llm_response'] in text

    def test_format_example_includes_similarity(self, sample_example):
        """Test that formatted output includes similarity score"""
        text = ResultsFormatter.format_example_output(sample_example)

        assert 'Similarity Score' in text
        assert '0.9245' in text

    def test_format_example_with_empty_dict(self):
        """Test formatting with empty example"""
        text = ResultsFormatter.format_example_output({})

        assert 'No example available' in text

    def test_format_example_with_none(self):
        """Test formatting with None"""
        text = ResultsFormatter.format_example_output(None)

        assert 'No example available' in text

    def test_format_example_markdown_structure(self, sample_example):
        """Test that output uses markdown formatting"""
        text = ResultsFormatter.format_example_output(sample_example)

        # Should have markdown bold markers
        assert '**' in text

    def test_format_example_with_long_response(self):
        """Test formatting with long LLM response"""
        example = {
            'question': 'Explain gravity',
            'expected_answer': 'Force that attracts objects',
            'llm_response': 'Gravity is a fundamental force...' + 'x' * 500,
            'similarity_score': 0.85
        }

        text = ResultsFormatter.format_example_output(example)

        # Should handle long text
        assert 'Gravity is a fundamental force' in text

    def test_format_example_with_special_characters(self):
        """Test formatting with special characters"""
        example = {
            'question': 'What is <html>?',
            'expected_answer': 'Markup language',
            'llm_response': 'HTML is <markup>',
            'similarity_score': 0.75
        }

        text = ResultsFormatter.format_example_output(example)

        # Should preserve special characters
        assert '<html>' in text
        assert '<markup>' in text


class TestFormatterIntegration:
    """Integration tests for results formatter"""

    def test_all_methods_are_static(self):
        """Test that all methods can be called without instantiation"""
        # Should not need to create instance
        ResultsFormatter.format_dataset_name('test')
        ResultsFormatter.format_technique_name('test')

        techniques = {'baseline': {'scores': [0.6]}}
        ResultsFormatter.generate_performance_table('test', techniques)
        ResultsFormatter.generate_statistical_significance('test', techniques)

        example = {'question': 'Q', 'expected_answer': 'A',
                  'llm_response': 'R', 'similarity_score': 0.8}
        ResultsFormatter.format_example_output(example)

    def test_formatter_consistency(self, sample_techniques):
        """Test that multiple calls produce same output"""
        table1 = ResultsFormatter.generate_performance_table(
            'test', sample_techniques
        )
        table2 = ResultsFormatter.generate_performance_table(
            'test', sample_techniques
        )

        assert table1 == table2

    def test_complete_formatting_pipeline(self, sample_techniques, sample_example):
        """Test using all formatter methods together"""
        dataset_name = ResultsFormatter.format_dataset_name('test_dataset')
        table = ResultsFormatter.generate_performance_table(
            'test_dataset', sample_techniques
        )
        significance = ResultsFormatter.generate_statistical_significance(
            'test_dataset', sample_techniques
        )
        example_text = ResultsFormatter.format_example_output(sample_example)

        # All should be non-empty strings
        assert all(isinstance(x, str) and len(x) > 0
                  for x in [dataset_name, table, significance, example_text])

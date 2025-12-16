"""Format experimental results for markdown output"""

from typing import Dict, List
from .stats_calculator import StatisticsCalculator
from .statistical_tests import StatisticalTests


class ResultsFormatter:
    """Format results into markdown tables and sections"""

    @staticmethod
    def format_dataset_name(dataset: str) -> str:
        """Format dataset name for display

        Args:
            dataset: Raw dataset name

        Returns:
            Formatted display name
        """
        return dataset.replace('_', ' ').title()

    @staticmethod
    def format_technique_name(technique: str) -> str:
        """Format technique name for display

        Args:
            technique: Raw technique name

        Returns:
            Formatted display name
        """
        return technique.replace('_', ' ').title()

    @staticmethod
    def generate_performance_table(dataset: str, techniques: Dict) -> str:
        """Generate performance metrics table

        Args:
            dataset: Dataset name
            techniques: Dict mapping technique names to their results

        Returns:
            Markdown table string
        """
        baseline_stats = None
        if 'baseline' in techniques:
            baseline_stats = StatisticsCalculator.calculate_statistics(
                techniques['baseline']['scores']
            )

        table = "| Technique | Mean Similarity | Std Dev | Min | Max | "
        table += "Sample Size | Improvement vs Baseline |\n"
        table += "|-----------|----------------|---------|-----|-----|"
        table += "-------------|------------------------|\n"

        # Order: baseline, standard, few_shot, chain_of_thought
        technique_order = [
            'baseline', 'standard_improved', 'standard',
            'few_shot', 'chain_of_thought'
        ]

        for technique in technique_order:
            if technique not in techniques:
                continue

            stats = StatisticsCalculator.calculate_statistics(
                techniques[technique]['scores']
            )
            technique_name = ResultsFormatter.format_technique_name(technique)

            if technique in ['baseline', 'standard'] or baseline_stats is None:
                improvement = "-"
            else:
                improvement_pct = StatisticsCalculator.calculate_improvement(
                    baseline_stats['mean'], stats['mean']
                )
                sign = "+" if improvement_pct > 0 else ""
                improvement = f"{sign}{improvement_pct:.2f}%"

            table += f"| {technique_name} | {stats['mean']:.4f} | "
            table += f"{stats['std']:.4f} | {stats['min']:.4f} | "
            table += f"{stats['max']:.4f} | {stats['count']} | {improvement} |\n"

        return table

    @staticmethod
    def generate_statistical_significance(dataset: str, techniques: Dict) -> str:
        """Generate statistical significance section

        Args:
            dataset: Dataset name
            techniques: Dict mapping technique names to their results

        Returns:
            Markdown formatted statistical test results
        """
        if 'baseline' not in techniques:
            return "*No baseline results available for comparison*\n"

        baseline_scores = techniques['baseline']['scores']
        text = ""

        for technique in ['standard_improved', 'standard', 'few_shot', 'chain_of_thought']:
            if technique not in techniques:
                continue

            comparison = StatisticalTests.compare_to_baseline(
                baseline_scores,
                techniques[technique]['scores'],
                technique
            )

            sig_text = "✓ Significant" if comparison['significant'] else "✗ Not significant"
            technique_name = ResultsFormatter.format_technique_name(technique)

            text += f"- **{technique_name} vs Baseline**: "
            text += f"p-value = {comparison['p_value']:.4f} ({sig_text}), "
            text += f"Cohen's d = {comparison['cohens_d']:.3f} "
            text += f"({comparison['effect_size']} effect)\n"

        return text

    @staticmethod
    def format_example_output(example: Dict) -> str:
        """Format a single example output

        Args:
            example: Example result dictionary

        Returns:
            Formatted markdown string
        """
        if not example:
            return "*No example available*\n"

        text = f"**Question:** {example['question']}\n\n"
        text += f"**Expected Answer:** {example['expected_answer']}\n\n"
        text += f"**LLM Response:** {example['llm_response']}\n\n"
        text += f"**Similarity Score:** {example['similarity_score']:.4f}\n"

        return text

"""Orchestrate results document generation - main coordinator"""

from pathlib import Path
from datetime import datetime
from typing import Dict

from .results_loader import ResultsLoader
from .stats_calculator import StatisticsCalculator
from .results_formatter import ResultsFormatter
from .insights_generator import InsightsGenerator


class ResultsUpdater:
    """Coordinate results loading, analysis, and document generation"""

    def __init__(self, results_dir: Path = Path("results/experiments")):
        """Initialize results updater

        Args:
            results_dir: Directory containing experiment JSON files
        """
        self.loader = ResultsLoader(results_dir)
        self.results_loaded = False

    def load_results(self) -> bool:
        """Load all experiment results

        Returns:
            True if results were loaded successfully
        """
        self.results_loaded = self.loader.load_all_results()
        return self.results_loaded

    def generate_results_document(self) -> str:
        """Generate complete RESULTS.md content

        Returns:
            Markdown formatted results document
        """
        if not self.results_loaded:
            return "# No Results Available\n\nPlease run experiments first.\n"

        doc = self._generate_header()
        doc += self._generate_executive_summary()
        doc += self._generate_dataset_sections()
        doc += self._generate_conclusions()

        return doc

    def _generate_header(self) -> str:
        """Generate document header"""
        header = "# Experimental Results & Analysis\n\n"
        header += "**Project:** Prompt Engineering for Mass Production Optimization\n"
        header += f"**Date:** {datetime.now().strftime('%B %d, %Y')}\n"
        header += "**Execution Engine:** Ollama (llama3.2)\n\n"
        header += "---\n\n"
        return header

    def _generate_executive_summary(self) -> str:
        """Generate executive summary section"""
        summary = "## Executive Summary\n\n"
        summary += ("This document presents results from our systematic "
                   "investigation of prompt engineering techniques' impact "
                   "on LLM response accuracy and consistency.\n\n")
        summary += "### Key Findings\n\n"

        insights = InsightsGenerator.generate_overall_insights(
            self.loader.results_by_dataset
        )
        for idx, insight in enumerate(insights, 1):
            summary += f"{idx}. {insight}\n"

        summary += "\n---\n\n"
        return summary

    def _generate_dataset_sections(self) -> str:
        """Generate sections for each dataset"""
        content = ""

        for dataset in self.loader.get_all_datasets():
            content += self._generate_single_dataset_section(dataset)

        return content

    def _generate_single_dataset_section(self, dataset: str) -> str:
        """Generate section for a single dataset"""
        techniques = self.loader.get_dataset_techniques(dataset)
        dataset_name = ResultsFormatter.format_dataset_name(dataset)

        section = f"## {dataset_name}\n\n### Performance Metrics\n\n"
        section += ResultsFormatter.generate_performance_table(dataset, techniques) + "\n"
        section += "### Statistical Significance\n\n"
        section += ResultsFormatter.generate_statistical_significance(dataset, techniques) + "\n"
        section += "### Key Insights\n\n"
        section += InsightsGenerator.generate_dataset_insights(dataset, techniques) + "\n"

        example = self.loader.get_example_output(dataset, 'few_shot', 0)
        if example:
            section += "### Example\n\n" + ResultsFormatter.format_example_output(example)

        return section + "\n---\n\n"

    def _generate_conclusions(self) -> str:
        """Generate conclusions section"""
        conclusions = "## Conclusions\n\n"

        recommendations = InsightsGenerator.generate_recommendations(
            self.loader.results_by_dataset
        )

        if recommendations:
            conclusions += "### Recommendations\n\n"
            for idx, rec in enumerate(recommendations, 1):
                conclusions += f"{idx}. {rec}\n"
            conclusions += "\n"

        conclusions += "### Methodology\n\n"
        conclusions += "- Cosine similarity (sentence-transformers)\n"
        conclusions += "- T-test & Cohen's d (α = 0.05)\n"
        conclusions += "- Ollama llama3.2 (local)\n\n---\n\n"
        conclusions += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"

        return conclusions

    def update_results_file(self, output_path: Path = Path("RESULTS.md")) -> None:
        """Generate and write RESULTS.md file

        Args:
            output_path: Path where to write the results file
        """
        content = self.generate_results_document()

        with open(output_path, 'w') as f:
            f.write(content)

        print(f"✓ Results updated: {output_path}")

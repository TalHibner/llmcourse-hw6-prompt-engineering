#!/usr/bin/env python3
"""Analyze results and generate visualizations"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import StatisticalAnalyzer, VisualizationGenerator, ResultsUpdater
from src.utils.logging_config import setup_logging
import logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def main():
    """Analyze results and generate visualizations"""
    logger.info("=" * 80)
    logger.info("RESULTS ANALYSIS & VISUALIZATION")
    logger.info("=" * 80)

    # Load all results
    results_dir = Path("results/experiments")
    result_files = list(results_dir.glob("*.json"))

    if not result_files:
        logger.error("No result files found! Please run experiments first.")
        return

    logger.info(f"Found {len(result_files)} result files")

    # Organize results by dataset and technique
    results_by_dataset = defaultdict(dict)

    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)

        dataset = data['configuration']['dataset']
        technique = data['configuration']['technique']

        # Extract similarity scores
        similarities = [r['similarity_score'] for r in data['results']]
        results_by_dataset[dataset][technique] = similarities

    # Create visualization generator
    viz_gen = VisualizationGenerator()
    analyzer = StatisticalAnalyzer()

    # Generate visualizations for each dataset
    for dataset, techniques in results_by_dataset.items():
        logger.info(f"\nAnalyzing dataset: {dataset}")

        # Generate histograms for each technique
        for technique, similarities in techniques.items():
            logger.info(f"  - {technique}: mean={sum(similarities)/len(similarities):.3f}")

            viz_gen.plot_histogram(
                similarities,
                title=f"{technique.replace('_', ' ').title()} on {dataset.replace('_', ' ').title()}",
                filename=f"histogram_{dataset}_{technique}.png"
            )

        # Generate comparison bar chart
        viz_gen.plot_comparison_bars(
            techniques,
            title=f"Technique Comparison: {dataset.replace('_', ' ').title()} Dataset",
            filename=f"comparison_{dataset}.png"
        )

        # Generate box plots
        viz_gen.plot_box_plots(
            techniques,
            title=f"Distribution Comparison: {dataset.replace('_', ' ').title()} Dataset",
            filename=f"boxplot_{dataset}.png"
        )

        # Perform statistical comparison if baseline exists
        if 'baseline' in techniques:
            baseline_scores = techniques['baseline']
            logger.info(f"\n  Statistical Comparison to Baseline:")

            for technique, scores in techniques.items():
                if technique != 'baseline':
                    comparison = analyzer.compare_techniques(baseline_scores, scores)
                    logger.info(f"\n    {technique}:")
                    logger.info(f"      Improvement: {comparison['improvement_pct']:.2f}%")
                    logger.info(f"      p-value: {comparison['p_value']:.4f}")
                    logger.info(f"      Significant: {comparison['significant']}")
                    logger.info(f"      Effect size (Cohen's d): {comparison['cohens_d']:.3f}")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Visualizations saved to: results/visualizations/")
    logger.info("\nGenerated files:")
    viz_dir = Path("results/visualizations")
    for viz_file in sorted(viz_dir.glob("*.png")):
        logger.info(f"  - {viz_file.name}")

    # Update RESULTS.md with actual findings
    logger.info("\n" + "=" * 80)
    logger.info("UPDATING RESULTS.md")
    logger.info("=" * 80)

    updater = ResultsUpdater()
    success = updater.update_results_file()

    if success:
        logger.info("\n✓ RESULTS.md has been automatically updated with experimental findings!")
        logger.info("  Review RESULTS.md to see the complete analysis.")
    else:
        logger.error("\n✗ Failed to update RESULTS.md")

    logger.info("\n" + "=" * 80)
    logger.info("ALL TASKS COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

"""Visualization generation"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict


class VisualizationGenerator:
    """Generate publication-quality visualizations"""

    def __init__(self, output_dir: str = "results/visualizations"):
        """Initialize visualization generator

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 11

    def plot_histogram(
        self,
        similarities: List[float],
        title: str,
        filename: str
    ):
        """Plot similarity score distribution

        Args:
            similarities: List of similarity scores
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.axvline(
            np.mean(similarities),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {np.mean(similarities):.3f}'
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_comparison_bars(
        self,
        technique_results: Dict[str, List[float]],
        title: str,
        filename: str
    ):
        """Bar chart comparing techniques

        Args:
            technique_results: Dictionary mapping technique names to scores
            title: Plot title
            filename: Output filename
        """
        techniques = list(technique_results.keys())
        means = [np.mean(scores) for scores in technique_results.values()]
        stds = [np.std(scores) for scores in technique_results.values()]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(techniques, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        plt.xlabel('Prompt Technique')
        plt.ylabel('Mean Similarity Score')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{mean:.3f}',
                ha='center',
                va='bottom'
            )

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_box_plots(
        self,
        technique_results: Dict[str, List[float]],
        title: str,
        filename: str
    ):
        """Box plots showing distributions

        Args:
            technique_results: Dictionary mapping technique names to scores
            title: Plot title
            filename: Output filename
        """
        data = list(technique_results.values())
        labels = list(technique_results.keys())

        plt.figure(figsize=(10, 6))
        box_plot = plt.boxplot(data, labels=labels, patch_artist=True)

        # Color the boxes
        for patch in box_plot['boxes']:
            patch.set_facecolor('lightblue')

        plt.xlabel('Prompt Technique')
        plt.ylabel('Similarity Score')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

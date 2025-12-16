"""Generate insights from experimental results"""

from typing import Dict, List
from .stats_calculator import StatisticsCalculator


class InsightsGenerator:
    """Generate human-readable insights from experiment results"""

    @staticmethod
    def generate_dataset_insights(dataset: str, techniques: Dict) -> str:
        """Generate insights for a specific dataset

        Args:
            dataset: Dataset name
            techniques: Dict mapping technique names to their results

        Returns:
            Markdown formatted insights
        """
        insights = []

        if 'baseline' not in techniques:
            return "*Insufficient data for insights*\n"

        baseline_stats = StatisticsCalculator.calculate_statistics(
            techniques['baseline']['scores']
        )
        baseline_mean = baseline_stats['mean']

        # Find best performing technique
        best_technique, best_mean, improvement = StatisticsCalculator.find_best_technique(
            {k: v for k, v in techniques.items() if k != 'baseline'},
            baseline_mean
        )

        if best_technique and improvement > 0:
            technique_name = best_technique.replace('_', ' ').title()
            insights.append(
                f"**{technique_name}** performed best with "
                f"{improvement:.1f}% improvement over baseline"
            )
        else:
            insights.append("No technique significantly outperformed baseline")

        # Check consistency (lower std = better)
        most_consistent, min_std = StatisticsCalculator.find_most_consistent(techniques)

        if most_consistent and most_consistent != 'baseline':
            technique_name = most_consistent.replace('_', ' ').title()
            insights.append(
                f"**{technique_name}** showed most consistent "
                f"performance (std={min_std:.4f})"
            )

        # Check for poor performance (5% worse than baseline)
        for technique, data in techniques.items():
            if technique == 'baseline':
                continue
            stats = StatisticsCalculator.calculate_statistics(data['scores'])
            if stats['mean'] < baseline_mean * 0.95:
                technique_name = technique.replace('_', ' ').title()
                decline = abs(StatisticsCalculator.calculate_improvement(
                    baseline_mean, stats['mean']))
                insights.append(f"⚠️ **{technique_name}** {decline:.1f}% worse")

        return "\n".join(f"- {insight}" for insight in insights)

    @staticmethod
    def generate_overall_insights(all_results: Dict) -> List[str]:
        """Generate overall insights across all datasets

        Args:
            all_results: Dict mapping datasets to their technique results

        Returns:
            List of insight strings
        """
        insights = []

        # Aggregate all baseline vs improved scores
        all_baselines = []
        all_improved = []

        for dataset, techniques in all_results.items():
            if 'baseline' in techniques:
                all_baselines.extend(techniques['baseline']['scores'])
                for tech, data in techniques.items():
                    if tech != 'baseline':
                        all_improved.extend(data['scores'])

        if all_baselines and all_improved:
            baseline_mean = StatisticsCalculator.calculate_statistics(all_baselines)['mean']
            improved_mean = StatisticsCalculator.calculate_statistics(all_improved)['mean']
            overall_improvement = StatisticsCalculator.calculate_improvement(
                baseline_mean, improved_mean
            )

            if overall_improvement > 0:
                insights.append(
                    f"**Overall Improvement**: Prompt engineering techniques showed "
                    f"{overall_improvement:.1f}% average improvement over baseline"
                )
            else:
                insights.append(
                    f"**Overall Result**: Prompt engineering techniques did not show "
                    f"consistent improvement over baseline"
                )

        # Count significant improvements
        return insights

    @staticmethod
    def generate_recommendations(all_results: Dict) -> List[str]:
        """Generate recommendations based on results

        Args:
            all_results: Dict mapping datasets to their technique results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Analyze which technique works best overall
        technique_totals = {}

        for dataset, techniques in all_results.items():
            for technique, data in techniques.items():
                if technique not in technique_totals:
                    technique_totals[technique] = []
                technique_totals[technique].extend(data['scores'])

        # Find best overall technique
        best_overall = None
        best_mean = 0

        for technique, scores in technique_totals.items():
            if technique == 'baseline':
                continue
            stats = StatisticsCalculator.calculate_statistics(scores)
            if stats['mean'] > best_mean:
                best_mean = stats['mean']
                best_overall = technique

        if best_overall:
            technique_name = best_overall.replace('_', ' ').title()
            recommendations.append(
                f"**{technique_name}** shows the strongest overall performance "
                f"and is recommended for production use"
            )

        return recommendations

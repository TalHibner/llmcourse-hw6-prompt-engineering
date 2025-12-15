#!/usr/bin/env python3
"""Main experiment runner - Uses Ollama by default"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers.ollama_provider import OllamaProvider
from src.techniques import (
    BaselinePrompt,
    StandardPrompt,
    FewShotPrompt,
    ChainOfThoughtPrompt
)
from src.datasets.generator import DatasetGenerator
from src.evaluation import SimilarityCalculator, ExperimentRunner
from src.utils.logging_config import setup_logging
import logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def main():
    """Run experiments"""
    parser = argparse.ArgumentParser(description="Run prompt engineering experiments")
    parser.add_argument(
        "--technique",
        type=str,
        default="all",
        choices=["all", "baseline", "standard", "few_shot", "chain_of_thought"],
        help="Prompt technique to test"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "sentiment", "cot"],
        help="Dataset to use"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PROMPT ENGINEERING RESEARCH EXPERIMENT")
    logger.info("=" * 80)

    # Initialize provider (Ollama - PRIMARY)
    logger.info("Initializing Ollama provider...")
    provider = OllamaProvider(model="llama3.2")

    if not provider.is_available():
        logger.error("Ollama is not available! Please ensure Ollama is running.")
        logger.error("Install: https://ollama.ai")
        logger.error("Start: ollama serve")
        logger.error("Pull model: ollama pull llama3.2")
        return

    logger.info(f"✓ Ollama is available: {provider.get_model_name()}")

    # Initialize similarity calculator
    logger.info("Loading embedding model...")
    calculator = SimilarityCalculator()
    logger.info("✓ Embedding model loaded")

    # Load datasets
    logger.info("Loading datasets...")
    data_dir = Path("data/datasets")

    datasets_to_run = []
    if args.dataset in ["all", "sentiment"]:
        sentiment_path = data_dir / "sentiment_analysis.json"
        if sentiment_path.exists():
            sentiment_dataset = DatasetGenerator.load_dataset(sentiment_path)
            datasets_to_run.append(("sentiment", sentiment_dataset))
            logger.info(f"✓ Loaded sentiment dataset: {len(sentiment_dataset)} examples")
        else:
            logger.warning("Sentiment dataset not found. Run generate_datasets.py first.")

    if args.dataset in ["all", "cot"]:
        cot_path = data_dir / "chain_of_thought.json"
        if cot_path.exists():
            cot_dataset = DatasetGenerator.load_dataset(cot_path)
            datasets_to_run.append(("cot", cot_dataset))
            logger.info(f"✓ Loaded CoT dataset: {len(cot_dataset)} examples")
        else:
            logger.warning("CoT dataset not found. Run generate_datasets.py first.")

    if not datasets_to_run:
        logger.error("No datasets found! Please run generate_datasets.py first.")
        return

    # Define techniques to test
    techniques_map = {
        "baseline": BaselinePrompt(),
        "standard": StandardPrompt(),
        "few_shot": FewShotPrompt(),
        "chain_of_thought": ChainOfThoughtPrompt(),
    }

    if args.technique == "all":
        techniques_to_run = list(techniques_map.items())
    else:
        techniques_to_run = [(args.technique, techniques_map[args.technique])]

    logger.info(f"Will run {len(techniques_to_run)} techniques on {len(datasets_to_run)} datasets")

    # Run experiments
    all_results = []
    results_dir = Path("results/experiments")
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset in datasets_to_run:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"DATASET: {dataset_name.upper()}")
        logger.info(f"{'=' * 80}")

        for technique_name, technique in techniques_to_run:
            logger.info(f"\n--- Running: {technique_name} ---")

            # Create experiment runner
            runner = ExperimentRunner(provider, technique, calculator)

            # Run experiment
            result = runner.run_experiment(dataset)
            all_results.append(result)

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = results_dir / f"{dataset_name}_{technique_name}_{timestamp}.json"

            result_data = {
                "experiment_id": f"{dataset_name}_{technique_name}_{timestamp}",
                "timestamp": datetime.now().isoformat(),
                "configuration": {
                    "provider": result.provider,
                    "technique": result.technique,
                    "dataset": result.dataset
                },
                "results": result.results,
                "metrics": result.metrics
            }

            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)

            logger.info(f"✓ Saved results to {result_file}")
            logger.info(f"  Mean similarity: {result.metrics['mean_similarity']:.3f}")
            logger.info(f"  Std deviation: {result.metrics['std_similarity']:.3f}")

    logger.info("\n" + "=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total experiments run: {len(all_results)}")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Run: python scripts/analyze_results.py")
    logger.info("  2. Check visualizations in: results/visualizations/")


if __name__ == "__main__":
    main()

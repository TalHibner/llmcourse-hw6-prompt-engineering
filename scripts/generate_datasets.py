#!/usr/bin/env python3
"""Generate datasets for experiments"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.generator import DatasetGenerator
from src.utils.logging_config import setup_logging
import logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def main():
    """Generate and save datasets"""
    output_dir = Path("data/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating sentiment analysis dataset...")
    sentiment_dataset = DatasetGenerator.generate_sentiment_analysis()
    sentiment_path = output_dir / "sentiment_analysis.json"
    DatasetGenerator.save_dataset(sentiment_dataset, sentiment_path)
    logger.info(f"Saved {len(sentiment_dataset)} examples to {sentiment_path}")

    logger.info("Generating chain-of-thought dataset...")
    cot_dataset = DatasetGenerator.generate_chain_of_thought()
    cot_path = output_dir / "chain_of_thought.json"
    DatasetGenerator.save_dataset(cot_dataset, cot_path)
    logger.info(f"Saved {len(cot_dataset)} examples to {cot_path}")

    logger.info("Dataset generation complete!")
    logger.info(f"Total examples: {len(sentiment_dataset) + len(cot_dataset)}")


if __name__ == "__main__":
    main()

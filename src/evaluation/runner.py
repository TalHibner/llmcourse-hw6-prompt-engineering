"""Experiment runner"""

import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any
from tqdm import tqdm

from ..providers.base import LLMProvider
from ..techniques.base import PromptTechnique
from ..datasets.base import Dataset, DatasetExample
from .similarity import SimilarityCalculator

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    technique: str
    provider: str
    dataset: str
    results: List[Dict[str, Any]]
    metrics: Dict[str, float]


class ExperimentRunner:
    """Run experiments and collect results"""

    def __init__(
        self,
        provider: LLMProvider,
        technique: PromptTechnique,
        calculator: SimilarityCalculator
    ):
        """Initialize experiment runner

        Args:
            provider: LLM provider to use
            technique: Prompt technique to apply
            calculator: Similarity calculator
        """
        self.provider = provider
        self.technique = technique
        self.calculator = calculator

    def run_experiment(self, dataset: Dataset) -> ExperimentResult:
        """Run experiment on dataset

        Args:
            dataset: Dataset to evaluate

        Returns:
            Experiment results
        """
        logger.info(
            f"Running experiment: {self.technique.get_name()} "
            f"on {dataset.name} with {self.provider.get_model_name()}"
        )

        results = []
        similarities = []

        # Run through dataset with progress bar
        for example in tqdm(dataset.examples, desc=f"{self.technique.get_name()}"):
            try:
                # Format prompt
                prompt = self.technique.format_prompt(example.question)

                # Get LLM response
                start_time = time.time()
                response = self.provider.generate(prompt)
                elapsed_time = time.time() - start_time

                # Calculate similarity
                similarity = self.calculator.calculate_similarity(
                    response,
                    example.expected_answer
                )

                similarities.append(similarity)

                results.append({
                    'example_id': example.id,
                    'question': example.question,
                    'expected_answer': example.expected_answer,
                    'actual_answer': response,
                    'similarity_score': similarity,
                    'execution_time_ms': int(elapsed_time * 1000)
                })

                logger.debug(f"Example {example.id}: similarity={similarity:.3f}")

            except Exception as e:
                logger.error(f"Error processing example {example.id}: {e}")
                results.append({
                    'example_id': example.id,
                    'question': example.question,
                    'expected_answer': example.expected_answer,
                    'actual_answer': f"ERROR: {str(e)}",
                    'similarity_score': 0.0,
                    'execution_time_ms': 0
                })
                similarities.append(0.0)

        # Calculate metrics
        import numpy as np
        metrics = {
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'median_similarity': float(np.median(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'total_examples': len(dataset.examples),
            'successful': sum(1 for r in results if r['similarity_score'] > 0),
            'failed': sum(1 for r in results if r['similarity_score'] == 0)
        }

        logger.info(f"Completed: mean={metrics['mean_similarity']:.3f}, std={metrics['std_similarity']:.3f}")

        return ExperimentResult(
            technique=self.technique.get_name(),
            provider=self.provider.get_model_name(),
            dataset=dataset.name,
            results=results,
            metrics=metrics
        )

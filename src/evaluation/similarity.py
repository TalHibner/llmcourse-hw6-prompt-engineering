"""Similarity calculation using embeddings"""

import logging
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Calculate similarity between texts using embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with embedding model

        Args:
            model_name: HuggingFace model name
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def batch_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """Calculate similarities for batches

        Args:
            texts1: List of first texts
            texts2: List of second texts

        Returns:
            List of similarity scores
        """
        if len(texts1) != len(texts2):
            raise ValueError("Input lists must have same length")

        embeddings1 = self.model.encode(texts1)
        embeddings2 = self.model.encode(texts2)

        similarities = [
            cosine_similarity([e1], [e2])[0][0]
            for e1, e2 in zip(embeddings1, embeddings2)
        ]

        return [float(s) for s in similarities]

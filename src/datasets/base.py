"""Dataset structures"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class DatasetExample:
    """Single example in a dataset"""
    id: str
    question: str
    expected_answer: str
    category: str
    metadata: Dict[str, Any]


@dataclass
class Dataset:
    """Collection of examples"""
    name: str
    description: str
    examples: List[DatasetExample]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DatasetExample:
        return self.examples[idx]

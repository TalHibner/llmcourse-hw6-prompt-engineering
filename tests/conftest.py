"""Pytest configuration and shared fixtures."""
import pytest
from typing import Dict, List


@pytest.fixture
def sample_question() -> str:
    """Sample question for testing."""
    return "What is the capital of France?"


@pytest.fixture
def sample_answer() -> str:
    """Sample expected answer for testing."""
    return "Paris"


@pytest.fixture
def sample_context() -> str:
    """Sample context for testing."""
    return "France is a country in Western Europe."


@pytest.fixture
def sample_dataset() -> List[Dict]:
    """Sample dataset for testing."""
    return [
        {
            "id": 1,
            "question": "What is 2+2?",
            "expected_answer": "4",
            "context": "Basic arithmetic",
            "category": "math",
            "difficulty": "easy"
        },
        {
            "id": 2,
            "question": "What is the capital of Italy?",
            "expected_answer": "Rome",
            "context": "European geography",
            "category": "geography",
            "difficulty": "easy"
        }
    ]


@pytest.fixture
def sample_llm_response() -> str:
    """Sample LLM response for testing."""
    return "The capital of France is Paris."


@pytest.fixture
def sample_config() -> Dict:
    """Sample configuration for testing."""
    return {
        "model": "llama3.2",
        "temperature": 0.7,
        "timeout": 30
    }

"""Tests for base LLM provider interface."""
import pytest
from src.providers.base import LLMProvider


class MockProvider(LLMProvider):
    """Mock provider for testing abstract base class."""

    def generate(self, prompt: str, **kwargs) -> str:
        return f"Mock response to: {prompt}"

    def get_model_name(self) -> str:
        return "mock:test-model"

    def is_available(self) -> bool:
        return True


def test_provider_interface():
    """Test that provider implements required interface."""
    provider = MockProvider()

    # Test generate method
    response = provider.generate("Test prompt")
    assert isinstance(response, str)
    assert "Test prompt" in response

    # Test get_model_name method
    model_name = provider.get_model_name()
    assert isinstance(model_name, str)
    assert "mock" in model_name

    # Test is_available method
    available = provider.is_available()
    assert isinstance(available, bool)
    assert available is True


def test_provider_cannot_be_instantiated():
    """Test that abstract base class cannot be instantiated."""
    with pytest.raises(TypeError):
        LLMProvider()


def test_provider_kwargs():
    """Test that provider accepts additional kwargs."""
    provider = MockProvider()
    response = provider.generate("Test", temperature=0.5, max_tokens=100)
    assert isinstance(response, str)

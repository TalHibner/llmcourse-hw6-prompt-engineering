"""Tests for similarity calculator."""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.evaluation.similarity import SimilarityCalculator


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer to avoid loading real model."""
    with patch('src.evaluation.similarity.SentenceTransformer') as mock:
        mock_model = Mock()
        mock.return_value = mock_model
        yield mock_model


@pytest.fixture
def calculator(mock_sentence_transformer):
    """Create similarity calculator with mocked model."""
    return SimilarityCalculator()


class TestSimilarityCalculatorInitialization:
    """Tests for SimilarityCalculator initialization."""

    @patch('src.evaluation.similarity.SentenceTransformer')
    def test_default_initialization(self, mock_st):
        """Test initialization with default model."""
        calc = SimilarityCalculator()

        # Should load the default model
        mock_st.assert_called_once_with("all-MiniLM-L6-v2")

    @patch('src.evaluation.similarity.SentenceTransformer')
    def test_custom_model(self, mock_st):
        """Test initialization with custom model."""
        calc = SimilarityCalculator("custom-model")

        mock_st.assert_called_once_with("custom-model")


class TestCalculateSimilarity:
    """Tests for calculate_similarity method."""

    def test_identical_texts(self, calculator, mock_sentence_transformer):
        """Test similarity of identical texts."""
        # Mock embeddings for identical texts (should be very similar)
        mock_sentence_transformer.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])

        similarity = calculator.calculate_similarity("test", "test")

        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_different_texts(self, calculator, mock_sentence_transformer):
        """Test similarity of different texts."""
        # Mock embeddings for different texts
        mock_sentence_transformer.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

        similarity = calculator.calculate_similarity("hello", "goodbye")

        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert similarity == pytest.approx(0.0, abs=0.01)

    def test_similar_texts(self, calculator, mock_sentence_transformer):
        """Test similarity of similar texts."""
        # Mock embeddings for similar texts
        mock_sentence_transformer.encode.return_value = np.array([
            [1.0, 0.5, 0.0],
            [0.9, 0.6, 0.1]
        ])

        similarity = calculator.calculate_similarity(
            "The cat is sleeping",
            "The cat is resting"
        )

        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_empty_strings(self, calculator, mock_sentence_transformer):
        """Test with empty strings."""
        mock_sentence_transformer.encode.return_value = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        # Should handle empty strings without error
        similarity = calculator.calculate_similarity("", "")
        assert isinstance(similarity, float)

    def test_returns_float(self, calculator, mock_sentence_transformer):
        """Test that return value is always float."""
        mock_sentence_transformer.encode.return_value = np.array([
            [1.0, 0.0],
            [1.0, 0.0]
        ])

        result = calculator.calculate_similarity("a", "b")
        assert type(result) == float


class TestBatchSimilarity:
    """Tests for batch_similarity method."""

    def test_batch_equal_lengths(self, calculator, mock_sentence_transformer):
        """Test batch similarity with equal length lists."""
        texts1 = ["text1", "text2", "text3"]
        texts2 = ["text4", "text5", "text6"]

        # Mock embeddings
        mock_sentence_transformer.encode.side_effect = [
            np.array([[1, 0], [0, 1], [1, 1]]),  # embeddings1
            np.array([[1, 0], [1, 0], [0, 1]])   # embeddings2
        ]

        similarities = calculator.batch_similarity(texts1, texts2)

        assert isinstance(similarities, list)
        assert len(similarities) == 3
        assert all(isinstance(s, float) for s in similarities)
        assert all(0.0 <= s <= 1.0 for s in similarities)

    def test_batch_unequal_lengths_raises_error(self, calculator, mock_sentence_transformer):
        """Test that unequal lengths raise ValueError."""
        texts1 = ["text1", "text2"]
        texts2 = ["text3", "text4", "text5"]

        with pytest.raises(ValueError, match="same length"):
            calculator.batch_similarity(texts1, texts2)

    def test_batch_empty_lists(self, calculator, mock_sentence_transformer):
        """Test batch with empty lists."""
        mock_sentence_transformer.encode.side_effect = [
            np.array([]),
            np.array([])
        ]

        similarities = calculator.batch_similarity([], [])

        assert isinstance(similarities, list)
        assert len(similarities) == 0

    def test_batch_single_item(self, calculator, mock_sentence_transformer):
        """Test batch with single item."""
        texts1 = ["hello"]
        texts2 = ["world"]

        mock_sentence_transformer.encode.side_effect = [
            np.array([[1.0, 0.0]]),
            np.array([[0.5, 0.5]])
        ]

        similarities = calculator.batch_similarity(texts1, texts2)

        assert len(similarities) == 1
        assert isinstance(similarities[0], float)

    def test_batch_returns_list_of_floats(self, calculator, mock_sentence_transformer):
        """Test that batch returns list of floats."""
        texts1 = ["a", "b"]
        texts2 = ["c", "d"]

        mock_sentence_transformer.encode.side_effect = [
            np.array([[1, 0], [0, 1]]),
            np.array([[1, 0], [0, 1]])
        ]

        result = calculator.batch_similarity(texts1, texts2)

        assert all(type(s) == float for s in result)

    def test_batch_encode_called_correctly(self, calculator, mock_sentence_transformer):
        """Test that encode is called with correct arguments."""
        texts1 = ["text1", "text2"]
        texts2 = ["text3", "text4"]

        mock_sentence_transformer.encode.side_effect = [
            np.array([[1, 0], [0, 1]]),
            np.array([[1, 0], [0, 1]])
        ]

        calculator.batch_similarity(texts1, texts2)

        # Should be called twice (once for each list)
        assert mock_sentence_transformer.encode.call_count == 2

        # Check arguments
        calls = mock_sentence_transformer.encode.call_args_list
        assert calls[0][0][0] == texts1
        assert calls[1][0][0] == texts2


class TestSimilarityEdgeCases:
    """Tests for edge cases."""

    def test_very_long_texts(self, calculator, mock_sentence_transformer):
        """Test with very long texts."""
        long_text = "word " * 1000

        mock_sentence_transformer.encode.return_value = np.array([
            [1.0, 0.0],
            [1.0, 0.0]
        ])

        similarity = calculator.calculate_similarity(long_text, long_text)
        assert isinstance(similarity, float)

    def test_special_characters(self, calculator, mock_sentence_transformer):
        """Test with special characters."""
        text1 = "Hello! @#$%^&*()"
        text2 = "World? []{}<>"

        mock_sentence_transformer.encode.return_value = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        similarity = calculator.calculate_similarity(text1, text2)
        assert isinstance(similarity, float)

    def test_unicode_characters(self, calculator, mock_sentence_transformer):
        """Test with unicode characters."""
        text1 = "Hello 世界"
        text2 = "Bonjour 🌍"

        mock_sentence_transformer.encode.return_value = np.array([
            [1.0, 0.0],
            [0.8, 0.2]
        ])

        similarity = calculator.calculate_similarity(text1, text2)
        assert isinstance(similarity, float)

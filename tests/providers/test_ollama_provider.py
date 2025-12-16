"""Tests for Ollama provider."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.providers.ollama_provider import OllamaProvider
import requests


@pytest.fixture
def ollama_provider():
    """Create Ollama provider instance for testing."""
    return OllamaProvider(
        model="llama3.2",
        base_url="http://localhost:11434",
        timeout=30
    )


def test_ollama_provider_initialization():
    """Test Ollama provider initialization."""
    provider = OllamaProvider(model="mistral", timeout=60)

    assert provider.model == "mistral"
    assert provider.timeout == 60
    assert "localhost" in provider.base_url


def test_ollama_provider_initialization_strips_trailing_slash():
    """Test that trailing slash is removed from base_url."""
    provider = OllamaProvider(base_url="http://localhost:11434/")
    assert provider.base_url == "http://localhost:11434"


def test_get_model_name(ollama_provider):
    """Test get_model_name method."""
    model_name = ollama_provider.get_model_name()
    assert model_name == "ollama:llama3.2"
    assert "ollama:" in model_name


@patch('requests.post')
def test_generate_success(mock_post, ollama_provider):
    """Test successful generation."""
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": "The capital is Paris."
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    # Test generation
    result = ollama_provider.generate("What is the capital of France?")

    assert result == "The capital is Paris."
    assert mock_post.called
    assert mock_post.call_count == 1

    # Verify request parameters
    call_args = mock_post.call_args
    assert call_args[1]['timeout'] == 30
    assert 'json' in call_args[1]
    assert call_args[1]['json']['model'] == 'llama3.2'
    assert call_args[1]['json']['stream'] is False


@patch('requests.post')
def test_generate_with_kwargs(mock_post, ollama_provider):
    """Test generation with additional kwargs."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Test"}
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    result = ollama_provider.generate(
        "Test prompt",
        temperature=0.8,
        top_p=0.9
    )

    # Verify kwargs passed through
    call_args = mock_post.call_args
    payload = call_args[1]['json']
    assert payload['temperature'] == 0.8
    assert payload['top_p'] == 0.9


@patch('requests.post')
def test_generate_strips_whitespace(mock_post, ollama_provider):
    """Test that response whitespace is stripped."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": "  Test response  \n"
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    result = ollama_provider.generate("Test")
    assert result == "Test response"


@patch('requests.post')
def test_generate_request_exception(mock_post, ollama_provider):
    """Test handling of request exceptions."""
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

    with pytest.raises(RuntimeError) as exc_info:
        ollama_provider.generate("Test prompt")

    assert "Ollama generation failed" in str(exc_info.value)


@patch('requests.post')
def test_generate_http_error(mock_post, ollama_provider):
    """Test handling of HTTP errors."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
    mock_post.return_value = mock_response

    with pytest.raises(RuntimeError):
        ollama_provider.generate("Test prompt")


@patch('requests.post')
def test_generate_timeout(mock_post, ollama_provider):
    """Test handling of timeout."""
    mock_post.side_effect = requests.exceptions.Timeout("Request timeout")

    with pytest.raises(RuntimeError) as exc_info:
        ollama_provider.generate("Test prompt")

    assert "Ollama generation failed" in str(exc_info.value)


@patch('requests.get')
def test_is_available_success(mock_get, ollama_provider):
    """Test is_available when Ollama is running."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    assert ollama_provider.is_available() is True
    mock_get.assert_called_once()


@patch('requests.get')
def test_is_available_failure(mock_get, ollama_provider):
    """Test is_available when Ollama is not running."""
    mock_get.side_effect = requests.exceptions.ConnectionError()

    assert ollama_provider.is_available() is False


@patch('requests.get')
def test_is_available_non_200_status(mock_get, ollama_provider):
    """Test is_available with non-200 status code."""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    assert ollama_provider.is_available() is False


@patch('requests.post')
def test_generate_empty_response(mock_post, ollama_provider):
    """Test handling of empty response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": ""}
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    result = ollama_provider.generate("Test")
    assert result == ""


@patch('requests.post')
def test_generate_missing_response_key(mock_post, ollama_provider):
    """Test handling of missing 'response' key."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"other_key": "value"}
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    result = ollama_provider.generate("Test")
    assert result == ""

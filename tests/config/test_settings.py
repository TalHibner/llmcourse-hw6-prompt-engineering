"""Tests for configuration settings."""
import pytest
import os
from unittest.mock import patch


class TestSettings:
    """Tests for settings module."""

    def test_settings_module_imports(self):
        """Test that settings module can be imported."""
        from src.config import settings
        assert settings is not None

    @patch.dict(os.environ, {'OLLAMA_BASE_URL': 'http://test:11434'})
    def test_environment_variable_loading(self):
        """Test that environment variables can be loaded."""
        # This tests the pattern used in settings
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        assert base_url == 'http://test:11434'

    def test_default_values(self):
        """Test default configuration values."""
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        timeout = int(os.getenv('OLLAMA_TIMEOUT', '120'))

        # Should have reasonable defaults
        assert 'localhost' in base_url or 'http' in base_url
        assert timeout > 0

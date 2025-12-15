"""Ollama provider implementation - PRIMARY EXECUTION ENGINE"""

import requests
import logging
from typing import Optional
from .base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM execution (PRIMARY PROVIDER)"""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """Initialize Ollama provider

        Args:
            model: Model name (e.g., 'llama3.2', 'mistral', 'phi3')
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"Initialized Ollama provider with model: {model}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama"""
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }

            logger.debug(f"Sending request to Ollama: {self.model}")
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}")

    def get_model_name(self) -> str:
        """Return model identifier"""
        return f"ollama:{self.model}"

    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False

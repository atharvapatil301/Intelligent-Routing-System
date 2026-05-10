"""Ollama client for local model inference."""
import time
import requests
from typing import Optional, Dict, Any

from ..config import config


class OllamaClient:
    """Client for interacting with local Ollama models."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """Initialize Ollama client.

        Args:
            base_url: Ollama API base URL (defaults to config value)
            model: Model name to use (defaults to config value)
        """
        self.base_url = base_url or config.ollama_base_url
        self.model = model or config.ollama_model
        self.api_url = f"{self.base_url}/api/generate"
        self.context = None

    def generate(self, prompt: str, stream: bool = False,
                 temperature: float = 0.7, max_tokens: int = 2048,
                 continue_conversation: bool = False) -> Dict[str, Any]:
        """Generate code using Ollama model.

        Args:
            prompt: The coding prompt
            stream: Whether to stream the response
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            continue_conversation: If True, include previous context

        Returns:
            Dictionary containing:
                - response: Generated text
                - latency_ms: Inference latency in milliseconds
                - model: Model used
                - success: Whether the request succeeded
                - error: Error message if failed
                - context: Conversation context for continuation
        """
        start_time = time.time()

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }

            if continue_conversation and self.context is not None:
                payload["context"] = self.context

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()

            result = response.json()
            latency_ms = int((time.time() - start_time) * 1000)

            context = result.get("context")
            if context is not None:
                self.context = context

            return {
                "response": result.get("response", ""),
                "latency_ms": latency_ms,
                "model": self.model,
                "success": True,
                "error": None,
                "done": result.get("done", True),
                "context": context,
            }

        except requests.exceptions.ConnectionError:
            return {
                "response": "",
                "latency_ms": int((time.time() - start_time) * 1000),
                "model": self.model,
                "success": False,
                "error": f"Failed to connect to Ollama at {self.base_url}. Is Ollama running?",
            }

        except requests.exceptions.Timeout:
            return {
                "response": "",
                "latency_ms": int((time.time() - start_time) * 1000),
                "model": self.model,
                "success": False,
                "error": "Request timed out after 120 seconds",
            }

        except requests.exceptions.RequestException as e:
            return {
                "response": "",
                "latency_ms": int((time.time() - start_time) * 1000),
                "model": self.model,
                "success": False,
                "error": f"Request failed: {str(e)}",
            }

        except Exception as e:
            return {
                "response": "",
                "latency_ms": int((time.time() - start_time) * 1000),
                "model": self.model,
                "success": False,
                "error": f"Unexpected error: {str(e)}",
            }

    def health_check(self) -> bool:
        """Check if Ollama is running and model is available.

        Returns:
            True if Ollama is accessible and model is available
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            model_available = any(
                self.model in name or name.startswith(self.model)
                for name in model_names
            )

            if not model_available:
                print(f"Warning: Model '{self.model}' not found in Ollama.")
                print(f"Available models: {', '.join(model_names)}")
                return False

            return True

        except Exception as e:
            print(f"Ollama health check failed: {e}")
            return False

    def list_models(self) -> list[str]:
        """List available models in Ollama.

        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m.get("name", "") for m in models]
        except Exception:
            return []

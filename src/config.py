"""Configuration management for the Intelligent Routing System."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Central configuration for the routing system."""

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder")

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = "claude-sonnet-4-5-20250929"

    prompt_length_threshold: int = int(os.getenv("PROMPT_LENGTH_THRESHOLD", "100"))

    cloud_keywords: list[str] = None

    log_dir: str = "logs"
    log_level: str = "INFO"

    def __post_init__(self):
        """Initialize default values that can't be set in dataclass."""
        if self.cloud_keywords is None:
            self.cloud_keywords = [
                "optimize", "optimization",
                "design", "architecture",
                "complex", "algorithm",
                "performance", "scalability",
                "concurrent", "parallel",
                "thread-safe", "race condition",
                "dynamic programming", "dp",
                "system design"
            ]

    def validate(self):
        """Validate configuration."""
        if not self.ollama_base_url:
            raise ValueError("OLLAMA_BASE_URL must be set")

        if not self.ollama_model:
            raise ValueError("OLLAMA_MODEL must be set")

        return True


config = Config()

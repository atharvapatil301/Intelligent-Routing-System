"""Claude Code CLI client for cloud model inference."""
import subprocess
import time
import shutil
from typing import Optional, Dict, Any

from ..config import config


class ClaudeClient:
    """Client for interacting with Claude Code CLI."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize Claude Code client.

        Args:
            api_key: Not used (kept for compatibility)
            model: Model name to use (defaults to config value)
        """
        self.model = model or config.claude_model
        self.claude_path = shutil.which("claude")
        self.session_id = None

    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 4096, continue_conversation: bool = False) -> Dict[str, Any]:
        """Generate code using Claude Code CLI.

        Args:
            prompt: The coding prompt
            temperature: Sampling temperature (not used in CLI)
            max_tokens: Maximum tokens to generate (not used in CLI)
            continue_conversation: If True, continue the previous conversation

        Returns:
            Dictionary containing:
                - response: Generated text
                - latency_ms: Inference latency in milliseconds
                - model: Model used
                - success: Whether the request succeeded
                - error: Error message if failed
        """
        start_time = time.time()

        if not self.claude_path:
            return {
                "response": "",
                "latency_ms": 0,
                "model": "claude-code",
                "success": False,
                "error": "Claude Code CLI not found. Make sure 'claude' is in your PATH.",
            }

        try:
            cmd = [
                self.claude_path,
                "--print",
                "--tools", ""
            ]

            if continue_conversation and self.session_id is not None:
                cmd.insert(2, "--continue")

            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=300
            )

            latency_ms = int((time.time() - start_time) * 1000)

            if result.returncode == 0:
                self.session_id = "active"

                return {
                    "response": result.stdout.strip(),
                    "latency_ms": latency_ms,
                    "model": "claude-code",
                    "success": True,
                    "error": None,
                }
            else:
                return {
                    "response": "",
                    "latency_ms": latency_ms,
                    "model": "claude-code",
                    "success": False,
                    "error": f"Claude Code CLI error: {result.stderr.strip()}",
                }

        except subprocess.TimeoutExpired:
            return {
                "response": "",
                "latency_ms": int((time.time() - start_time) * 1000),
                "model": "claude-code",
                "success": False,
                "error": "Claude Code CLI timed out after 5 minutes",
            }

        except Exception as e:
            return {
                "response": "",
                "latency_ms": int((time.time() - start_time) * 1000),
                "model": "claude-code",
                "success": False,
                "error": f"Claude Code CLI error: {str(e)}",
            }

    def health_check(self) -> bool:
        """Check if Claude Code CLI is available.

        Returns:
            True if claude CLI is in PATH
        """
        return self.claude_path is not None

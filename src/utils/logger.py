"""Logging infrastructure for tracking requests and performance."""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from ..config import config


class RequestLogger:
    """Logger for tracking routing decisions and model performance."""

    def __init__(self, log_dir: Optional[str] = None):
        """Initialize logger.

        Args:
            log_dir: Directory to store logs (defaults to config value)
        """
        self.log_dir = Path(log_dir or config.log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.request_log_file = self.log_dir / "requests.jsonl"
        self.stats_log_file = self.log_dir / "stats.json"

    def log_request(
        self,
        prompt: str,
        routing_decision: Dict[str, Any],
        response: str,
        model_used: str,
        latency_ms: int,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a complete request-response cycle.

        Args:
            prompt: Original user prompt
            routing_decision: Dictionary with routing decision details
            response: Model response
            model_used: Which model was used (local/cloud)
            latency_ms: Response latency in milliseconds
            success: Whether the request succeeded
            error: Error message if failed
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "routing_decision": routing_decision,
            "response": response,
            "model_used": model_used,
            "latency_ms": latency_ms,
            "success": success,
            "error": error,
            "metadata": metadata or {}
        }

        with open(self.request_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_recent_logs(self, limit: int = 10) -> list[Dict[str, Any]]:
        """Retrieve recent log entries.

        Args:
            limit: Number of recent entries to return

        Returns:
            List of log entries
        """
        if not self.request_log_file.exists():
            return []

        logs = []
        with open(self.request_log_file, "r") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))

        return logs[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics from logs.

        Returns:
            Dictionary with performance metrics
        """
        if not self.request_log_file.exists():
            return self._empty_stats()

        logs = []
        with open(self.request_log_file, "r") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))

        if not logs:
            return self._empty_stats()

        total_requests = len(logs)
        successful_requests = sum(1 for log in logs if log["success"])
        local_requests = sum(1 for log in logs if log["model_used"] == "local")
        cloud_requests = sum(1 for log in logs if log["model_used"] == "cloud")

        latencies = [log["latency_ms"] for log in logs if log["success"]]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        local_latencies = [
            log["latency_ms"] for log in logs
            if log["success"] and log["model_used"] == "local"
        ]
        cloud_latencies = [
            log["latency_ms"] for log in logs
            if log["success"] and log["model_used"] == "cloud"
        ]

        stats = {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": total_requests - successful_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "local_requests": local_requests,
            "cloud_requests": cloud_requests,
            "local_percentage": (local_requests / total_requests * 100) if total_requests > 0 else 0,
            "cloud_percentage": (cloud_requests / total_requests * 100) if total_requests > 0 else 0,
            "avg_latency_ms": avg_latency,
            "avg_local_latency_ms": sum(local_latencies) / len(local_latencies) if local_latencies else 0,
            "avg_cloud_latency_ms": sum(cloud_latencies) / len(cloud_latencies) if cloud_latencies else 0,
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.stats_log_file, "w") as f:
            json.dump(stats, f, indent=2)

        return stats

    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics."""
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "success_rate": 0.0,
            "local_requests": 0,
            "cloud_requests": 0,
            "local_percentage": 0.0,
            "cloud_percentage": 0.0,
            "avg_latency_ms": 0.0,
            "avg_local_latency_ms": 0.0,
            "avg_cloud_latency_ms": 0.0,
            "last_updated": datetime.now().isoformat(),
        }

    def clear_logs(self):
        """Clear all logs (use with caution!)."""
        if self.request_log_file.exists():
            os.remove(self.request_log_file)
        if self.stats_log_file.exists():
            os.remove(self.stats_log_file)

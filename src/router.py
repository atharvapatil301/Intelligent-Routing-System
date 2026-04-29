"""Intelligent routing logic for model selection."""
from typing import Literal, Dict, Any
from dataclasses import dataclass

from .config import config


@dataclass
class RoutingDecision:
    """Represents a routing decision."""
    target: Literal["local", "cloud"]
    reason: str
    confidence: float
    features: Dict[str, Any]


class Router:
    """Router that decides which model to use for a given prompt."""

    def __init__(self, strategy: str = "rule_based"):
        """Initialize router.

        Args:
            strategy: Routing strategy to use ('rule_based' for Phase 1)
        """
        self.strategy = strategy
        self.prompt_length_threshold = config.prompt_length_threshold
        self.cloud_keywords = config.cloud_keywords

    def route(self, prompt: str) -> RoutingDecision:
        """Decide which model to route the prompt to.

        Args:
            prompt: The user's coding prompt

        Returns:
            RoutingDecision with target model and reasoning
        """
        if self.strategy == "rule_based":
            return self._rule_based_routing(prompt)
        else:
            raise NotImplementedError(f"Strategy '{self.strategy}' not implemented")

    def _rule_based_routing(self, prompt: str) -> RoutingDecision:
        """Rule-based routing logic (Phase 1 baseline).

        Rules:
        1. If prompt_length < threshold → local
        2. If complexity keywords present → cloud
        3. Else → local

        Args:
            prompt: The user's coding prompt

        Returns:
            RoutingDecision
        """
        features = self._extract_features(prompt)

        if features["has_complexity_keywords"]:
            matched_keywords = features["matched_keywords"]
            return RoutingDecision(
                target="cloud",
                reason=f"Complexity keywords detected: {', '.join(matched_keywords[:3])}",
                confidence=0.8,
                features=features
            )

        if features["token_count"] < self.prompt_length_threshold:
            return RoutingDecision(
                target="local",
                reason=f"Short prompt ({features['token_count']} tokens < {self.prompt_length_threshold} threshold)",
                confidence=0.7,
                features=features
            )

        return RoutingDecision(
            target="local",
            reason="Default routing (no complexity signals)",
            confidence=0.6,
            features=features
        )

    def _extract_features(self, prompt: str) -> Dict[str, Any]:
        """Extract features from the prompt for routing decision.

        Args:
            prompt: The user's coding prompt

        Returns:
            Dictionary of extracted features
        """
        prompt_lower = prompt.lower()

        token_count = len(prompt) // 4

        matched_keywords = [
            kw for kw in self.cloud_keywords
            if kw in prompt_lower
        ]
        has_complexity_keywords = len(matched_keywords) > 0

        question_indicators = {
            "implementation": ["implement", "create", "write", "build", "make"],
            "debugging": ["debug", "fix", "error", "bug", "issue", "wrong"],
            "design": ["design", "architecture", "approach", "structure"],
            "optimization": ["optimize", "improve", "faster", "efficient"],
            "explanation": ["explain", "what", "how", "why", "understand"],
        }

        question_types = []
        for q_type, indicators in question_indicators.items():
            if any(ind in prompt_lower for ind in indicators):
                question_types.append(q_type)

        has_code_block = "```" in prompt or "    " in prompt

        return {
            "char_count": len(prompt),
            "token_count": token_count,
            "has_complexity_keywords": has_complexity_keywords,
            "matched_keywords": matched_keywords,
            "question_types": question_types,
            "has_code_block": has_code_block,
            "line_count": prompt.count("\n") + 1,
        }

    def get_statistics(self, decisions: list[RoutingDecision]) -> Dict[str, Any]:
        """Calculate routing statistics from a list of decisions.

        Args:
            decisions: List of routing decisions

        Returns:
            Statistics dictionary
        """
        if not decisions:
            return {
                "total": 0,
                "local_count": 0,
                "cloud_count": 0,
                "local_percentage": 0.0,
                "cloud_percentage": 0.0,
            }

        local_count = sum(1 for d in decisions if d.target == "local")
        cloud_count = sum(1 for d in decisions if d.target == "cloud")
        total = len(decisions)

        return {
            "total": total,
            "local_count": local_count,
            "cloud_count": cloud_count,
            "local_percentage": (local_count / total) * 100,
            "cloud_percentage": (cloud_count / total) * 100,
            "avg_confidence": sum(d.confidence for d in decisions) / total,
        }

"""Intelligent routing logic for model selection."""
from typing import Literal, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .config import config
from .feature_extractor import FeatureExtractor, QueryFeatures
from .vector_db import QueryVectorDB


@dataclass
class RoutingDecision:
    """Represents a routing decision."""
    target: Literal["local", "cloud"]
    reason: str
    confidence: float
    features: Dict[str, Any]


class Router:
    """Router that decides which model to use for a given prompt."""

    def __init__(
        self,
        use_vector_db: bool = True,
        ml_threshold: float = 0.5,
        model_dir: str = "models",
        confidence_threshold: float = 0.7,
        use_confidence_routing: bool = False
    ):
        """Initialize router.

        Args:
            use_vector_db: Whether to use vector DB for similarity search
            ml_threshold: Threshold for ML routing (if P(cloud) > threshold, route to cloud)
            model_dir: Directory containing trained ML model
            confidence_threshold: Minimum confidence for local routing
            use_confidence_routing: Enable confidence-based escalation
        """
        self.prompt_length_threshold = config.prompt_length_threshold
        self.cloud_keywords = config.cloud_keywords
        self.use_vector_db = use_vector_db
        self.ml_threshold = ml_threshold
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold
        self.use_confidence_routing = use_confidence_routing

        self.feature_extractor = FeatureExtractor()

        self.vector_db: Optional[QueryVectorDB] = None
        if use_vector_db:
            self.vector_db = QueryVectorDB()

        self.ml_model: Optional[Any] = None
        self._load_ml_model()

    def route(self, prompt: str) -> RoutingDecision:
        """Decide which model to route the prompt to.

        Uses ML-based routing as first priority, with enhanced rule-based routing as fallback.

        Args:
            prompt: The user's coding prompt

        Returns:
            RoutingDecision with target model and reasoning
        """
        decision = self._ml_routing(prompt)

        if self.use_confidence_routing:
            decision = self._apply_confidence_routing(decision)

        return decision

    def _apply_confidence_routing(self, decision: RoutingDecision) -> RoutingDecision:
        """Apply confidence-based escalation.

        Args:
            decision: Initial routing decision

        Returns:
            Potentially modified routing decision
        """
        if decision.target == "local" and decision.confidence < self.confidence_threshold:
            return RoutingDecision(
                target="cloud",
                reason=f"Escalated to cloud (low confidence: {decision.confidence:.2f} < {self.confidence_threshold})",
                confidence=decision.confidence,
                features=decision.features
            )
        return decision

    def _enhanced_routing(self, prompt: str) -> RoutingDecision:
        """Enhanced routing using advanced features (fallback when ML is unavailable).

        Uses advanced feature extraction and similarity to past queries.

        Args:
            prompt: The user's coding prompt

        Returns:
            RoutingDecision
        """
        features = self.feature_extractor.extract(prompt, generate_embedding=True)

        if self.vector_db is not None and features.embedding is not None:
            features.similarity_to_failures = self.vector_db.calculate_similarity_to_failures(
                features.embedding, k=10
            )

        if features.similarity_to_failures > 0.7:
            return RoutingDecision(
                target="cloud",
                reason=f"High similarity to past local failures ({features.similarity_to_failures:.2f})",
                confidence=0.85,
                features=features.to_dict()
            )

        if features.has_concurrency_mentions or features.has_algorithm_complexity:
            reasons = []
            if features.has_concurrency_mentions:
                reasons.append("concurrency")
            if features.has_algorithm_complexity:
                reasons.append("algorithm complexity")
            return RoutingDecision(
                target="cloud",
                reason=f"Complex features detected: {', '.join(reasons)}",
                confidence=0.8,
                features=features.to_dict()
            )

        if features.is_design or features.is_optimization:
            question_type = "design" if features.is_design else "optimization"
            return RoutingDecision(
                target="cloud",
                reason=f"{question_type.capitalize()} question detected",
                confidence=0.75,
                features=features.to_dict()
            )

        if features.has_complexity_keywords:
            return RoutingDecision(
                target="cloud",
                reason=f"Complexity keywords: {', '.join(features.matched_keywords[:3])}",
                confidence=0.7,
                features=features.to_dict()
            )

        if features.num_functions_requested > 2 or features.num_classes_requested > 1:
            return RoutingDecision(
                target="cloud",
                reason=f"Multiple components requested (funcs={features.num_functions_requested}, classes={features.num_classes_requested})",
                confidence=0.65,
                features=features.to_dict()
            )

        if features.token_count > 200:
            return RoutingDecision(
                target="cloud",
                reason=f"Long prompt ({features.token_count} tokens)",
                confidence=0.6,
                features=features.to_dict()
            )

        return RoutingDecision(
            target="local",
            reason="Simple task - using local model",
            confidence=0.7,
            features=features.to_dict()
        )

    def _load_ml_model(self):
        """Load trained ML model."""
        try:
            from .ml_model import RoutingMLModel

            model_path = Path(self.model_dir)
            if not (model_path / 'routing_model.pt').exists():
                print(f"⚠ ML model not found in {model_path}")
                print("  Please train the model first: python train_model.py")
                print("  Will use enhanced rule-based routing as fallback")
                self.ml_model = None
                return

            self.ml_model = RoutingMLModel()
            self.ml_model.load(str(model_path))
            print(f"✓ ML model loaded from {model_path}")

        except Exception as e:
            print(f"⚠ Failed to load ML model: {e}")
            print("  Will use enhanced rule-based routing as fallback")
            self.ml_model = None

    def _ml_routing(self, prompt: str) -> RoutingDecision:
        """ML-based routing using trained neural network.

        Falls back to enhanced rule-based routing if ML model is unavailable.

        Args:
            prompt: The user's coding prompt

        Returns:
            RoutingDecision
        """
        if self.ml_model is None:
            return self._enhanced_routing(prompt)

        features = self.feature_extractor.extract(prompt, generate_embedding=True)

        if self.vector_db is not None and features.embedding is not None:
            features.similarity_to_failures = self.vector_db.calculate_similarity_to_failures(
                features.embedding, k=10
            )

        feature_vector = self._extract_feature_vector(features)

        try:
            prediction, confidence = self.ml_model.predict(
                feature_vector,
                threshold=self.ml_threshold
            )

            target = "cloud" if prediction == 1 else "local"

            prob_cloud = confidence if prediction == 1 else 1 - confidence
            reason = (
                f"ML model prediction (P(cloud)={prob_cloud:.2f}, "
                f"threshold={self.ml_threshold})"
            )

            return RoutingDecision(
                target=target,
                reason=reason,
                confidence=confidence,
                features=features.to_dict()
            )

        except Exception as e:
            print(f"⚠ ML prediction failed: {e}")
            print("  Falling back to enhanced routing")
            return self._enhanced_routing(prompt)

    def _extract_feature_vector(self, features: QueryFeatures) -> np.ndarray:
        """Extract feature vector for ML model from QueryFeatures.

        Args:
            features: Extracted query features

        Returns:
            Feature vector (403,)
        """
        feature_list = []

        if features.embedding is not None:
            feature_list.extend(features.embedding.tolist())
        else:
            feature_list.extend([0.0] * 384)

        feature_list.extend([
            features.char_count,
            features.token_count,
            features.line_count,
            features.word_count,
            float(features.has_code_block),
            features.num_code_blocks,
            features.num_functions_requested,
            features.num_classes_requested,
            float(features.is_implementation),
            float(features.is_debugging),
            float(features.is_design),
            float(features.is_optimization),
            float(features.is_explanation),
            float(features.has_complexity_keywords),
            len(features.matched_keywords),
            float(features.has_concurrency_mentions),
            float(features.has_algorithm_complexity),
            float(features.has_reasoning_keywords),
            features.similarity_to_failures
        ])

        return np.array(feature_list, dtype=np.float32)

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

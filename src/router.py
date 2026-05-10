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
        strategy: str = "ml",
        use_features: bool = False,
        use_vector_db: bool = False,
        ml_threshold: float = 0.5,
        model_dir: str = "models"
    ):
        """Initialize router.

        Args:
            strategy: Routing strategy to use ('rule_based', 'enhanced', 'ml')
            use_features: Whether to use advanced feature extraction (Phase 2)
            use_vector_db: Whether to use vector DB for similarity search (Phase 2)
            ml_threshold: Threshold for ML routing (if P(cloud) > threshold, route to cloud)
            model_dir: Directory containing trained ML model
        """
        self.strategy = strategy
        self.prompt_length_threshold = config.prompt_length_threshold
        self.cloud_keywords = config.cloud_keywords
        self.use_features = use_features
        self.use_vector_db = use_vector_db
        self.ml_threshold = ml_threshold
        self.model_dir = model_dir

        # Initialize advanced components if needed
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.vector_db: Optional[QueryVectorDB] = None
        self.ml_model: Optional[Any] = None  # Will be loaded lazily

        if use_features:
            self.feature_extractor = FeatureExtractor()

        if use_vector_db:
            self.vector_db = QueryVectorDB()

        # Load ML model if using ML strategy
        if strategy == "ml":
            self._load_ml_model()

    def route(self, prompt: str) -> RoutingDecision:
        """Decide which model to route the prompt to.

        Args:
            prompt: The user's coding prompt

        Returns:
            RoutingDecision with target model and reasoning
        """
        if self.strategy == "rule_based":
            return self._rule_based_routing(prompt)
        elif self.strategy == "enhanced":
            return self._enhanced_routing(prompt)
        elif self.strategy == "ml":
            return self._ml_routing(prompt)
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

    def _enhanced_routing(self, prompt: str) -> RoutingDecision:
        """Enhanced routing using Phase 2 features.

        Uses advanced feature extraction and similarity to past queries.

        Args:
            prompt: The user's coding prompt

        Returns:
            RoutingDecision
        """
        # Extract advanced features
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()

        features = self.feature_extractor.extract(prompt, generate_embedding=True)

        # Calculate similarity to failures if vector DB available
        if self.vector_db is not None and features.embedding is not None:
            features.similarity_to_failures = self.vector_db.calculate_similarity_to_failures(
                features.embedding, k=10
            )

        # Enhanced decision logic
        # Rule 1: High similarity to past failures → cloud
        if features.similarity_to_failures > 0.7:
            return RoutingDecision(
                target="cloud",
                reason=f"High similarity to past local failures ({features.similarity_to_failures:.2f})",
                confidence=0.85,
                features=features.to_dict()
            )

        # Rule 2: Concurrency or algorithm complexity → cloud
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

        # Rule 3: Design or optimization questions → cloud
        if features.is_design or features.is_optimization:
            question_type = "design" if features.is_design else "optimization"
            return RoutingDecision(
                target="cloud",
                reason=f"{question_type.capitalize()} question detected",
                confidence=0.75,
                features=features.to_dict()
            )

        # Rule 4: Complexity keywords → cloud
        if features.has_complexity_keywords:
            return RoutingDecision(
                target="cloud",
                reason=f"Complexity keywords: {', '.join(features.matched_keywords[:3])}",
                confidence=0.7,
                features=features.to_dict()
            )

        # Rule 5: Multiple functions/classes requested → cloud
        if features.num_functions_requested > 2 or features.num_classes_requested > 1:
            return RoutingDecision(
                target="cloud",
                reason=f"Multiple components requested (funcs={features.num_functions_requested}, classes={features.num_classes_requested})",
                confidence=0.65,
                features=features.to_dict()
            )

        # Rule 6: Very long prompts → cloud
        if features.token_count > 200:
            return RoutingDecision(
                target="cloud",
                reason=f"Long prompt ({features.token_count} tokens)",
                confidence=0.6,
                features=features.to_dict()
            )

        # Default: Local for simple tasks
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
                print("  Falling back to enhanced routing")
                self.strategy = "enhanced"
                return

            self.ml_model = RoutingMLModel()
            self.ml_model.load(str(model_path))
            print(f"✓ ML model loaded from {model_path}")

            # Ensure feature extractor is initialized for ML routing
            if self.feature_extractor is None:
                self.feature_extractor = FeatureExtractor()

        except Exception as e:
            print(f"⚠ Failed to load ML model: {e}")
            print("  Falling back to enhanced routing")
            self.strategy = "enhanced"
            self.ml_model = None

    def _ml_routing(self, prompt: str) -> RoutingDecision:
        """ML-based routing using trained neural network (Phase 4).

        Args:
            prompt: The user's coding prompt

        Returns:
            RoutingDecision
        """
        # If model not loaded, fall back to enhanced routing
        if self.ml_model is None:
            return self._enhanced_routing(prompt)

        # Extract features
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()

        features = self.feature_extractor.extract(prompt, generate_embedding=True)

        # Calculate similarity to failures if vector DB available
        if self.vector_db is not None and features.embedding is not None:
            features.similarity_to_failures = self.vector_db.calculate_similarity_to_failures(
                features.embedding, k=10
            )

        # Extract feature vector for ML model
        feature_vector = self._extract_feature_vector(features)

        # Predict using ML model
        try:
            prediction, confidence = self.ml_model.predict(
                feature_vector,
                threshold=self.ml_threshold
            )

            # prediction: 0 = local, 1 = cloud
            target = "cloud" if prediction == 1 else "local"

            # Generate reason
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

        # Embedding (384 dimensions)
        if features.embedding is not None:
            feature_list.extend(features.embedding.tolist())
        else:
            feature_list.extend([0.0] * 384)

        # Structural features (19 features)
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

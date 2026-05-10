"""Advanced feature extraction for query analysis (Phase 2)."""
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class QueryFeatures:
    """Comprehensive features extracted from a coding query."""

    # Basic text features
    char_count: int
    token_count: int
    line_count: int
    word_count: int

    # Code structure features
    has_code_block: bool
    num_code_blocks: int
    num_functions_requested: int
    num_classes_requested: int

    # Question type features
    question_types: List[str]
    is_implementation: bool
    is_debugging: bool
    is_design: bool
    is_optimization: bool
    is_explanation: bool

    # Complexity signals
    has_complexity_keywords: bool
    matched_keywords: List[str]
    has_concurrency_mentions: bool
    has_algorithm_complexity: bool
    has_reasoning_keywords: bool

    # Advanced features
    embedding: Optional[np.ndarray] = None
    similarity_to_failures: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling numpy arrays."""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data

    def to_feature_vector(self) -> np.ndarray:
        """Convert to numerical feature vector for ML models."""
        features = [
            self.char_count,
            self.token_count,
            self.line_count,
            self.word_count,
            float(self.has_code_block),
            self.num_code_blocks,
            self.num_functions_requested,
            self.num_classes_requested,
            float(self.is_implementation),
            float(self.is_debugging),
            float(self.is_design),
            float(self.is_optimization),
            float(self.is_explanation),
            float(self.has_complexity_keywords),
            len(self.matched_keywords),
            float(self.has_concurrency_mentions),
            float(self.has_algorithm_complexity),
            float(self.has_reasoning_keywords),
            self.similarity_to_failures,
        ]

        # Include embedding if available
        if self.embedding is not None:
            features.extend(self.embedding.tolist())

        return np.array(features)


class FeatureExtractor:
    """Extract advanced features from coding queries."""

    # Complexity keywords (from config.py)
    CLOUD_KEYWORDS = [
        "optimize", "optimization",
        "design", "architecture",
        "complex", "algorithm",
        "performance", "scalability",
        "concurrent", "parallel",
        "thread-safe", "race condition",
        "dynamic programming", "dp",
        "system design"
    ]

    # Reasoning keywords
    REASONING_KEYWORDS = [
        "why", "derive", "prove", "justify", "explain",
        "analyze", "compare", "evaluate", "reason"
    ]

    # Concurrency keywords
    CONCURRENCY_KEYWORDS = [
        "thread", "concurrent", "parallel", "async", "await",
        "lock", "mutex", "semaphore", "race condition",
        "deadlock", "synchronize", "atomic"
    ]

    # Algorithm complexity indicators
    COMPLEXITY_PATTERNS = [
        r"O\([^)]+\)",  # Big-O notation
        r"time complexity",
        r"space complexity",
        r"dynamic programming",
        r"greedy",
        r"divide and conquer",
        r"backtracking",
        r"memoization",
    ]

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize feature extractor.

        Args:
            embedding_model: Name of sentence-transformers model to use
        """
        self.embedding_model_name = embedding_model
        self._embedding_model: Optional[SentenceTransformer] = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}...")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def extract(self, prompt: str, generate_embedding: bool = True) -> QueryFeatures:
        """Extract comprehensive features from a prompt.

        Args:
            prompt: The user's coding prompt
            generate_embedding: Whether to generate semantic embedding

        Returns:
            QueryFeatures object with all extracted features
        """
        prompt_lower = prompt.lower()

        # Basic text features
        char_count = len(prompt)
        token_count = len(prompt) // 4  # Rough approximation
        line_count = prompt.count("\n") + 1
        word_count = len(prompt.split())

        # Code structure features
        has_code_block = "```" in prompt or "    " in prompt
        num_code_blocks = prompt.count("```") // 2
        num_functions_requested = self._count_function_requests(prompt_lower)
        num_classes_requested = self._count_class_requests(prompt_lower)

        # Question type detection
        question_types = self._detect_question_types(prompt_lower)
        is_implementation = "implementation" in question_types
        is_debugging = "debugging" in question_types
        is_design = "design" in question_types
        is_optimization = "optimization" in question_types
        is_explanation = "explanation" in question_types

        # Complexity signals
        matched_keywords = [kw for kw in self.CLOUD_KEYWORDS if kw in prompt_lower]
        has_complexity_keywords = len(matched_keywords) > 0
        has_concurrency_mentions = any(kw in prompt_lower for kw in self.CONCURRENCY_KEYWORDS)
        has_algorithm_complexity = any(
            re.search(pattern, prompt_lower) for pattern in self.COMPLEXITY_PATTERNS
        )
        has_reasoning_keywords = any(kw in prompt_lower for kw in self.REASONING_KEYWORDS)

        # Generate embedding
        embedding = None
        if generate_embedding:
            embedding = self.embedding_model.encode(prompt, convert_to_numpy=True)

        return QueryFeatures(
            char_count=char_count,
            token_count=token_count,
            line_count=line_count,
            word_count=word_count,
            has_code_block=has_code_block,
            num_code_blocks=num_code_blocks,
            num_functions_requested=num_functions_requested,
            num_classes_requested=num_classes_requested,
            question_types=question_types,
            is_implementation=is_implementation,
            is_debugging=is_debugging,
            is_design=is_design,
            is_optimization=is_optimization,
            is_explanation=is_explanation,
            has_complexity_keywords=has_complexity_keywords,
            matched_keywords=matched_keywords,
            has_concurrency_mentions=has_concurrency_mentions,
            has_algorithm_complexity=has_algorithm_complexity,
            has_reasoning_keywords=has_reasoning_keywords,
            embedding=embedding,
            similarity_to_failures=0.0,  # Will be computed by vector DB
        )

    def _count_function_requests(self, prompt_lower: str) -> int:
        """Count how many functions are requested."""
        patterns = [
            r"write (\d+) functions?",
            r"create (\d+) functions?",
            r"implement (\d+) functions?",
        ]

        count = 0
        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                count = max(count, int(match.group(1)))

        # Check for single function requests
        if any(word in prompt_lower for word in ["function", "method", "procedure"]):
            count = max(count, 1)

        return count

    def _count_class_requests(self, prompt_lower: str) -> int:
        """Count how many classes are requested."""
        patterns = [
            r"write (\d+) classes",
            r"create (\d+) classes",
            r"implement (\d+) classes",
        ]

        count = 0
        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                count = max(count, int(match.group(1)))

        # Check for single class requests
        if "class" in prompt_lower:
            count = max(count, 1)

        return count

    def _detect_question_types(self, prompt_lower: str) -> List[str]:
        """Detect what type of question is being asked."""
        question_indicators = {
            "implementation": ["implement", "create", "write", "build", "make", "code"],
            "debugging": ["debug", "fix", "error", "bug", "issue", "wrong", "broken"],
            "design": ["design", "architecture", "approach", "structure", "pattern"],
            "optimization": ["optimize", "improve", "faster", "efficient", "better"],
            "explanation": ["explain", "what", "how", "why", "understand", "describe"],
        }

        question_types = []
        for q_type, indicators in question_indicators.items():
            if any(ind in prompt_lower for ind in indicators):
                question_types.append(q_type)

        return question_types

    def batch_extract(self, prompts: List[str], generate_embeddings: bool = True) -> List[QueryFeatures]:
        """Extract features from multiple prompts efficiently.

        Args:
            prompts: List of prompts to process
            generate_embeddings: Whether to generate embeddings

        Returns:
            List of QueryFeatures
        """
        features_list = []

        # Extract non-embedding features first
        for prompt in prompts:
            features = self.extract(prompt, generate_embedding=False)
            features_list.append(features)

        # Generate embeddings in batch for efficiency
        if generate_embeddings:
            embeddings = self.embedding_model.encode(prompts, convert_to_numpy=True)
            for i, features in enumerate(features_list):
                features.embedding = embeddings[i]

        return features_list

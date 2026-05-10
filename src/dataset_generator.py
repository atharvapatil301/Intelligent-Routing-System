"""Dataset generation for training routing model (Phase 3)."""
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from .clients.ollama_client import OllamaClient
from .clients.claude_client import ClaudeClient
from .feature_extractor import FeatureExtractor, QueryFeatures
from .config import config


@dataclass
class DatasetSample:
    """A single sample in the training dataset."""

    sample_id: str
    timestamp: str
    prompt: str

    features: Dict[str, Any]
    embedding: List[float]

    qwen_output: str
    claude_output: str

    qwen_success: bool
    claude_success: bool
    qwen_score: float
    claude_score: float

    label: int

    qwen_latency_ms: float
    claude_latency_ms: float
    evaluation_method: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DatasetGenerator:
    """Generate training dataset by running both models and evaluating outputs."""

    def __init__(
        self,
        output_dir: str = "data/datasets",
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        """Initialize dataset generator.

        Args:
            output_dir: Directory to save datasets
            feature_extractor: Feature extractor instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.ollama_client = OllamaClient()
        self.claude_client = ClaudeClient()

        self.samples: List[DatasetSample] = []

    def generate_sample(
        self,
        prompt: str,
        evaluation_method: str = "heuristic",
        timeout: int = 120
    ) -> DatasetSample:
        """Generate a single dataset sample.

        Args:
            prompt: Coding prompt to test
            evaluation_method: Method to evaluate outputs ('heuristic', 'unit_test', 'llm_judge')
            timeout: Timeout in seconds for each model

        Returns:
            DatasetSample with results from both models
        """
        print(f"\nGenerating sample for: {prompt[:60]}...")

        features = self.feature_extractor.extract(prompt, generate_embedding=True)

        print("  Running Qwen (local)...")
        qwen_output, qwen_latency = self._run_model_with_timing(
            self.ollama_client, prompt, timeout
        )

        print("  Running Claude (cloud)...")
        claude_output, claude_latency = self._run_model_with_timing(
            self.claude_client, prompt, timeout
        )

        print(f"  Evaluating with {evaluation_method}...")
        qwen_success, qwen_score = self._evaluate_output(
            prompt, qwen_output, evaluation_method
        )
        claude_success, claude_score = self._evaluate_output(
            prompt, claude_output, evaluation_method
        )

        if qwen_success and qwen_score >= 0.7:
            label = 0
        elif claude_success and claude_score > qwen_score:
            label = 1
        else:
            label = 0

        sample = DatasetSample(
            sample_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            features=features.to_dict(),
            embedding=features.embedding.tolist() if features.embedding is not None else [],
            qwen_output=qwen_output,
            claude_output=claude_output,
            qwen_success=qwen_success,
            claude_success=claude_success,
            qwen_score=qwen_score,
            claude_score=claude_score,
            label=label,
            qwen_latency_ms=qwen_latency,
            claude_latency_ms=claude_latency,
            evaluation_method=evaluation_method,
            notes=""
        )

        self.samples.append(sample)
        print(f"  Result: Qwen={qwen_score:.2f} Claude={claude_score:.2f} Label={label}")

        return sample

    def _run_model_with_timing(
        self,
        client,
        prompt: str,
        timeout: int
    ) -> Tuple[str, float]:
        """Run a model and measure latency.

        Args:
            client: Model client (OllamaClient or ClaudeClient)
            prompt: Input prompt
            timeout: Timeout in seconds

        Returns:
            Tuple of (output, latency_ms)
        """
        start_time = time.time()

        try:
            response = client.generate(
                prompt,
                continue_conversation=False
            )
            output = response.get('response', '')
        except Exception as e:
            output = f"ERROR: {str(e)}"

        latency_ms = (time.time() - start_time) * 1000

        return output, latency_ms

    def _evaluate_output(
        self,
        prompt: str,
        output: str,
        method: str = "heuristic"
    ) -> Tuple[bool, float]:
        """Evaluate model output quality.

        Args:
            prompt: Original prompt
            output: Model output to evaluate
            method: Evaluation method ('heuristic', 'unit_test', 'llm_judge')

        Returns:
            Tuple of (success: bool, score: float [0-1])
        """
        if method == "heuristic":
            return self._heuristic_evaluation(output)
        elif method == "unit_test":
            return self._unit_test_evaluation(prompt, output)
        elif method == "llm_judge":
            return self._llm_judge_evaluation(prompt, output)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

    def _heuristic_evaluation(self, output: str) -> Tuple[bool, float]:
        """Evaluate output using heuristics.

        Args:
            output: Model output

        Returns:
            Tuple of (success, score)
        """
        if not output or output.startswith("ERROR:"):
            return False, 0.0

        score = 0.0
        reasons = []

        if "def " in output or "class " in output or "function" in output:
            score += 0.3
            reasons.append("code_present")

        if "```" in output:
            score += 0.2
            reasons.append("code_block")

        if len(output) > 50:
            score += 0.2
            reasons.append("adequate_length")

        error_indicators = ["error", "cannot", "unable", "sorry", "don't know"]
        has_errors = any(ind in output.lower() for ind in error_indicators)
        if not has_errors:
            score += 0.3
            reasons.append("no_errors")

        success = score >= 0.5

        return success, min(score, 1.0)

    def _unit_test_evaluation(self, prompt: str, output: str) -> Tuple[bool, float]:
        """Evaluate output using unit tests (if applicable).

        Args:
            prompt: Original prompt
            output: Model output

        Returns:
            Tuple of (success, score)
        """
        return self._heuristic_evaluation(output)

    def _llm_judge_evaluation(self, prompt: str, output: str) -> Tuple[bool, float]:
        """Evaluate output using LLM-as-judge.

        Args:
            prompt: Original prompt
            output: Model output

        Returns:
            Tuple of (success, score)
        """
        return self._heuristic_evaluation(output)

    def generate_from_prompts(
        self,
        prompts: List[str],
        evaluation_method: str = "heuristic",
        save_interval: int = 10
    ) -> List[DatasetSample]:
        """Generate samples from a list of prompts.

        Args:
            prompts: List of coding prompts
            evaluation_method: Evaluation method to use
            save_interval: Save progress every N samples

        Returns:
            List of generated samples
        """
        samples = []

        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Processing prompt...")

            try:
                sample = self.generate_sample(prompt, evaluation_method)
                samples.append(sample)

                if i % save_interval == 0:
                    self.save_dataset(f"dataset_progress_{i}.jsonl")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                continue

        return samples

    def save_dataset(self, filename: str) -> str:
        """Save dataset to JSONL file.

        Args:
            filename: Output filename

        Returns:
            Full path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict()) + '\n')

        print(f"\nSaved {len(self.samples)} samples to {output_path}")
        return str(output_path)

    def load_dataset(self, filepath: str) -> List[DatasetSample]:
        """Load dataset from JSONL file.

        Args:
            filepath: Path to dataset file

        Returns:
            List of DatasetSample objects
        """
        samples = []

        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                sample = DatasetSample(**data)
                samples.append(sample)

        self.samples = samples
        print(f"Loaded {len(samples)} samples from {filepath}")
        return samples

    def create_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True
    ) -> Tuple[List[DatasetSample], List[DatasetSample], List[DatasetSample]]:
        """Split dataset into train/val/test sets.

        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            shuffle: Whether to shuffle before splitting

        Returns:
            Tuple of (train, val, test) datasets
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "Ratios must sum to 1.0"

        samples = self.samples.copy()

        if shuffle:
            np.random.shuffle(samples)

        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = samples[:train_end]
        val = samples[train_end:val_end]
        test = samples[val_end:]

        print(f"\nDataset split:")
        print(f"  Train: {len(train)} samples ({len(train)/n*100:.1f}%)")
        print(f"  Val:   {len(val)} samples ({len(val)/n*100:.1f}%)")
        print(f"  Test:  {len(test)} samples ({len(test)/n*100:.1f}%)")

        return train, val, test

    def save_splits(
        self,
        train: List[DatasetSample],
        val: List[DatasetSample],
        test: List[DatasetSample],
        prefix: str = "dataset"
    ) -> Dict[str, str]:
        """Save train/val/test splits to separate files.

        Args:
            train: Training samples
            val: Validation samples
            test: Test samples
            prefix: Filename prefix

        Returns:
            Dictionary of split names to file paths
        """
        paths = {}

        for split_name, samples in [("train", train), ("val", val), ("test", test)]:
            filename = f"{prefix}_{split_name}.jsonl"
            filepath = self.output_dir / filename

            with open(filepath, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample.to_dict()) + '\n')

            paths[split_name] = str(filepath)
            print(f"Saved {len(samples)} {split_name} samples to {filepath}")

        return paths

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Statistics dictionary
        """
        if not self.samples:
            return {"total": 0}

        total = len(self.samples)
        local_sufficient = sum(1 for s in self.samples if s.label == 0)
        cloud_needed = sum(1 for s in self.samples if s.label == 1)

        qwen_successes = sum(1 for s in self.samples if s.qwen_success)
        claude_successes = sum(1 for s in self.samples if s.claude_success)

        avg_qwen_score = sum(s.qwen_score for s in self.samples) / total
        avg_claude_score = sum(s.claude_score for s in self.samples) / total

        avg_qwen_latency = sum(s.qwen_latency_ms for s in self.samples) / total
        avg_claude_latency = sum(s.claude_latency_ms for s in self.samples) / total

        return {
            "total": total,
            "local_sufficient": local_sufficient,
            "cloud_needed": cloud_needed,
            "local_percentage": (local_sufficient / total) * 100,
            "cloud_percentage": (cloud_needed / total) * 100,
            "qwen_success_rate": (qwen_successes / total) * 100,
            "claude_success_rate": (claude_successes / total) * 100,
            "avg_qwen_score": avg_qwen_score,
            "avg_claude_score": avg_claude_score,
            "avg_qwen_latency_ms": avg_qwen_latency,
            "avg_claude_latency_ms": avg_claude_latency,
        }

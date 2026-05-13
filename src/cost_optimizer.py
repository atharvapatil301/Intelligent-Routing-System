"""Cost optimization and threshold tuning for routing decisions."""
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class CostConfig:
    """Configuration for cost optimization.

    Default values based on measured production latencies:
    - Local: ~42s (Ollama qwen2.5-coder with full generation)
    - Cloud: ~27s (Claude API with network + generation)

    Cost estimates:
    - Local: $0 (free, self-hosted)
    - Cloud: ~$0.01 per call (typical Claude API pricing)

    Penalty weights:
    - latency_penalty_weight: How much to penalize latency (0.00001 = minimal)
    - cost_penalty_weight: How much to penalize cost (1.0 = strong)
    """
    local_latency_ms: float = 42000.0
    cloud_latency_ms: float = 27000.0
    local_cost_per_call: float = 0.0
    cloud_cost_per_call: float = 0.01
    latency_penalty_weight: float = 0.00001
    cost_penalty_weight: float = 1.0


class CostOptimizer:
    """Optimizer for multi-objective cost function."""

    def __init__(self, config: CostConfig = None):
        """Initialize cost optimizer.

        Args:
            config: Cost configuration parameters
        """
        self.config = config or CostConfig()

    def calculate_score(
        self,
        accuracy: float,
        latency_ms: float,
        cost: float
    ) -> float:
        """Calculate multi-objective score.

        Args:
            accuracy: Accuracy score (0-1)
            latency_ms: Latency in milliseconds
            cost: Cost in dollars

        Returns:
            Overall score (higher is better)
        """
        score = accuracy - (
            self.config.latency_penalty_weight * latency_ms +
            self.config.cost_penalty_weight * cost
        )
        return score

    def evaluate_routing_strategy(
        self,
        decisions: List[Dict],
        ground_truth: List[bool] = None
    ) -> Dict[str, float]:
        """Evaluate a routing strategy.

        Args:
            decisions: List of routing decisions with results
            ground_truth: Optional ground truth for accuracy calculation

        Returns:
            Dictionary with evaluation metrics
        """
        total_cost = 0.0
        total_latency = 0.0
        correct = 0
        local_calls = 0
        cloud_calls = 0

        for i, decision in enumerate(decisions):
            if decision['model_used'] == 'local':
                local_calls += 1
                total_cost += self.config.local_cost_per_call
                total_latency += decision.get('latency_ms', self.config.local_latency_ms)
            else:
                cloud_calls += 1
                total_cost += self.config.cloud_cost_per_call
                total_latency += decision.get('latency_ms', self.config.cloud_latency_ms)

            if ground_truth and i < len(ground_truth):
                if decision.get('success', False) == ground_truth[i]:
                    correct += 1

        n = len(decisions)
        accuracy = correct / n if ground_truth and n > 0 else 0.0
        avg_latency = total_latency / n if n > 0 else 0.0

        score = self.calculate_score(accuracy, avg_latency, total_cost)

        return {
            'accuracy': accuracy,
            'total_cost': total_cost,
            'avg_latency_ms': avg_latency,
            'total_latency_ms': total_latency,
            'score': score,
            'local_calls': local_calls,
            'cloud_calls': cloud_calls,
            'local_percentage': (local_calls / n * 100) if n > 0 else 0.0,
            'cloud_percentage': (cloud_calls / n * 100) if n > 0 else 0.0,
        }

    def optimize_threshold(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        thresholds: np.ndarray = None
    ) -> Tuple[float, Dict[str, float]]:
        """Find optimal routing threshold.

        Args:
            predictions: Model predictions (probabilities for cloud)
            ground_truth: Ground truth labels (0=local sufficient, 1=cloud needed)
            thresholds: Array of thresholds to test

        Returns:
            Tuple of (best_threshold, best_metrics)
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)

        best_score = float('-inf')
        best_threshold = 0.5
        best_metrics = {}

        for threshold in thresholds:
            decisions = []
            for i, pred in enumerate(predictions):
                model_used = 'cloud' if pred > threshold else 'local'
                success = (model_used == 'cloud') == ground_truth[i]
                decisions.append({
                    'model_used': model_used,
                    'success': success
                })

            metrics = self.evaluate_routing_strategy(decisions, ground_truth)

            if metrics['score'] > best_score:
                best_score = metrics['score']
                best_threshold = threshold
                best_metrics = metrics

        return best_threshold, best_metrics

    def compare_strategies(
        self,
        test_samples: int,
        always_local_accuracy: float,
        always_cloud_accuracy: float,
        smart_router_decisions: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """Compare different routing strategies.

        Args:
            test_samples: Number of test samples
            always_local_accuracy: Accuracy if always using local
            always_cloud_accuracy: Accuracy if always using cloud
            smart_router_decisions: Decisions from smart router

        Returns:
            Comparison dictionary
        """
        always_local = {
            'model_used': 'local',
            'success': True
        }
        always_cloud = {
            'model_used': 'cloud',
            'success': True
        }

        local_decisions = [always_local] * test_samples
        cloud_decisions = [always_cloud] * test_samples

        local_metrics = self.evaluate_routing_strategy(local_decisions)
        local_metrics['accuracy'] = always_local_accuracy

        cloud_metrics = self.evaluate_routing_strategy(cloud_decisions)
        cloud_metrics['accuracy'] = always_cloud_accuracy

        smart_metrics = self.evaluate_routing_strategy(smart_router_decisions)

        return {
            'always_local': local_metrics,
            'always_cloud': cloud_metrics,
            'smart_router': smart_metrics
        }

    def calculate_savings(
        self,
        baseline_cost: float,
        optimized_cost: float
    ) -> Dict[str, float]:
        """Calculate cost savings.

        Args:
            baseline_cost: Baseline cost (usually always-cloud)
            optimized_cost: Optimized cost with smart routing

        Returns:
            Savings metrics
        """
        absolute_savings = baseline_cost - optimized_cost
        percentage_savings = (absolute_savings / baseline_cost * 100) if baseline_cost > 0 else 0.0

        return {
            'baseline_cost': baseline_cost,
            'optimized_cost': optimized_cost,
            'absolute_savings': absolute_savings,
            'percentage_savings': percentage_savings
        }

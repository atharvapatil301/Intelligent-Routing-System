"""Evaluation framework for routing system performance."""
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import numpy as np
from datetime import datetime

from .cost_optimizer import CostOptimizer, CostConfig
from .router import Router
from .clients.ollama_client import OllamaClient
from .clients.claude_client import ClaudeClient


class RoutingEvaluator:
    """Evaluator for comparing routing strategies."""

    def __init__(self, cost_config: CostConfig = None):
        """Initialize evaluator.

        Args:
            cost_config: Cost configuration for optimization
        """
        self.cost_optimizer = CostOptimizer(cost_config or CostConfig())
        self.ollama_client = OllamaClient()
        self.claude_client = ClaudeClient()

    def evaluate_on_dataset(
        self,
        test_file: str,
        router: Router = None
    ) -> Dict:
        """Evaluate routing system on a test dataset.

        Args:
            test_file: Path to test dataset (JSONL format)
            router: Router instance to evaluate

        Returns:
            Evaluation results dictionary
        """
        if router is None:
            router = Router()

        test_data = self._load_dataset(test_file)

        results = {
            'smart_router': [],
            'always_local': [],
            'always_cloud': [],
            'ground_truth': []
        }

        for sample in test_data:
            prompt = sample['prompt']
            ground_truth_label = sample.get('label', None)

            decision = router.route(prompt)

            smart_result = {
                'prompt': prompt,
                'model_used': decision.target,
                'reason': decision.reason,
                'confidence': decision.confidence,
                'latency_ms': 0,
                'success': None
            }

            results['smart_router'].append(smart_result)
            results['always_local'].append({'model_used': 'local'})
            results['always_cloud'].append({'model_used': 'cloud'})

            if ground_truth_label is not None:
                results['ground_truth'].append(ground_truth_label == 1)

        metrics = self._calculate_metrics(results)

        return {
            'metrics': metrics,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

    def run_live_evaluation(
        self,
        prompts: List[str],
        router: Router = None,
        run_both_models: bool = False
    ) -> Dict:
        """Run live evaluation by actually calling models.

        Args:
            prompts: List of prompts to evaluate
            router: Router instance to use
            run_both_models: If True, run both models for ground truth

        Returns:
            Evaluation results
        """
        if router is None:
            router = Router()

        results = []

        for prompt in prompts:
            decision = router.route(prompt)

            result = {
                'prompt': prompt,
                'routing_decision': decision.target,
                'routing_confidence': decision.confidence,
                'routing_reason': decision.reason
            }

            if run_both_models:
                local_result = self.ollama_client.generate(prompt)
                cloud_result = self.claude_client.generate(prompt)

                result['local_response'] = local_result.get('response', '')
                result['local_latency_ms'] = local_result.get('latency_ms', 0)
                result['local_success'] = local_result.get('success', False)

                result['cloud_response'] = cloud_result.get('response', '')
                result['cloud_latency_ms'] = cloud_result.get('latency_ms', 0)
                result['cloud_success'] = cloud_result.get('success', False)

                if decision.target == 'local':
                    result['used_response'] = result['local_response']
                    result['latency_ms'] = result['local_latency_ms']
                    result['success'] = result['local_success']
                else:
                    result['used_response'] = result['cloud_response']
                    result['latency_ms'] = result['cloud_latency_ms']
                    result['success'] = result['cloud_success']
            else:
                if decision.target == 'local':
                    model_result = self.ollama_client.generate(prompt)
                else:
                    model_result = self.claude_client.generate(prompt)

                result['used_response'] = model_result.get('response', '')
                result['latency_ms'] = model_result.get('latency_ms', 0)
                result['success'] = model_result.get('success', False)

            results.append(result)

        metrics = self._calculate_live_metrics(results, run_both_models)

        return {
            'metrics': metrics,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

    def compare_strategies(
        self,
        test_file: str,
        router: Router = None
    ) -> Dict:
        """Compare all routing strategies.

        Args:
            test_file: Path to test dataset
            router: Router to evaluate

        Returns:
            Comparison results
        """
        test_data = self._load_dataset(test_file)
        n_samples = len(test_data)

        router = router or Router()

        smart_decisions = []
        for sample in test_data:
            decision = router.route(sample['prompt'])
            smart_decisions.append({
                'model_used': decision.target,
                'confidence': decision.confidence,
                'success': True
            })

        always_local_accuracy = self._estimate_accuracy(test_data, 'local')
        always_cloud_accuracy = self._estimate_accuracy(test_data, 'cloud')

        comparison = self.cost_optimizer.compare_strategies(
            test_samples=n_samples,
            always_local_accuracy=always_local_accuracy,
            always_cloud_accuracy=always_cloud_accuracy,
            smart_router_decisions=smart_decisions
        )

        baseline_cost = comparison['always_cloud']['total_cost']
        optimized_cost = comparison['smart_router']['total_cost']
        savings = self.cost_optimizer.calculate_savings(baseline_cost, optimized_cost)

        comparison['savings'] = savings

        return comparison

    def generate_report(
        self,
        evaluation_results: Dict,
        output_file: str = None
    ) -> str:
        """Generate evaluation report.

        Args:
            evaluation_results: Results from evaluation
            output_file: Optional file to save report

        Returns:
            Report as string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ROUTING SYSTEM EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append(f"\nGenerated: {evaluation_results.get('timestamp', 'N/A')}\n")

        if 'metrics' in evaluation_results:
            metrics = evaluation_results['metrics']
            lines.append("\nPERFORMANCE METRICS")
            lines.append("-" * 80)
            for key, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"{key:30s}: {value:.2f}")
                else:
                    lines.append(f"{key:30s}: {value}")

        if 'savings' in evaluation_results:
            savings = evaluation_results['savings']
            lines.append("\nCOST SAVINGS")
            lines.append("-" * 80)
            lines.append(f"Baseline Cost:                ${savings['baseline_cost']:.2f}")
            lines.append(f"Optimized Cost:               ${savings['optimized_cost']:.2f}")
            lines.append(f"Absolute Savings:             ${savings['absolute_savings']:.2f}")
            lines.append(f"Percentage Savings:           {savings['percentage_savings']:.1f}%")

        lines.append("\n" + "=" * 80)

        report = "\n".join(lines)

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report)

        return report

    def _load_dataset(self, file_path: str) -> List[Dict]:
        """Load dataset from JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of samples
        """
        samples = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def _calculate_metrics(self, results: Dict) -> Dict:
        """Calculate evaluation metrics.

        Args:
            results: Results dictionary

        Returns:
            Metrics dictionary
        """
        smart_router = results['smart_router']

        local_count = sum(1 for r in smart_router if r['model_used'] == 'local')
        cloud_count = sum(1 for r in smart_router if r['model_used'] == 'cloud')
        total = len(smart_router)

        metrics = {
            'total_queries': total,
            'local_calls': local_count,
            'cloud_calls': cloud_count,
            'local_percentage': (local_count / total * 100) if total > 0 else 0,
            'cloud_percentage': (cloud_count / total * 100) if total > 0 else 0,
        }

        avg_confidence = sum(r['confidence'] for r in smart_router) / total if total > 0 else 0
        metrics['avg_confidence'] = avg_confidence

        return metrics

    def _calculate_live_metrics(
        self,
        results: List[Dict],
        has_both_models: bool
    ) -> Dict:
        """Calculate metrics from live evaluation.

        Args:
            results: List of evaluation results
            has_both_models: Whether both models were run

        Returns:
            Metrics dictionary
        """
        metrics = {}

        total = len(results)
        if total == 0:
            return metrics

        local_count = sum(1 for r in results if r['routing_decision'] == 'local')
        cloud_count = total - local_count

        metrics['total_queries'] = total
        metrics['local_calls'] = local_count
        metrics['cloud_calls'] = cloud_count
        metrics['local_percentage'] = (local_count / total * 100)
        metrics['cloud_percentage'] = (cloud_count / total * 100)

        avg_latency = sum(r.get('latency_ms', 0) for r in results) / total
        metrics['avg_latency_ms'] = avg_latency

        if has_both_models:
            correct = sum(
                1 for r in results
                if (r['routing_decision'] == 'local' and r['local_success']) or
                   (r['routing_decision'] == 'cloud' and r['cloud_success'])
            )
            metrics['routing_accuracy'] = (correct / total * 100)

            avg_local_latency = sum(
                r.get('local_latency_ms', 0) for r in results
            ) / total
            avg_cloud_latency = sum(
                r.get('cloud_latency_ms', 0) for r in results
            ) / total

            metrics['avg_local_latency_ms'] = avg_local_latency
            metrics['avg_cloud_latency_ms'] = avg_cloud_latency

        return metrics

    def _estimate_accuracy(
        self,
        test_data: List[Dict],
        model: str
    ) -> float:
        """Estimate accuracy for a model from dataset.

        Args:
            test_data: Test dataset
            model: 'local' or 'cloud'

        Returns:
            Estimated accuracy
        """
        if not test_data:
            return 0.65 if model == 'local' else 0.95

        accuracies = []
        for sample in test_data:
            if model == 'local':
                if 'qwen_success' in sample:
                    accuracies.append(1.0 if sample['qwen_success'] else 0.0)
            else:
                if 'claude_success' in sample:
                    accuracies.append(1.0 if sample['claude_success'] else 0.0)

        if not accuracies:
            return 0.65 if model == 'local' else 0.95

        return sum(accuracies) / len(accuracies)

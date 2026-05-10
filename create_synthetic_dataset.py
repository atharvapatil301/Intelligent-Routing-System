#!/usr/bin/env python3.11
"""Quick synthetic dataset generator for training (Phase 4)."""
import json
import uuid
from datetime import datetime
from pathlib import Path
from src.feature_extractor import FeatureExtractor


def label_query_heuristic(features: dict) -> int:
    """Simple heuristic to label queries.

    Returns:
        0 for local-sufficient, 1 for cloud-needed
    """
    if features.get('has_complexity_keywords'):
        return 1

    if features.get('has_concurrency_mentions') or features.get('has_algorithm_complexity'):
        return 1

    if features.get('is_design') or features.get('is_optimization'):
        return 1

    if features.get('num_functions_requested', 0) > 2 or features.get('num_classes_requested', 0) > 1:
        return 1

    if features.get('token_count', 0) > 200:
        return 1

    return 0


def generate_synthetic_dataset(num_samples: int = 100, output_path: str = "data/datasets/synthetic_dataset.jsonl"):
    """Generate synthetic dataset quickly using heuristic labeling."""

    with open('data/seed_prompts.json', 'r') as f:
        seed_data = json.load(f)

    prompts = []
    if 'categories' in seed_data:
        for category_data in seed_data['categories'].values():
            prompts.extend(category_data)

    additional_prompts = [
        "Write a hello world function",
        "Create a function to add two numbers",
        "Implement binary search algorithm",
        "Design a scalable microservices architecture",
        "Optimize this sorting algorithm for performance",
        "Create a thread-safe queue implementation",
        "Write a function to check if number is prime",
        "Implement dynamic programming solution for longest common subsequence",
        "Design a distributed caching system",
        "Create a simple calculator function",
        "Implement quicksort with O(n log n) complexity",
        "Build a concurrent web scraper",
        "Write a function to reverse a string",
        "Design an API rate limiting system",
        "Create a simple for loop example",
        "Implement a red-black tree",
        "Optimize database queries for million records",
        "Write a function to find factorial",
        "Design a load balancer architecture",
        "Create a recursive fibonacci function",
    ]

    prompts.extend(additional_prompts)

    feature_extractor = FeatureExtractor()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = []

    print(f"Generating {num_samples} synthetic samples...")

    for i in range(min(num_samples, len(prompts))):
        prompt = prompts[i % len(prompts)]

        features = feature_extractor.extract(prompt, generate_embedding=True)

        label = label_query_heuristic(features.to_dict())

        sample = {
            "sample_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "features": features.to_dict(),
            "embedding": features.embedding.tolist() if features.embedding is not None else None,
            "qwen_output": "",
            "claude_output": "",
            "qwen_success": label == 0,
            "claude_success": True,
            "qwen_score": 0.8 if label == 0 else 0.3,
            "claude_score": 1.0,
            "label": label,
            "qwen_latency_ms": 0.0,
            "claude_latency_ms": 0.0,
            "evaluation_method": "heuristic",
            "notes": "synthetic"
        }

        dataset.append(sample)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    with open(output_path, 'w') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')

    print(f"\n✓ Generated {len(dataset)} samples")
    print(f"  Output: {output_path}")

    local_count = sum(1 for s in dataset if s['label'] == 0)
    cloud_count = sum(1 for s in dataset if s['label'] == 1)
    print(f"\n  Label distribution:")
    print(f"    Local (0): {local_count} ({100*local_count/len(dataset):.1f}%)")
    print(f"    Cloud (1): {cloud_count} ({100*cloud_count/len(dataset):.1f}%)")

    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('-n', '--num-samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('-o', '--output', type=str, default='data/datasets/synthetic_dataset.jsonl',
                        help='Output file path')

    args = parser.parse_args()

    print("=" * 60)
    print("SYNTHETIC DATASET GENERATOR")
    print("=" * 60)
    print()

    output_path = generate_synthetic_dataset(args.num_samples, args.output)

    print()
    print("=" * 60)
    print("Next steps:")
    print(f"  1. Split dataset: ./run.sh split-dataset {output_path}")
    print("  2. Train model: python3.11 train_model.py")
    print("=" * 60)

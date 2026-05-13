#!/usr/bin/env python3
"""Script to generate dataset with real prompts and train neural network."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.dataset_generator import DatasetGenerator
from src.ml_model import RoutingMLModel


def load_prompts(prompts_file: str, limit: int = 100) -> list:
    """Load prompts from seed file."""
    with open(prompts_file, 'r') as f:
        data = json.load(f)

    all_prompts = []
    for category, prompts in data['categories'].items():
        all_prompts.extend(prompts)

    return all_prompts[:limit]


def main():
    print("=" * 80)
    print("DATASET GENERATION AND MODEL TRAINING PIPELINE")
    print("=" * 80)

    print("\n[Step 1/5] Loading prompts...")
    prompts = load_prompts("data/seed_prompts.json", limit=100)
    print(f"Loaded {len(prompts)} prompts")

    print("\n[Step 2/5] Generating dataset with both models + Groq LLM-as-judge evaluation...")
    print("This will take a while as each prompt needs to be processed by:")
    print("  - Qwen (local)")
    print("  - Claude (cloud)")
    print("  - Groq llama-3.1-8b-instant (evaluation)")
    print()

    generator = DatasetGenerator(output_dir="data/datasets")

    try:
        samples = generator.generate_from_prompts(
            prompts=prompts,
            evaluation_method="llm_judge",
            save_interval=10
        )

        generator.save_dataset("complete_dataset.jsonl")

        stats = generator.get_statistics()
        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)
        print(f"Total samples: {stats['total']}")
        print(f"Local sufficient: {stats['local_sufficient']} ({stats['local_percentage']:.1f}%)")
        print(f"Cloud needed: {stats['cloud_needed']} ({stats['cloud_percentage']:.1f}%)")
        print(f"Qwen success rate: {stats['qwen_success_rate']:.1f}%")
        print(f"Claude success rate: {stats['claude_success_rate']:.1f}%")
        print(f"Avg Qwen score: {stats['avg_qwen_score']:.2f}")
        print(f"Avg Claude score: {stats['avg_claude_score']:.2f}")
        print(f"Avg Qwen latency: {stats['avg_qwen_latency_ms']:.2f}ms")
        print(f"Avg Claude latency: {stats['avg_claude_latency_ms']:.2f}ms")

    except Exception as e:
        print(f"\n ERROR during dataset generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("\n[Step 3/5] Splitting dataset (70 train / 15 val / 15 test)...")
    train, val, test = generator.create_splits(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=True
    )

    split_paths = generator.save_splits(train, val, test, prefix="dataset")

    print("\n[Step 4/5] Training neural network...")
    model = RoutingMLModel(
        input_dim=403,
        hidden_dims=[256, 128, 64],
        dropout=0.3,
        learning_rate=0.001
    )

    history = model.train(
        train_path=split_paths['train'],
        val_path=split_paths['val'],
        epochs=50,
        batch_size=16,
        early_stopping_patience=10
    )

    print("\n[Step 5/5] Evaluating on test set...")
    test_metrics = model.evaluate(split_paths['test'])

    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    model.save(model_dir="models")

    with open("models/metrics.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)

    with open("models/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model saved to: models/routing_model.pt")
    print(f"Scaler saved to: models/scaler.pkl")
    print(f"Training history: models/training_history.json")
    print(f"Final metrics: models/metrics.json")


if __name__ == "__main__":
    main()

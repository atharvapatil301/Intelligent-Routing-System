#!/usr/bin/env python3
"""Quick test of dataset generation with a single prompt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.dataset_generator import DatasetGenerator

def main():
    print("Testing dataset generation with a single prompt...")

    generator = DatasetGenerator(output_dir="data/datasets")

    test_prompt = "Write a function to add two numbers"

    try:
        sample = generator.generate_sample(test_prompt, evaluation_method="llm_judge")

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Prompt: {sample.prompt}")
        print(f"\nQwen score: {sample.qwen_score:.2f}")
        print(f"Claude score: {sample.claude_score:.2f}")
        print(f"Label: {'Local' if sample.label == 0 else 'Cloud'}")
        print(f"\nQwen latency: {sample.qwen_latency_ms:.2f}ms")
        print(f"Claude latency: {sample.claude_latency_ms:.2f}ms")

    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

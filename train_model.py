#!/usr/bin/env python3.11
"""Training script for routing neural network (Phase 4)."""
import argparse
import json
from pathlib import Path
from src.ml_model import RoutingMLModel


def main():
    parser = argparse.ArgumentParser(description='Train routing neural network')
    parser.add_argument('--train', type=str, default='data/datasets/dataset_train.jsonl',
                        help='Path to training dataset')
    parser.add_argument('--val', type=str, default='data/datasets/dataset_val.jsonl',
                        help='Path to validation dataset')
    parser.add_argument('--test', type=str, default='data/datasets/dataset_test.jsonl',
                        help='Path to test dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128, 64],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save model')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu, cuda, mps)')

    args = parser.parse_args()

    print("=" * 80)
    print("ROUTING NEURAL NETWORK - TRAINING SCRIPT (PHASE 4)")
    print("=" * 80)
    print()

    # Check if datasets exist
    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)

    if not train_path.exists():
        print(f"❌ Training dataset not found: {train_path}")
        print("\nPlease generate training data first:")
        print("  ./run.sh generate-dataset -n 100 -o data/datasets/ml_training_dataset.jsonl")
        print("  ./run.sh split-dataset data/datasets/ml_training_dataset.jsonl")
        return

    print(f"✓ Training dataset: {train_path}")

    if val_path.exists():
        print(f"✓ Validation dataset: {val_path}")
    else:
        print(f"⚠ Validation dataset not found: {val_path}")
        val_path = None

    if test_path.exists():
        print(f"✓ Test dataset: {test_path}")
    else:
        print(f"⚠ Test dataset not found: {test_path}")
        test_path = None

    print()
    print("=" * 80)
    print("MODEL CONFIGURATION")
    print("=" * 80)
    print(f"Input dimension: 403 (384 embedding + 19 features)")
    print(f"Hidden layers: {args.hidden_dims}")
    print(f"Dropout: {args.dropout}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Early stopping patience: {args.patience}")
    print()

    # Initialize model
    model = RoutingMLModel(
        input_dim=403,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.lr,
        device=args.device
    )

    print()
    print("=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    print(model.model)
    print()
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Train model
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    history = model.train(
        train_path=str(train_path),
        val_path=str(val_path) if val_path else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience
    )

    # Evaluate on test set
    if test_path:
        print()
        print("=" * 80)
        print("EVALUATION ON TEST SET")
        print("=" * 80)
        metrics = model.evaluate(str(test_path))

        print(f"\n📊 Test Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"\n  Samples: {metrics['num_samples']}")
        print(f"  Class distribution:")
        print(f"    Local (0): {metrics['class_distribution']['local']}")
        print(f"    Cloud (1): {metrics['class_distribution']['cloud']}")
        print(f"\n  Confusion Matrix:")
        print(f"    {metrics['confusion_matrix']}")

        # Save metrics to file
        metrics_path = Path(args.model_dir) / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Metrics saved to: {metrics_path}")

    # Save model
    print()
    print("=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    model.save(args.model_dir)

    # Save training history
    history_path = Path(args.model_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review training metrics in models/metrics.json")
    print("2. Check training history in models/training_history.json")
    print("3. Use the model for routing:")
    print("   ./run.sh generate --interactive --strategy ml")
    print()


if __name__ == "__main__":
    main()

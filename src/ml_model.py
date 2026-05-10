"""Neural network model for routing decisions (Phase 4)."""
import os
import json
from typing import Dict, Any, Tuple, List
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle


class RoutingDataset(Dataset):
    """PyTorch dataset for routing decisions."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """Initialize dataset.

        Args:
            features: Feature matrix (N x D)
            labels: Label vector (N,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class RoutingNeuralNetwork(nn.Module):
    """Neural network for routing decisions.

    Architecture:
        Input (403 features) -> Hidden1 (256) -> Dropout ->
        Hidden2 (128) -> Dropout -> Hidden3 (64) -> Output (2 classes)
    """

    def __init__(self, input_dim: int = 403, hidden_dims: List[int] = None, dropout: float = 0.3):
        """Initialize neural network.

        Args:
            input_dim: Number of input features (384 embedding + 19 structural)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(RoutingNeuralNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (binary classification: local=0, cloud=1)
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Logits (batch_size, 2)
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions.

        Args:
            x: Input features

        Returns:
            Probabilities (batch_size, 2)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


class RoutingMLModel:
    """Complete ML model for routing with training and inference."""

    def __init__(
        self,
        input_dim: int = 403,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """Initialize ML model.

        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            device: Device to use ('cpu', 'cuda', or 'mps')
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = RoutingNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)

        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Scaler for feature normalization
        self.scaler = StandardScaler()

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def extract_features_from_sample(self, sample: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from dataset sample.

        Args:
            sample: Dataset sample dictionary

        Returns:
            Feature vector
        """
        features = []

        # Embedding (384 dimensions)
        embedding = sample.get('embedding', sample['features'].get('embedding', []))
        features.extend(embedding)

        # Structural features (19 features)
        feat_dict = sample['features']
        features.extend([
            feat_dict.get('char_count', 0),
            feat_dict.get('token_count', 0),
            feat_dict.get('line_count', 0),
            feat_dict.get('word_count', 0),
            float(feat_dict.get('has_code_block', False)),
            feat_dict.get('num_code_blocks', 0),
            feat_dict.get('num_functions_requested', 0),
            feat_dict.get('num_classes_requested', 0),
            float(feat_dict.get('is_implementation', False)),
            float(feat_dict.get('is_debugging', False)),
            float(feat_dict.get('is_design', False)),
            float(feat_dict.get('is_optimization', False)),
            float(feat_dict.get('is_explanation', False)),
            float(feat_dict.get('has_complexity_keywords', False)),
            len(feat_dict.get('matched_keywords', [])),
            float(feat_dict.get('has_concurrency_mentions', False)),
            float(feat_dict.get('has_algorithm_complexity', False)),
            float(feat_dict.get('has_reasoning_keywords', False)),
            feat_dict.get('similarity_to_failures', 0.0)
        ])

        return np.array(features, dtype=np.float32)

    def load_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from JSONL file.

        Args:
            dataset_path: Path to JSONL dataset file

        Returns:
            Tuple of (features, labels)
        """
        features_list = []
        labels_list = []

        with open(dataset_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                features = self.extract_features_from_sample(sample)
                label = sample['label']

                features_list.append(features)
                labels_list.append(label)

        return np.array(features_list), np.array(labels_list)

    def train(
        self,
        train_path: str,
        val_path: str = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            train_path: Path to training data
            val_path: Path to validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history
        """
        # Load data
        print(f"Loading training data from {train_path}...")
        X_train, y_train = self.load_dataset(train_path)
        print(f"Loaded {len(X_train)} training samples")

        # Normalize features
        print("Normalizing features...")
        X_train = self.scaler.fit_transform(X_train)

        # Create dataset and dataloader
        train_dataset = RoutingDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Validation data
        val_loader = None
        if val_path and Path(val_path).exists():
            print(f"Loading validation data from {val_path}...")
            X_val, y_val = self.load_dataset(val_path)
            print(f"Loaded {len(X_val)} validation samples")
            X_val = self.scaler.transform(X_val)
            val_dataset = RoutingDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print("-" * 60)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation phase
            val_loss = 0.0
            val_acc = 0.0

            if val_loader:
                self.model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)

                        outputs = self.model(batch_features)
                        loss = self.criterion(outputs, batch_labels)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()

                val_loss /= len(val_loader)
                val_acc = val_correct / val_total

                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Print progress
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                # Print progress without validation
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        print("-" * 60)
        print("Training complete!")
        return self.history

    def evaluate(self, test_path: str) -> Dict[str, Any]:
        """Evaluate model on test data.

        Args:
            test_path: Path to test data

        Returns:
            Evaluation metrics
        """
        # Load test data
        print(f"Loading test data from {test_path}...")
        X_test, y_test = self.load_dataset(test_path)
        X_test = self.scaler.transform(X_test)

        # Create dataset and dataloader
        test_dataset = RoutingDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Evaluation
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(batch_labels.numpy())

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
            'num_samples': len(all_labels),
            'class_distribution': {
                'local': int(np.sum(all_labels == 0)),
                'cloud': int(np.sum(all_labels == 1))
            }
        }

        return metrics

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> Tuple[int, float]:
        """Predict routing decision for a single sample.

        Args:
            features: Feature vector (403,)
            threshold: Threshold for cloud routing (if P(cloud) > threshold, route to cloud)

        Returns:
            Tuple of (prediction, confidence) where prediction is 0 (local) or 1 (cloud)
        """
        self.model.eval()

        # Normalize features
        features = self.scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prob_cloud = probabilities[0, 1].item()

        # Make decision based on threshold
        prediction = 1 if prob_cloud > threshold else 0
        confidence = prob_cloud if prediction == 1 else 1 - prob_cloud

        return prediction, confidence

    def save(self, model_dir: str = "models"):
        """Save model and scaler to disk.

        Args:
            model_dir: Directory to save model
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, model_dir / 'routing_model.pt')

        # Save scaler
        with open(model_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"Model saved to {model_dir}")

    def load(self, model_dir: str = "models"):
        """Load model and scaler from disk.

        Args:
            model_dir: Directory containing saved model
        """
        model_dir = Path(model_dir)

        # Load model state
        checkpoint = torch.load(model_dir / 'routing_model.pt', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']

        # Load scaler
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        print(f"Model loaded from {model_dir}")


if __name__ == "__main__":
    # Example usage
    print("Routing Neural Network Model - Phase 4")
    print("=" * 60)

    # Initialize model
    model = RoutingMLModel(
        input_dim=403,
        hidden_dims=[256, 128, 64],
        dropout=0.3,
        learning_rate=0.001
    )

    print(f"Model architecture:")
    print(model.model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.model.parameters()):,}")

#!/usr/bin/env python3
"""
Evaluation script for visualizing model predictions on test data.
Creates a 3x3 grid of test images with model predictions.

Usage:
    python evaluate.py --experiment_path experiments_mlr/2025-06-25_15-30 --model_type mnist_mlr
    python evaluate.py --experiment_path experiments/2025-06-25_12-00 --model_type simple_cnn
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Add src to path for imports
sys.path.append("src")
from src.data.dataloader import MNISTDataModule
from src.models.cnn import MNIST_CNN, MNIST_MLR, SimpleCNN

# MNIST class names
MNIST_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def load_model_from_checkpoint(checkpoint_path: str, model_type: str):
    """Load model from checkpoint file."""
    print(f"Loading model from: {checkpoint_path}")

    if model_type == "mnist_mlr":
        model = MNIST_MLR.load_from_checkpoint(checkpoint_path)
    elif model_type == "mnist_cnn":
        model = MNIST_CNN.load_from_checkpoint(checkpoint_path)
    elif model_type == "simple_cnn":
        model = SimpleCNN.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    return model


def get_random_test_samples(
    data_module: MNISTDataModule, num_samples: int = 30, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get random samples from test dataset."""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup data module
    data_module.setup(stage="test")
    test_dataset = data_module.test_dataset

    # Get random indices
    indices = torch.randperm(len(test_dataset))[:num_samples].tolist()
    subset = Subset(test_dataset, indices)

    # Create dataloader
    test_loader = DataLoader(subset, batch_size=num_samples, shuffle=False)

    # Get the batch
    images, labels = next(iter(test_loader))
    return images, labels


def predict_batch(model, images: torch.Tensor) -> torch.Tensor:
    """Get model predictions for a batch of images."""
    # Get model device
    device = next(model.parameters()).device

    # Move images to same device as model
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        predictions = F.softmax(logits, dim=1)
        predicted_classes = torch.argmax(predictions, dim=1)

    # Move predictions back to CPU for plotting
    return predicted_classes.cpu()


def create_prediction_plot(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    predictions: torch.Tensor,
    save_path: str,
    layout="auto",
):
    """Create and save a 3x3 plot showing images with predictions."""

    fig, axes = plt.subplots(5, 10, figsize=(12, 12))
    fig.suptitle("Model Predictions on Test Data", fontsize=14, fontweight="bold")
    print(f"axes size:{len(axes.flatten())}")
    print(f"img size:{len(images)}")

    for i, ax in enumerate(axes.flat):
        # Convert image to numpy and reshape for display
        img = images[i].squeeze().numpy()

        # Display image
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        # Get labels
        true_label = MNIST_CLASSES[true_labels[i].item()]
        pred_label = MNIST_CLASSES[predictions[i].item()]

        # Set color based on correctness
        color = "green" if true_label == pred_label else "red"

        # Set title with prediction and true label
        title = f"Pred: {pred_label}\nTrue: {true_label}"
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)

    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Prediction plot saved to: {save_path}")

    # Also display if running interactively
    plt.show()


def calculate_accuracy(true_labels: torch.Tensor, predictions: torch.Tensor) -> float:
    """Calculate accuracy for the sample."""
    correct = (true_labels == predictions).sum().item()
    total = len(true_labels)
    return correct / total


def find_best_checkpoint(experiment_path: str) -> str:
    """Find the best checkpoint in the experiment directory."""
    checkpoint_dir = Path(experiment_path) / "checkpoints"

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Look for best checkpoint (contains 'best' in filename)
    best_checkpoints = list(checkpoint_dir.glob("best-*.ckpt"))

    if best_checkpoints:
        return str(best_checkpoints[0])  # Take the first best checkpoint

    # Fallback to any .ckpt file
    all_checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if all_checkpoints:
        return str(all_checkpoints[0])

    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions on test data"
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Path to experiment directory (e.g., experiments_mlr/2025-06-25_15-30)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["mnist_mlr", "mnist_cnn", "simple_cnn"],
        help="Type of model to load",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to visualize (default: 9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Base directory for results (default: results)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="auto",
        help="Subplot layout: 'auto', '3x3', '2x5', '1x10', '5x2', '10x1', etc. (default: auto)",
    )

    args = parser.parse_args()

    try:
        # Find best checkpoint
        checkpoint_path = find_best_checkpoint(args.experiment_path)

        # Load model
        model = load_model_from_checkpoint(checkpoint_path, args.model_type)

        # Setup data module
        data_module = MNISTDataModule(
            batch_size=32
        )  # batch_size doesn't matter for testing

        # Get random test samples
        print(f"Getting {args.num_samples} random test samples (seed={args.seed})...")
        images, true_labels = get_random_test_samples(
            data_module, args.num_samples, args.seed
        )

        # Get predictions
        print("Running inference...")
        predictions = predict_batch(model, images)

        # Calculate accuracy on this sample
        accuracy = calculate_accuracy(true_labels, predictions)
        print(
            f"Sample accuracy: {accuracy:.2%} ({(true_labels == predictions).sum().item()}/{len(true_labels)})"
        )

        # Create save path
        experiment_name = Path(args.experiment_path).name
        save_path = (
            Path(args.results_dir)
            / args.model_type
            / experiment_name
            / "prediction_visualization.png"
        )

        # Create and save plot
        print("Creating visualization...")
        create_prediction_plot(
            images, true_labels, predictions, str(save_path), args.layout
        )

        print("Evaluation completed successfully!")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

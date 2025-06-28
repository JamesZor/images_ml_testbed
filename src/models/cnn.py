# src/models/cnn.py (Enhanced for multiple dtypes)
import logging
from typing import Any, Dict, List, Literal, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import average
from torch.types import Tensor
from torchmetrics import Accuracy, F1Score

logger = logging.getLogger(__name__)


class SimpleCNN(pl.LightningModule):
    """
    Dtype-compatible CNN for MNIST classification.
    Handles float32, float16, bfloat16, and quantized inputs.
    """

    def __init__(
        self,
        # Model architecture
        num_classes: int = 10,
        input_channels: int = 1,
        conv_channels: List[int] = [32, 64],
        conv_kernel_size: int = 3,
        conv_padding: int = 1,
        pool_size: int = 2,
        hidden_dim: int = 128,
        dropout_rate: float = 0.5,
        # Training parameters
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        weight_decay: float = 1e-4,
        # 🎯 NEW: Model dtype configuration
        model_dtype: str = "float32",  # Model's internal dtype
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store parameters
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay

        # 🎯 Parse model dtype
        self.model_dtype = self._parse_dtype(model_dtype)

        # Build CNN layers
        self.conv_layers = self._build_conv_layers(
            input_channels, conv_channels, conv_kernel_size, conv_padding, pool_size
        )

        # Calculate flattened size
        conv_output_size = self._get_conv_output_size(input_channels)

        # Build classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

        # 🎯 Set model dtype
        if self.model_dtype != torch.float32:
            self.to(self.model_dtype)
            logger.info(f"Model converted to {self.model_dtype}")

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        logger.info(f"Created SimpleCNN with {self.count_parameters():,} parameters")
        logger.info(f"Model dtype: {self.model_dtype}")

    def _get_conv_output_size(self, input_channels: int) -> int:
        """Dynamically calculate the flattened conv output size"""
        # Create a dummy input tensor matching MNIST dimensions
        dummy_input = torch.zeros(1, input_channels, 28, 28)

        # Pass through conv layers to get actual output shape
        with torch.no_grad():
            conv_output = self.conv_layers(dummy_input)

        # Calculate flattened size
        flattened_size = conv_output.view(1, -1).size(1)

        # Debug logging
        logger.info(f"Conv layers output shape: {conv_output.shape}")
        logger.info(f"Calculated flattened size: {flattened_size}")

        return flattened_size

    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }

        if dtype_str not in dtype_mapping:
            available = list(dtype_mapping.keys())
            raise ValueError(
                f"Model dtype '{dtype_str}' not supported. Available: {available}"
            )

        return dtype_mapping[dtype_str]

    def _build_conv_layers(
        self, input_channels, conv_channels, kernel_size, padding, pool_size
    ):
        """Build convolutional layers."""
        layers = []
        in_channels = input_channels

        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.MaxPool2d(pool_size),
                ]
            )
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic dtype handling.
        """
        # 🎯 Handle input dtype conversion
        if x.dtype != self.model_dtype:
            # Convert input to model's dtype
            x = x.to(self.model_dtype)

        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

    def _compute_loss_and_metrics(self, batch, stage: str):
        """Compute loss and metrics with dtype handling."""
        x, y = batch

        # Forward pass (handles dtype conversion internally)
        logits = self(x)

        # 🎯 Ensure loss computation in float32 for stability
        if logits.dtype != torch.float32:
            loss = F.cross_entropy(logits.float(), y)
        else:
            loss = F.cross_entropy(logits, y)

        # Get predictions
        preds = torch.argmax(logits, dim=1)

        # Compute metrics
        if stage == "train":
            acc = self.train_acc(preds, y)
            f1 = self.train_f1(preds, y)
        elif stage == "val":
            acc = self.val_acc(preds, y)
            f1 = self.val_f1(preds, y)
        elif stage == "test":
            acc = self.test_acc(preds, y)
            f1 = self.test_f1(preds, y)

        return loss, acc, f1

    def training_step(self, batch, batch_idx):
        """Training step with dtype handling."""
        loss, acc, f1 = self._compute_loss_and_metrics(batch, "train")

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, acc, f1 = self._compute_loss_and_metrics(batch, "val")

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, acc, f1 = self._compute_loss_and_metrics(batch, "test")

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_f1", f1, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer with dtype considerations."""

        # 🎯 Some optimizers work better with specific dtypes
        if self.model_dtype == torch.float16 and self.optimizer.lower() == "adam":
            logger.warning("Adam with float16 can be unstable. Consider AdamW or SGD.")

        if self.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information including dtype."""
        return {
            "num_classes": self.num_classes,
            "total_parameters": self.count_parameters(),
            "model_dtype": str(self.model_dtype),
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
        }


class MNIST_MLR(pl.LightningModule):
    """
    Model for simple logistic Regression.
    """

    def __init__(
        self,
        # training parameters
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        weight_decay: float = 1e-4,
    ):

        super(MNIST_MLR, self).__init__()
        self.save_hyperparameters()

        # store parameters
        self.learning_rate: float = learning_rate
        self.optimizer: str = optimizer
        self.weight_decay: float = weight_decay

        # metrics
        self.acc = Accuracy(task="multiclass", num_classes=10)
        self.f1 = F1Score(task="multiclass", num_classes=10, average="macro")

        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def forward(self, x):
        return self.linear(x)

    def _compute_loss_and_metrics(
        self, batch: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x, y = batch
        logits = self(x)
        loss: Tensor = F.cross_entropy(logits, y)
        acc: Tensor = self.acc(logits, y)
        f1: Tensor = self.f1(logits, y)

        return loss, acc, f1

    def _log_metircs(self, stage: str, loss: Tensor, acc: Tensor, f1: Tensor) -> None:
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        self.log(f"{stage}_f1", f1, prog_bar=True)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, acc, f1 = self._compute_loss_and_metrics(batch)
        self._log_metircs("train", loss, acc, f1)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, acc, f1 = self._compute_loss_and_metrics(batch)
        self._log_metircs("val", loss, acc, f1)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, acc, f1 = self._compute_loss_and_metrics(batch)
        self._log_metircs("test", loss, acc, f1)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "model": "MLR",
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
        }


class MNIST_CNN(pl.LightningModule):
    """
    Model for a simple CNN based on the ISLP book, for the mnist data set.
    """

    def __init__(
        self,
        # Training parameters - to be set by hydra.
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        weight_decay: float = 1e-4,
    ):

        super(MNIST_CNN, self).__init__()
        self.save_hyperparameters()

        # store model's parameters into the class, to be used by the methods.
        self.learning_rate: float = learning_rate
        self.optimizer: str = optimizer
        self.weight_decay: float = weight_decay

        # metrics for training, validation and training.
        self.acc = Accuracy(task="multiclass", num_classes=10)
        self.f1 = F1Score(task="multiclass", num_classes=10, average="macro")

        self.layer_1 = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(), nn.Dropout(0.4)
        )
        self.layer_2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4))
        self._forward = nn.Sequential(self.layer_1, self.layer_2, nn.Linear(128, 10))

    def forward(self, x: Tensor):
        return self._forward(x)

    def _compute_loss_and_metrics(
        self, batch: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Interal method to compute the metics for the training, validation, and test.
        """
        x, y = batch
        logits: Tensor = self(x)
        loss: Tensor = F.cross_entropy(logits, y)
        acc: Tensor = self.acc(logits, y)
        f1: Tensor = self.f1(logits, y)
        return loss, acc, f1

    def _log_metircs(
        self,
        stage: Literal["train", "val", "test"],
        loss: Tensor,
        acc: Tensor,
        f1: Tensor,
    ) -> None:
        """
        Interal method to log the results of the metrics
        """
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=False)
        self.log(f"{stage}_f1", f1, prog_bar=False)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, acc, f1 = self._compute_loss_and_metrics(batch)
        self._log_metircs("train", loss, acc, f1)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, acc, f1 = self._compute_loss_and_metrics(batch)
        self._log_metircs("val", loss, acc, f1)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, acc, f1 = self._compute_loss_and_metrics(batch)
        self._log_metircs("test", loss, acc, f1)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return a dictionary regarding the model and parameters to be printed.
        """
        return {
            "model": "Very Simple CNN",
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
        }

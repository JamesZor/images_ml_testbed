# src/data/dataloader.py (Enhanced with configurable dtypes)
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

logger = logging.getLogger(__name__)


class ConfigurableDtypeTransform:
    """Custom transform to convert tensor to specified dtype."""

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.dtype)

    def __repr__(self):
        return f"ConfigurableDtypeTransform(dtype={self.dtype})"


class MNISTDataModule(pl.LightningDataModule):
    """
    Enhanced MNIST DataModule with configurable tensor dtypes.

    Supports: float32, float16, bfloat16, int8, int16, int32, etc.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        # Data splitting
        val_split: float = 0.1,
        # Image preprocessing
        normalize: bool = True,
        # 🎯 NEW: Configurable tensor dtype
        tensor_dtype: str = "float32",  # "float32", "float16", "bfloat16", "int8", etc.
        # Data augmentation
        use_augmentation: bool = True,
        rotation_degrees: float = 10.0,
        # Random seed
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Core parameters
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        # Data parameters
        self.val_split = val_split
        self.normalize = normalize
        self.use_augmentation = use_augmentation
        self.rotation_degrees = rotation_degrees
        self.random_state = random_state

        # 🎯 Dtype configuration
        self.tensor_dtype_str = tensor_dtype
        self.tensor_dtype = self._parse_dtype(tensor_dtype)

        # MNIST constants
        self.num_classes = 10
        self.input_shape = (1, 28, 28)

        # Calculate normalization values based on dtype
        if self.normalize:
            self.mean, self.std = self._get_normalization_values()
        else:
            self.mean = (0.5,)
            self.std = (0.5,)

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        logger.info(
            f"Initialized MNISTDataModule with dtype={self.tensor_dtype}, "
            f"batch_size={batch_size}, normalize={normalize}"
        )

        # Log dtype-specific warnings
        self._log_dtype_warnings()

    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_mapping = {
            # Float types
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
            "double": torch.float64,
            "half": torch.float16,
            # Integer types
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
            # Boolean
            "bool": torch.bool,
        }

        if dtype_str not in dtype_mapping:
            available_types = list(dtype_mapping.keys())
            raise ValueError(
                f"Unsupported dtype '{dtype_str}'. "
                f"Available types: {available_types}"
            )

        return dtype_mapping[dtype_str]

    def _get_normalization_values(self) -> Tuple[Tuple[float], Tuple[float]]:
        """Get normalization values appropriate for the tensor dtype."""

        if self.tensor_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            # For integer types, we need to scale MNIST values appropriately
            if self.tensor_dtype == torch.int8:
                # Scale to [-128, 127] range
                mean = (0.1307 * 255 - 128,)  # Scale and shift for int8
                std = (0.3081 * 255,)
            elif self.tensor_dtype == torch.uint8:
                # Scale to [0, 255] range
                mean = (0.1307 * 255,)
                std = (0.3081 * 255,)
            else:
                # For int16/int32, use larger range
                scale = 32767 if self.tensor_dtype == torch.int16 else 2147483647
                mean = (0.1307 * scale,)
                std = (0.3081 * scale,)
        else:
            # For float types, use standard MNIST normalization
            mean = (0.1307,)
            std = (0.3081,)

        return mean, std

    def _log_dtype_warnings(self):
        """Log important warnings about dtype choices."""

        if self.tensor_dtype == torch.float16:
            logger.warning(
                "Using float16: May cause training instability. "
                "Consider mixed precision training instead."
            )

        elif self.tensor_dtype in [torch.int8, torch.uint8]:
            logger.warning(
                "Using int8: Information loss expected. "
                "Only recommended for inference or specific quantization research."
            )

        elif self.tensor_dtype == torch.bfloat16:
            logger.info("Using bfloat16: Good for TPU training and some modern GPUs.")

        elif self.tensor_dtype == torch.float32:
            logger.info("Using float32: Standard precision for training.")

    def _get_train_transforms(self) -> transforms.Compose:
        """Create training transforms with configurable dtype."""
        transform_list = []

        # Convert to tensor (always float32 initially)
        transform_list.append(transforms.ToTensor())

        # Data augmentation (only if enabled)
        if self.use_augmentation:
            transform_list.extend(
                [
                    transforms.RandomRotation(degrees=self.rotation_degrees),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                    ),
                ]
            )

        # Normalization (before dtype conversion)
        transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        # 🎯 Dtype conversion (LAST step)
        if self.tensor_dtype != torch.float32:
            transform_list.append(ConfigurableDtypeTransform(self.tensor_dtype))

        return transforms.Compose(transform_list)

    def _get_val_test_transforms(self) -> transforms.Compose:
        """Create validation/test transforms with configurable dtype."""
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]

        # 🎯 Dtype conversion
        if self.tensor_dtype != torch.float32:
            transform_list.append(ConfigurableDtypeTransform(self.tensor_dtype))

        return transforms.Compose(transform_list)

    def prepare_data(self) -> None:
        """Download MNIST data if not already present."""
        logger.info("Downloading MNIST dataset...")

        torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True)

        logger.info(f"MNIST dataset downloaded to {self.data_dir}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets with appropriate transforms."""

        train_transforms = self._get_train_transforms()
        val_test_transforms = self._get_val_test_transforms()

        if stage == "fit" or stage is None:
            # Load and split training data
            full_train = torchvision.datasets.MNIST(
                root=self.data_dir, train=True, transform=val_test_transforms
            )

            total_size = len(full_train)
            val_size = int(self.val_split * total_size)
            train_size = total_size - val_size

            generator = torch.Generator().manual_seed(self.random_state)
            train_indices, val_indices = random_split(
                range(total_size), [train_size, val_size], generator=generator
            )

            # Create datasets with appropriate transforms
            self.train_dataset = torchvision.datasets.MNIST(
                root=self.data_dir, train=True, transform=train_transforms
            )
            self.train_dataset = torch.utils.data.Subset(
                self.train_dataset, train_indices.indices
            )

            self.val_dataset = torchvision.datasets.MNIST(
                root=self.data_dir, train=True, transform=val_test_transforms
            )
            self.val_dataset = torch.utils.data.Subset(
                self.val_dataset, val_indices.indices
            )

            logger.info(
                f"Train dataset: {len(self.train_dataset)} samples, dtype={self.tensor_dtype}"
            )
            logger.info(
                f"Val dataset: {len(self.val_dataset)} samples, dtype={self.tensor_dtype}"
            )

        if stage == "test" or stage is None:
            self.test_dataset = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, transform=val_test_transforms
            )
            logger.info(
                f"Test dataset: {len(self.test_dataset)} samples, dtype={self.tensor_dtype}"
            )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader with dtype-aware settings."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
            and self.tensor_dtype.is_floating_point,  # Only pin float tensors
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and self.tensor_dtype.is_floating_point,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and self.tensor_dtype.is_floating_point,
            persistent_workers=self.persistent_workers,
        )

    def get_data_info(self) -> Dict[str, Any]:
        """Return information about the dataset including dtype."""
        return {
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "tensor_dtype": str(self.tensor_dtype),
            "train_size": len(self.train_dataset) if self.train_dataset else None,
            "val_size": len(self.val_dataset) if self.val_dataset else None,
            "test_size": len(self.test_dataset) if self.test_dataset else None,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "augmentation": self.use_augmentation,
        }

    def sample_batch_info(self) -> Dict[str, Any]:
        """Get information about a sample batch for debugging."""
        if not self.train_dataset:
            self.setup("fit")

        loader = self.train_dataloader()
        batch = next(iter(loader))
        images, labels = batch

        return {
            "batch_shape": tuple(images.shape),
            "tensor_dtype": images.dtype,
            "device": images.device,
            "min_value": float(images.min()),
            "max_value": float(images.max()),
            "mean_value": float(images.mean()),
            "memory_usage_mb": images.numel() * images.element_size() / (1024 * 1024),
        }

# src.data.dataloader.py

import logging
import os
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


class MNISTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MNIST dataset.

    This class handles:
    - Downloading MNIST data
    - Applying transforms and augmentations
    - Creating train/val/test splits
    - Creating DataLoaders with proper batching
    """

    def __init__(
        self,
        dataset_name: str = "MNIST",
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        peristent_workers: bool = True,
        # Data splitting.
        val_split: float = 0.1,  # %10 of train data fro the valuation split
        normalize: bool = False,
        use_augmentation: bool = False,
        rotation_degrees: float = 10.0,
        # Random seed
        random_state: int = 42,
        **kwargs,  # for other arguments if need.
    ) -> None:
        super().__init__()

        # store the hyperparameters
        self.save_hyperparameters()

        # core parameters.
        self.dateset_name: str = dataset_name
        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.peristent_workers: bool = peristent_workers

        # data parameters.
        self.val_split: float = val_split
        self.normalize: bool = normalize
        self.use_augmentation: bool = use_augmentation
        self.rotation_degrees: float = rotation_degrees

        # random state for repeat.
        self.random_state: int = random_state

        # MNIST constants.
        self.num_classes: int = 10
        self.input_shape: Tuple[int, int, int] = (1, 28, 28)

        # Calculation of normalization values for MNIST standard.
        if self.normalize:
            self.mean: Tuple[float, ...] = (0.1307,)
            self.std: Tuple[float, ...] = (0.3081,)
        else:
            self.mean: Tuple[float, ...] = (0.5,)
            self.std: Tuple[float, ...] = (0.5,)

        # initialise dataset as None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        logger.info(
            f"Initialized MNISTDataModule with batch_size={batch_size}, "
            f"val_split={val_split}, normalize={normalize}, "
            f"augmentation={use_augmentation}"
        )

    def prepare_data(self) -> None:
        """
        To Download MNIST data if not already downloaded.
        This is called only once per node in distributed training.
        """

        logger.info("Downloading MNIST dataset...")

        # Download the train and tests sets using pytorch vision.
        [
            torchvision.datasets.MNIST(root=self.data_dir, train=arg, download=True)
            for arg in (True, False)
        ]

        logger.info(f"MNIST dataset downloaded to {self.data_dir}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create train/val/test dataset with appropriate transformss.
        This is called on every GPU in distributed training.
        """

        train_transforms: transforms.Compose = self._get_train_transforms()
        val_test_transforms: transforms.Compose = self._get_val_test_transforms()

        if stage == "fit" or stage is None:
            # Load full training dataset.
            full_train: Dataset = torchvision.datasets.MNIST(
                root=self.data_dir, train=True, transform=val_test_transforms
            )

            # compute the split size
            total_size: int = len(full_train)
            val_size: int = int(self.val_split * total_size)
            train_size: int = total_size - val_size

            logger.info(f"Splitting train set: {train_size} train, {val_size} val")

            # Create random split
            generator: torch.Generator = torch.Generator().manual_seed(
                self.random_state
            )
            indices = torch.randperm(total_size, generator=generator)
            train_indices = indices[:train_size].tolist()
            val_indices = indices[train_size:].tolist()

            # create training dataset with traning transforms
            self.train_dataset = torchvision.datasets.MNIST(
                root=sefl.data_dir, train=True, transform=train_transforms
            )

# train.py
import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchinfo import summary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
    """Main training function."""

    logger.info("Starting MNIST training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seeds for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
        logger.info(f"Seed set to {cfg.seed}")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data module
    logger.info("Initializing data module...")
    data_module = hydra.utils.instantiate(cfg.data)

    # Setup data to get info
    data_module.prepare_data()
    data_module.setup()
    data_info = data_module.get_data_info()

    logger.info(f"Data statistics: {data_info}")

    # Initialize model
    logger.info("Initializing model...")
    model = hydra.utils.instantiate(cfg.model)

    logger.info(f"Model: {model}")
    logger.info(f"Model info: {model.get_model_info()}")
    model_summary = summary(model, input_size=tuple(cfg.data.input_size), col_names=['input_size', 'output_size', 'num_params'], verbose=0)
    logger.info(f"Model summary:\n{model_summary}")

    # Initialize W&B logger
    wandb_logger = None
    if cfg.wandb.mode != "disabled":
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.experiment.name}_{model.__class__.__name__}",
            tags=cfg.experiment.tags,
            notes=cfg.experiment.notes,
            save_dir=output_dir,
        )

        # Log configuration
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

        # Log data info and model info
        wandb_logger.experiment.config.update(
            {"data_info": data_info, "model_info": model.get_model_info()}
        )

    # Initialize callbacks
    callbacks = []

    # Early stopping
    if cfg.training.patience > 0:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.patience,
            min_delta=cfg.training.min_delta,
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stopping)

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.training.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        deterministic=cfg.get("seed") is not None,
        enable_progress_bar=True,
    )

    # Train the model
    logger.info("Starting training...")
    trainer.fit(model, data_module)

    # Test the model
    logger.info("Testing model...")
    test_results = trainer.test(model, data_module, ckpt_path="best")

    # Log final results
    best_val_loss = checkpoint_callback.best_model_score.item()
    test_loss = test_results[0]["test_loss"]
    test_acc = test_results[0]["test_acc"]
    test_f1 = test_results[0]["test_f1"]

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Test F1-score: {test_f1:.4f}")

    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "test_results": test_results[0],
            "best_val_loss": best_val_loss,
        },
        final_model_path,
    )

    logger.info(f"Model saved to {final_model_path}")

    # Log final metrics to W&B
    if wandb_logger:
        wandb_logger.experiment.log(
            {
                "final/best_val_loss": best_val_loss,
                "final/test_loss": test_loss,
                "final/test_acc": test_acc,
                "final/test_f1": test_f1,
            }
        )

    # Close W&B run
    if wandb_logger:
        wandb.finish()


if __name__ == "__main__":
    main()

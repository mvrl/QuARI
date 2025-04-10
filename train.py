import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Optional

from models import PersonalizedRetrievalModule
from feature_extractors import FeatureExtractorFactory
from datasets import SimpleTextImageDataset
from torch.utils.data import DataLoader

class SimpleDataModule(pl.LightningDataModule):
    """
    Simple PyTorch Lightning DataModule for text-image retrieval.
    """
    
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        batch_size: int = 512,
        num_workers: int = 4,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the data module.
        
        Args:
            json_path: Path to JSON file with text-image pairs
            image_dir: Directory containing images
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            shuffle: Whether to shuffle the data
            val_split: Fraction of data to use for validation
            max_samples: Maximum number of samples to use
            seed: Random seed for splitting
        """
        super().__init__()
        self.json_path = json_path
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.max_samples = max_samples
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup the data module for training, validation, or testing.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        self.train_dataset = SimpleTextImageDataset(
            json_path=self.json_path,
            image_dir=self.image_dir,
            max_samples=self.max_samples,

        )

        self.val_dataset = SimpleTextImageDataset(
            json_path=self.json_path.replace("train", "val"),
            image_dir=self.image_dir,
            max_samples=self.max_samples,
        )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train personalized text-to-image retrieval model")
    
    # Dataset arguments
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON file with text-image pairs")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers for dataloaders")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    
    # Feature extractor arguments
    parser.add_argument("--extractor_type", type=str, default="clip", choices=["clip", "siglip", "siglip2"], 
                        help="Type of feature extractor")
    parser.add_argument("--extractor_model", type=str, default="ViT-B/32", 
                        help="Model name for feature extractor")
    parser.add_argument("--freeze_extractors", action="store_true", 
                        help="Whether to freeze feature extractors")
    
    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=768, 
                        help="Hidden dimension for transformer")
    parser.add_argument("--num_denoising_steps", type=int, default=8, 
                        help="Number of denoising steps")
    parser.add_argument("--use_residual", action="store_true", 
                        help="Whether to use residual connection in personalization")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Number of attention heads in transformer")
    parser.add_argument("--num_encoder_layers", type=int, default=6,
                        help="Number of transformer encoder layers")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate")
    parser.add_argument("--scale_factor", type=float, default=0.1,
                        help="Scale factor for diffusion noise and updates")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.07, 
                        help="Temperature for contrastive loss")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Patience for early stopping")
    parser.add_argument("--precision", type=str, default="16-mixed", 
                        help="Precision for training")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                        help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="personalized_retrieval", 
                        help="Experiment name")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data module
    data_module = SimpleDataModule(
        json_path=args.json_path,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        max_samples=args.max_samples
    )
    
    # Create model
    model = PersonalizedRetrievalModule(
        extractor_type=args.extractor_type,
        extractor_model=args.extractor_model,
        hidden_dim=args.hidden_dim,
        num_denoising_steps=args.num_denoising_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        use_residual=args.use_residual,
        freeze_extractors=args.freeze_extractors,
        # Transformer hypernetwork parameters
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout,
        # Diffusion parameters
        scale_factor=args.scale_factor
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, args.experiment_name, "checkpoints"),
        filename="model-{epoch:02d}-{val_loss:.3e}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Set up trainer
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        max_epochs=args.max_epochs,
        precision=args.precision,
        accelerator="auto",  # Automatically detect GPU/CPU
        devices="auto",      # Automatically detect number of devices
        log_every_n_steps=10,
        val_check_interval=1500 if not args.max_samples else 5,  # Validate every 1500 training steps
    )
    
    torch.set_float32_matmul_precision("high")

    # Train model
    trainer.fit(model, data_module)
    
    print("Training complete!")
    print(f"Model checkpoints saved to: {checkpoint_callback.dirpath}")
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    

if __name__ == "__main__":
    main()
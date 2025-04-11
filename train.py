import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Optional, Dict, Any
import warnings

from models import PersonalizedRetrievalModule
from feature_extractors import FeatureExtractorFactory
from datasets import SimpleTextImageDataset, EmbeddingsDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def precompute_embeddings(data_module, extractor, output_path, batch_size=1024):
    """
    Precompute and save embeddings to reduce memory usage during training.
    
    Args:
        data_module: PyTorch Lightning DataModule
        extractor: Feature extractor
        output_path: Path to save embeddings
        batch_size: Batch size for extraction
        device: Device to run extraction on
    """
    extractor.to(device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Ensure data module is set up
    data_module.setup()
    
    # Extract from training set
    train_loader = DataLoader(
        data_module.train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True
    )
    
    # Initialize storage
    all_text_embeddings = []
    all_image_embeddings = []
    all_query_texts = []
    
    # Extract features
    print(f"Extracting features from {len(train_loader)} batches...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(train_loader)):
            
            # Extract text and image features
            query_texts = batch["query_text"]
            query_images = batch["query_image"].to(device)
            
            try:
                text_features = extractor.extract_text_features(query_texts)
                image_features = extractor.extract_image_features(query_images)
                
                # Store
                all_text_embeddings.append(text_features.cpu())
                all_image_embeddings.append(image_features.cpu())
                all_query_texts.extend(query_texts)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"OOM in batch {i}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
    
    # Concatenate
    text_embeddings = torch.cat(all_text_embeddings, dim=0)
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    
    # Save
    embeddings_data = {
        "text_embeddings": text_embeddings,
        "image_embeddings": image_embeddings,
        "query_texts": all_query_texts
    }
    
    print(f"Saving {len(all_query_texts)} embeddings to {output_path}")
    torch.save(embeddings_data, output_path)
    
    return output_path


class PrecomputedDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule that handles precomputed embeddings.
    """
    
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        extractor_type: str,
        extractor_model: str,
        batch_size: int = 1024,
        num_workers: int = 10,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
        precomputed_dir: str = "./precomputed_embeddings",
        force_recompute: bool = False
    ):
        super().__init__()
        self.json_path = json_path
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.precomputed_dir = precomputed_dir
        self.force_recompute = force_recompute
        
        # Feature extractor details
        self.extractor_type = extractor_type
        self.extractor_model = extractor_model
        
        # Create raw data module for precomputation
        self.raw_data_module = SimpleDataModule(
            json_path=json_path,
            image_dir=image_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            max_samples=max_samples
        )
        
        # Create feature extractor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = FeatureExtractorFactory.create_extractor(
            extractor_type=extractor_type,
            model_name=extractor_model,
            device=device
        )
        
        # Paths for precomputed embeddings
        os.makedirs(precomputed_dir, exist_ok=True)
        model_name = extractor_model.replace("/", "_")
        self.train_embeddings_path = os.path.join(
            precomputed_dir, f"train_{model_name}.pt"
        )
        self.val_embeddings_path = os.path.join(
            precomputed_dir, f"val_{model_name}.pt"
        )
        
        # Embeddings data module
        self.embeddings_data_module = None
    
    def prepare_data(self):
        """Precompute embeddings if needed."""
        # Check if precomputed embeddings exist
        train_exists = os.path.exists(self.train_embeddings_path)
        val_exists = os.path.exists(self.val_embeddings_path)
        
        # Recompute if forced or missing
        if self.force_recompute or not train_exists:
            print(f"Precomputing training embeddings...")
            precompute_embeddings(
                self.raw_data_module,
                self.feature_extractor,
                self.train_embeddings_path,
                batch_size=1024
            )
        
        if self.force_recompute or not val_exists:
            print(f"Precomputing validation embeddings...")
            # Create a validation data module using the validation JSON
            val_json_path = self.json_path.replace("train", "val")
            if os.path.exists(val_json_path):
                val_data_module = SimpleDataModule(
                    json_path=val_json_path,
                    image_dir=self.image_dir,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                    max_samples=self.max_samples
                )
                precompute_embeddings(
                    val_data_module,
                    self.feature_extractor,
                    self.val_embeddings_path,
                    batch_size=1024
                )
            else:
                print(f"Warning: No validation JSON found at {val_json_path}")
    
    def setup(self, stage: Optional[str] = None):
        """Set up the embeddings data module."""
        # Create embeddings data module
        self.embeddings_data_module = EmbeddingsDataModule(
            train_embeddings_path=self.train_embeddings_path,
            val_embeddings_path=self.val_embeddings_path if os.path.exists(self.val_embeddings_path) else None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle
        )
        self.embeddings_data_module.setup(stage)
    
    def train_dataloader(self):
        """Return training dataloader from embeddings data module."""
        return self.embeddings_data_module.train_dataloader()
    
    def val_dataloader(self):
        """Return validation dataloader from embeddings data module."""
        return self.embeddings_data_module.val_dataloader()


class SimpleDataModule(pl.LightningDataModule):
    """
    Simple PyTorch Lightning DataModule for text-image retrieval.
    """
    
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        batch_size: int = 256,
        num_workers: int = 10,
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
            max_samples: Maximum number of samples to use
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

        val_path = self.json_path.replace("train", "val")
        if os.path.exists(val_path):
            self.val_dataset = SimpleTextImageDataset(
                json_path=val_path,
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
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader."""
        if hasattr(self, 'val_dataset'):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train personalized text-to-image retrieval model")
    
    # Dataset arguments
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON file with text-image pairs")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers for dataloaders")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    
    # Precomputed embeddings arguments
    parser.add_argument("--use_precomputed", action="store_true", 
                        help="Whether to use precomputed embeddings")
    parser.add_argument("--precomputed_dir", type=str, default="./precomputed_embeddings",
                        help="Directory for precomputed embeddings")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Force recomputation of embeddings")
    
    # Feature extractor arguments
    parser.add_argument("--extractor_type", type=str, default="clip", choices=["clip", "siglip", "siglip2"], 
                        help="Type of feature extractor")
    parser.add_argument("--extractor_model", type=str, default="openai/clip-vit-base-patch32", 
                        help="Model name for feature extractor")
    parser.add_argument("--freeze_extractors", action="store_true", 
                        help="Whether to freeze feature extractors")
    
    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=512, 
                        help="Hidden dimension for transformer")
    parser.add_argument("--num_denoising_steps", type=int, default=4, 
                        help="Number of denoising steps")
    parser.add_argument("--use_residual", action="store_true", 
                        help="Whether to use residual connection in personalization")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Number of attention heads in transformer")
    parser.add_argument("--num_encoder_layers", type=int, default=4,
                        help="Number of transformer encoder layers")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate")
    parser.add_argument("--scale_factor", type=float, default=0.1,
                        help="Scale factor for diffusion noise and updates")
    parser.add_argument("--matrix_sequence_len", type=int, default=4,
                        help="Length of sequence for matrix reshaping")
    parser.add_argument("--low_rank_dim", type=int, default=64,
                        help="Dimension for low-rank matrix factorization")
    parser.add_argument("--column_wise", action="store_true", default=True,
                        help="Use column-wise processing (default: True)")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.07, 
                        help="Temperature for contrastive loss")
    parser.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=5, 
                        help="Patience for early stopping")
    parser.add_argument("--precision", type=str, default="16-mixed", 
                        help="Precision for training")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--accumulate_grad_batches", type=int, default=2,
                        help="Number of batches to accumulate gradients")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                        help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="personalized_retrieval", 
                        help="Experiment name")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    if not args.freeze_extractors and args.precompute_embeddings:
        raise ValueError("Cannot use precomputed embeddings with unfrozen extractors.")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logger = TensorBoardLogger(
        save_dir=os.path.join(args.output_dir, "logs"),
        name=args.experiment_name
    )
    
    # Use precomputed embeddings if specified
    if args.use_precomputed:
        print("Using precomputed embeddings for memory efficiency")
        data_module = PrecomputedDataModule(
            json_path=args.json_path,
            image_dir=args.image_dir,
            extractor_type=args.extractor_type,
            extractor_model=args.extractor_model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            max_samples=args.max_samples,
            precomputed_dir=args.precomputed_dir,
            force_recompute=args.force_recompute
        )
    else:
        print("Using on-the-fly feature extraction (may require more memory)")
        data_module = SimpleDataModule(
            json_path=args.json_path,
            image_dir=args.image_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            max_samples=args.max_samples
        )
    
        # Prepare data (if using precomputed embeddings)
    if args.use_precomputed:
        data_module.prepare_data()

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
        scale_factor=args.scale_factor,
        # Additional parameters for column-wise approach
        matrix_sequence_len=args.matrix_sequence_len,
        low_rank_dim=args.low_rank_dim
    )
    
    # # Enable gradient checkpointing for transformers if available
    # for name, module in model.named_modules():
    #     if hasattr(module, 'transformer_encoder'):
    #         print(f"Enabling gradient checkpointing for {name}")
    #         module.transformer_encoder.enable_input_require_grads()
    #         module.transformer_encoder.gradient_checkpointing_enable()
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, args.experiment_name, "checkpoints"),
        filename="model-{epoch:02d}-{val_loss:.3e}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
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
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        max_epochs=args.max_epochs,
        precision=args.precision,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=20,
        val_check_interval=750,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    
    # Enable high precision matmul
    torch.set_float32_matmul_precision("high")
    
    
    # Train model
    try:
        print("Starting training...")
        trainer.fit(model, data_module)
        print("Training complete!")
        print(f"Model checkpoints saved to: {checkpoint_callback.dirpath}")
        print(f"Best model path: {checkpoint_callback.best_model_path}")
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("\n\nERROR: CUDA out of memory. Try these options:")
            print("1. Use --use_precomputed flag to precompute embeddings")
            print("2. Reduce --batch_size (try halving it)")
            print("3. Reduce --hidden_dim (try 256 instead of 512)")
            print("4. Reduce --matrix_sequence_len (try 8 instead of 16)")
            print("5. Reduce --num_encoder_layers (try 1 instead of 2)")
            print("6. Reduce --nhead (try 2 instead of 4)")
            raise
        else:
            raise e


if __name__ == "__main__":
    main()
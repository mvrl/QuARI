import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
import numpy as np


class SimpleTextImageDataset(Dataset):
    """
    A simple text-image dataset for personalized retrieval.
    Expects a directory of images and a JSON file with text-image pairs.
    """
    
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        resize_resolution: int = 224,
    ):
        """
        Initialize the dataset.
        
        Args:
            json_path: Path to JSON file with text-image pairs
            image_dir: Directory containing images
            transform: Image transformation pipeline
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.image_dir = image_dir
        self.resize_resolution = resize_resolution
        
        # Load annotations
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Limit samples if specified
        if max_samples is not None:
            self.data = self.data[:max_samples]
        
        # Set up image transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.resize_resolution, self.resize_resolution)),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with keys:
            - query_text: Text query
            - target_image: Positive image tensor
        """
        item = self.data[idx]
        
        # Get text query
        query_text = item["caption"]
        
        # Load positive image
        image_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        
        return {
            "query_text": query_text,
            "target_image": image_tensor
        }


class PrecomputedEmbeddingsDataset(Dataset):
    """
    Dataset that loads precomputed embeddings.
    Useful for faster training by avoiding feature extraction during training.
    """
    
    def __init__(
        self,
        embeddings_path: str,
        transform: Optional[Callable] = None
    ):
        """
        Initialize precomputed embeddings dataset.
        
        Args:
            embeddings_path: Path to file containing precomputed embeddings
            transform: Optional transform to apply to items
        """
        self.transform = transform
        
        # Load precomputed embeddings
        self.data = torch.load(embeddings_path)
        
        # Validate data format
        required_keys = ["text_embeddings", "image_embeddings"]
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Missing required key '{key}' in precomputed embeddings file")
        
        # Optional elements
        self.has_query_texts = "query_texts" in self.data
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data["text_embeddings"])
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing embeddings
        """
        item = {
            "text_features": self.data["text_embeddings"][idx],
            "target_image_features": self.data["image_embeddings"][idx]
        }
        
        if self.has_query_texts:
            item["query_text"] = self.data["query_texts"][idx]
        
        # Apply transform if specified
        if self.transform is not None:
            item = self.transform(item)
        
        return item


class EmbeddingsDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for personalized text-image retrieval
    with precomputed embeddings.
    """
    
    def __init__(
        self,
        train_embeddings_path: str,
        val_embeddings_path: Optional[str] = None,
        test_embeddings_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True
    ):
        """
        Initialize precomputed embeddings data module.
        
        Args:
            train_embeddings_path: Path to training embeddings file
            val_embeddings_path: Path to validation embeddings file
            test_embeddings_path: Path to test embeddings file
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            shuffle: Whether to shuffle the training data
        """
        super().__init__()
        self.train_embeddings_path = train_embeddings_path
        self.val_embeddings_path = val_embeddings_path
        self.test_embeddings_path = test_embeddings_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup the data module for training, validation, or testing.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = PrecomputedEmbeddingsDataset(self.train_embeddings_path)
            
            if self.val_embeddings_path:
                self.val_dataset = PrecomputedEmbeddingsDataset(self.val_embeddings_path)
        
        if stage == 'test' or stage is None:
            if self.test_embeddings_path:
                self.test_dataset = PrecomputedEmbeddingsDataset(self.test_embeddings_path)
    
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
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test dataloader."""
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return None
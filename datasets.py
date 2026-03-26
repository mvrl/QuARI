import os
import json
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
import numpy as np

"""
Efficient DataLoader for COCO WebDataset.
Expands image-caption pairs on the loading side.
"""

import webdataset as wds
import json
from PIL import Image
import io
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def expand_to_pairs(sample):
    """
    Generator that expands one sample (image + multiple captions)
    into multiple (image, caption) pairs.
    """
    # Decode image once
    img_bytes = sample['jpg']

    # Parse captions
    caption_data = json.loads(sample['json'].decode('utf-8'))
    captions = caption_data['captions']

    # Yield one pair for each caption
    for caption in captions:
        yield {
            'image': img_bytes,
            'caption': caption,
            'image_id': caption_data['image_id'],
            'key': sample['__key__']
        }


def create_pair_dataloader(
    dataset_pattern,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    image_size=224,
    expand_pairs=True,
    world_size=1,
    rank=0
):
    """
    Create DataLoader that yields individual (image, caption) pairs.

    Args:
        dataset_pattern: Path pattern to WebDataset shards (e.g., 'path/to/data-*.tar')
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        image_size: Size to resize images to
        expand_pairs: If True, yield (image, caption) pairs. If False, yield (image, all_captions)
        world_size: Total number of GPUs for distributed training
        rank: Current GPU rank for distributed training
    """

    if expand_pairs:
        # Expand each image to multiple (image, caption) pairs
        def process_sample(sample):
            """Process and transform a single pair."""
            # Decode image
            img = Image.open(io.BytesIO(sample['image'])).convert('RGB')
            return img, sample['caption']

        dataset = (
            wds.WebDataset(dataset_pattern, shardshuffle=shuffle)
            .shuffle(1000 if shuffle else 0)
            # Expand: this is the key operation that creates individual pairs
            .compose(lambda source: (pair for sample in source for pair in expand_to_pairs(sample)))
            .map(process_sample)
            .batched(batch_size)
        )
    else:
        # Keep image with all captions (original behavior)
        def process_sample_all(sample):
            img_bytes = sample['jpg']
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            caption_data = json.loads(sample['json'].decode('utf-8'))
            return img, caption_data['captions']

        dataset = (
            wds.WebDataset(dataset_pattern, shardshuffle=shuffle)
            .shuffle(1000 if shuffle else 0)
            .map(process_sample_all)
            .batched(batch_size)
        )

    # For distributed training
    if world_size > 1:
        dataset = dataset.with_epoch(1000000 // world_size)  # Large epoch size

    # Convert to PyTorch DataLoader
    loader = wds.WebLoader(
        dataset,
        batch_size=None,  # Batching is done in the dataset
        shuffle=False,  # Shuffling is done in the dataset
        num_workers=num_workers,
    )

    # Repeat for multiple epochs if needed
    if shuffle:
        loader = loader.repeat()

    return loader


def test_dataloader(dataset_path, expand_pairs=True):
    """Test the dataloader."""
    print(f"\nTesting DataLoader with expand_pairs={expand_pairs}")
    print("=" * 60)

    loader = create_pair_dataloader(
        dataset_path,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        expand_pairs=expand_pairs
    )

    # Get first few batches
    for i, batch in enumerate(loader):
        if i >= 2:
            break

        images, captions = batch

        print(f"\nBatch {i+1}:")
        print(f"  Images shape: {images.shape}")

        if expand_pairs:
            print(f"  Batch size: {len(captions)}")
            print(f"  Caption type: single caption per image")
            print(f"  Example captions:")
            for j, cap in enumerate(captions[:3]):
                print(f"    {j+1}. {cap}")
        else:
            print(f"  Batch size: {len(captions)}")
            print(f"  Caption type: list of captions per image")
            print(f"  Captions per image: {[len(c) for c in captions]}")
            print(f"  Example (first image captions):")
            for j, cap in enumerate(captions[0]):
                print(f"    {j+1}. {cap}")

    print("\n" + "=" * 60)


def count_total_pairs(dataset_path):
    """Count total number of pairs in the dataset."""
    print("Counting total pairs in dataset...")

    dataset = wds.WebDataset(dataset_path)

    total_images = 0
    total_pairs = 0

    for sample in dataset:
        total_images += 1
        caption_data = json.loads(sample['json'].decode('utf-8'))
        total_pairs += len(caption_data['captions'])

        if total_images >= 1000:  # Sample first 1000
            break

    # Extrapolate
    estimated_total_pairs = total_pairs
    estimated_total_images = total_images

    print(f"\nDataset Statistics (first {total_images} images):")
    print(f"  Total images: {estimated_total_images}")
    print(f"  Total pairs: {estimated_total_pairs}")
    print(f"  Avg captions per image: {total_pairs/total_images:.2f}")
    print(f"  Expansion ratio: {total_pairs/total_images:.2f}x")


def benchmark_loading(dataset_path, expand_pairs=True, num_batches=100):
    """Benchmark loading speed."""
    import time

    print(f"\nBenchmarking with expand_pairs={expand_pairs}")
    print(f"Loading {num_batches} batches...")

    loader = create_pair_dataloader(
        dataset_path,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        expand_pairs=expand_pairs
    )

    start_time = time.time()
    total_samples = 0

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        images, captions = batch
        total_samples += images.shape[0]

    elapsed = time.time() - start_time
    throughput = total_samples / elapsed

    print(f"  Processed {total_samples} samples in {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    print(f"  Time per batch: {elapsed/num_batches*1000:.2f}ms")


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
    
    Supports both:
    1. Single .pt file with all embeddings
    2. Directory with chunked embedding files (from compute_paired_embeddings.py)
    """
    
    def __init__(
        self,
        embeddings_path: str,
        transform: Optional[Callable] = None,
        pattern: str = "*.pt",
        lazy_load: bool = False,
        semipositives_path: Optional[str] = None
    ):
        """
        Initialize precomputed embeddings dataset.
        
        Args:
            embeddings_path: Path to embeddings file OR directory containing chunks
            transform: Optional transform to apply to items
            pattern: Glob pattern for chunk files (if embeddings_path is a directory)
            lazy_load: If True and loading from directory, use lazy loading (memory efficient)
            semipositives_path: Optional path to semi-positive indices and weights
        """
        self.transform = transform
        embeddings_path = Path(embeddings_path)
        
        # Check if path is a directory or a file
        if embeddings_path.is_dir():
            # Load from directory of chunks
            if lazy_load:
                # Use lazy-loading dataset
                from embedding_utils import ChunkedEmbeddingsDataset
                self._delegate = ChunkedEmbeddingsDataset(str(embeddings_path), pattern=pattern)
                self._is_delegate = True
            else:
                # Load all chunks into memory
                from embedding_utils import load_embedding_chunks
                self.data = load_embedding_chunks(str(embeddings_path), pattern=pattern)
                self._is_delegate = False
        else:
            # Load single file
            self.data = torch.load(embeddings_path)
            self._is_delegate = False
        
        if not self._is_delegate:
            # Validate data format
            required_keys = ["text_embeddings", "image_embeddings"]
            for key in required_keys:
                if key not in self.data:
                    raise ValueError(f"Missing required key '{key}' in precomputed embeddings")
            
            # Optional elements
            self.has_query_texts = "query_texts" in self.data
        
        self.semipositives_data = None
        if semipositives_path is not None:
            self.semipositives_data = torch.load(semipositives_path)
            print(f"Loaded semi-positives from {semipositives_path}")
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        if self._is_delegate:
            return len(self._delegate)
        return len(self.data["text_embeddings"])
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing embeddings
        """
        if self._is_delegate:
            item = self._delegate[idx]
        else:
            item = {
                "text_features": self.data["text_embeddings"][idx],
                "target_image_features": self.data["image_embeddings"][idx],
                "idx": idx
            }
            
            if self.has_query_texts:
                item["query_text"] = self.data["query_texts"][idx]
            
            if self.semipositives_data is not None:
                item["semipositive_embeddings"] = torch.from_numpy(
                    self.semipositives_data['semipositive_embeddings'][idx]
                ).float()
                item["semipositive_weights"] = torch.from_numpy(
                    self.semipositives_data['semipositive_weights'][idx]
                ).float()
        
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
        shuffle: bool = True,
        train_semipositives_path: Optional[str] = None,
        val_semipositives_path: Optional[str] = None
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
            train_semipositives_path: Path to training semi-positives file
            val_semipositives_path: Path to validation semi-positives file
        """
        super().__init__()
        self.train_embeddings_path = train_embeddings_path
        self.val_embeddings_path = val_embeddings_path
        self.test_embeddings_path = test_embeddings_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_semipositives_path = train_semipositives_path
        self.val_semipositives_path = val_semipositives_path
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup the data module for training, validation, or testing.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = PrecomputedEmbeddingsDataset(
                self.train_embeddings_path,
                semipositives_path=self.train_semipositives_path
            )
            
            if self.val_embeddings_path:
                self.val_dataset = PrecomputedEmbeddingsDataset(
                    self.val_embeddings_path,
                    semipositives_path=self.val_semipositives_path
                )
        
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

#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
from PIL import Image
import json
from typing import List, Tuple, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent / "data" / "quari"))

from models import PersonalizedRetrievalModule
from feature_extractors import FeatureExtractorFactory
from datasets import PrecomputedEmbeddingsDataset


def load_coco_captions(coco_annotations_path: str) -> Dict[str, List[str]]:
    """Load COCO captions from annotations file."""
    try:
        with open(coco_annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build mapping: image_id -> list of captions
        captions_map = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            caption = ann['caption']
            if img_id not in captions_map:
                captions_map[img_id] = []
            captions_map[img_id].append(caption)
        
        print(f"Loaded {len(captions_map)} images with captions")
        return captions_map
    except Exception as e:
        print(f"Warning: Could not load captions: {e}")
        return {}


def load_flickr30k_captions(flickr_captions_path: str) -> Dict[str, List[str]]:
    """Load Flickr30k captions from captions.txt file."""
    import csv
    try:
        captions_map = {}
        
        with open(flickr_captions_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                if len(row) < 2:
                    continue
                    
                image_filename = row[0].strip()
                caption = row[1].strip()
                
                # Extract image ID (remove .jpg extension)
                image_id = image_filename.replace('.jpg', '')
                
                if image_id not in captions_map:
                    captions_map[image_id] = []
                captions_map[image_id].append(caption)
        
        print(f"Loaded {len(captions_map)} images with captions from Flickr30k")
        return captions_map
    except Exception as e:
        print(f"Warning: Could not load Flickr30k captions: {e}")
        return {}


def get_caption_from_key(key: str, captions_map: Dict, dataset_type: str = 'coco') -> str:
    """Extract caption from key using captions map."""
    import re
    
    # Get caption index
    cap_idx = 0
    if '_cap' in key:
        try:
            cap_idx = int(key.split('_cap')[-1])
        except:
            cap_idx = 0
    
    if dataset_type == 'flickr30k':
        # Flickr30k keys are simple numeric IDs: "1000092795" or "1000092795_cap2"
        # Remove caption suffix to get image ID
        image_id = key.split('_cap')[0]
        
        # Look up caption (captions_map uses string keys for Flickr30k)
        if image_id in captions_map and cap_idx < len(captions_map[image_id]):
            caption = captions_map[image_id][cap_idx]
            # Truncate if too long
            if len(caption) > 80:
                caption = caption[:77] + "..."
            return caption
    else:
        # COCO: Extract numeric image ID
        # Format: "train_000000057395_cap2" → image_id=57395, cap_idx=2
        numeric_match = re.search(r'(\d+)', key)
        if not numeric_match:
            return key
        
        image_id = int(numeric_match.group(1))
        
        # Look up caption (captions_map uses integer keys for COCO)
        if image_id in captions_map and cap_idx < len(captions_map[image_id]):
            caption = captions_map[image_id][cap_idx]
            # Truncate if too long
            if len(caption) > 80:
                caption = caption[:77] + "..."
            return caption
    
    return key  # Fallback to key if caption not found


def load_model_and_data(checkpoint_path: str, embedding_dir: str, backbone: str, n_samples: int = 10000):
    """Load trained model and embeddings."""
    print("Loading checkpoint...")
    
    # Create a dummy feature extractor (needed for model instantiation)
    dummy_extractor = FeatureExtractorFactory.create_extractor(
        model_name=backbone,
        device="cpu"  # Keep on CPU to save GPU memory
    )
    
    # Load checkpoint with feature extractor
    model = PersonalizedRetrievalModule.load_from_checkpoint(
        checkpoint_path,
        feature_extractor=dummy_extractor,
        map_location='cuda'
    )
    model.eval()
    model = model.cuda()
    
    print("Loading embeddings...")
    dataset = PrecomputedEmbeddingsDataset(
        embedding_dir,
        lazy_load=False,
        pattern="*.pt"
    )
    
    # Sample a subset if dataset is large
    if len(dataset) > n_samples:
        indices = torch.randperm(len(dataset))[:n_samples]
        text_embeds = dataset.data["text_embeddings"][indices]
        image_embeds = dataset.data["image_embeddings"][indices]
        keys = [dataset.data["keys"][i] for i in indices] if "keys" in dataset.data else None
    else:
        text_embeds = dataset.data["text_embeddings"]
        image_embeds = dataset.data["image_embeddings"]
        keys = dataset.data.get("keys", None)
    
    # Convert embeddings to fp32 to match model dtype
    text_embeds = text_embeds.float()
    image_embeds = image_embeds.float()
    
    print(f"Loaded {len(text_embeds):,} pairs")
    print(f"Embedding dtype: {text_embeds.dtype}")
    print(f"Embedding dim: {text_embeds.shape[1]}")
    print(f"Model expects dim: {model.embedding_dim}")
    
    return model, text_embeds, image_embeds, keys


def compute_retrieval_metrics(
    query_embeds: torch.Tensor,
    gallery_embeds: torch.Tensor,
    k: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k retrieval for each query.
    
    Returns:
        similarities: [N, k] top-k similarity scores
        indices: [N, k] top-k gallery indices
    """
    # Compute similarity matrix
    sim_matrix = torch.matmul(query_embeds, gallery_embeds.t())  # [N, N]
    
    # Get top-k for each query
    similarities, indices = torch.topk(sim_matrix, k=k, dim=1)
    
    return similarities, indices


def compare_retrieval(
    model,
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    batch_size: int = 256,
    k: int = 10
):
    """
    Compare retrieval before and after transformation.
    
    Returns:
        original_results: (similarities, indices) with original embeddings
        transformed_results: (similarities, indices) with transformed embeddings
    """
    device = next(model.parameters()).device
    n_samples = len(text_embeds)
    
    # Store results
    all_refined_queries = []
    all_transformed_images = []
    
    # Process in batches
    print(f"Processing {n_samples} samples in batches of {batch_size}...")
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_text = text_embeds[i:i+batch_size].to(device)
            batch_image = image_embeds[i:i+batch_size].to(device)
            
            # Create batch dict
            batch = {
                "text_features": batch_text,
                "target_image_features": batch_image
            }
            
            # Forward pass through hypernetwork
            outputs = model.forward(batch)
            
            all_refined_queries.append(outputs["refined_query"].cpu())
            all_transformed_images.append(outputs["transformed_images"].cpu())
    
    # Concatenate
    refined_queries = torch.cat(all_refined_queries, dim=0)
    transformed_images = torch.cat(all_transformed_images, dim=0)
    
    print("Computing retrieval with original embeddings...")
    original_sims, original_indices = compute_retrieval_metrics(
        text_embeds, image_embeds, k=k
    )
    
    print("Computing retrieval with transformed embeddings...")
    transformed_sims, transformed_indices = compute_retrieval_metrics(
        refined_queries, transformed_images, k=k
    )
    
    return (original_sims, original_indices), (transformed_sims, transformed_indices), refined_queries, transformed_images


def compute_recall_at_k(indices: torch.Tensor, k: int = 10) -> float:
    """
    Compute Recall@K (percentage of queries where correct image is in top-k).
    Assumes diagonal is correct (query i matches image i).
    """
    n_queries = indices.shape[0]
    correct_indices = torch.arange(n_queries).unsqueeze(1)  # [N, 1]
    
    # Check if correct index is in top-k
    recall = (indices[:, :k] == correct_indices).any(dim=1).float().mean()
    
    return recall.item()


def plot_retrieval_comparison(
    original_results,
    transformed_results,
    save_path: str
):
    """Plot retrieval metrics comparison."""
    original_sims, original_indices = original_results
    transformed_sims, transformed_indices = transformed_results
    
    # Compute recall@k for different k
    k_values = [1, 5, 10, 20, 50]
    original_recalls = [compute_recall_at_k(original_indices, k) for k in k_values]
    transformed_recalls = [compute_recall_at_k(transformed_indices, k) for k in k_values]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Recall@K
    axes[0].plot(k_values, original_recalls, 'o-', label='Original CLIP', linewidth=2, markersize=8)
    axes[0].plot(k_values, transformed_recalls, 's-', label='Transformed', linewidth=2, markersize=8)
    axes[0].set_xlabel('K', fontsize=12)
    axes[0].set_ylabel('Recall@K', fontsize=12)
    axes[0].set_title('Retrieval Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1.0])
    
    # Add improvement annotations
    for i, k in enumerate(k_values):
        improvement = (transformed_recalls[i] - original_recalls[i]) * 100
        if improvement > 0:
            axes[0].annotate(f'+{improvement:.1f}%', 
                           xy=(k, transformed_recalls[i]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, color='green')
    
    # Plot 2: Similarity score distribution
    axes[1].hist(original_sims[:, 0].numpy(), bins=50, alpha=0.6, label='Original (top-1)', color='blue')
    axes[1].hist(transformed_sims[:, 0].numpy(), bins=50, alpha=0.6, label='Transformed (top-1)', color='orange')
    axes[1].axvline(original_sims[:, 0].mean(), color='blue', linestyle='--', 
                   label=f'Original mean: {original_sims[:, 0].mean():.3f}')
    axes[1].axvline(transformed_sims[:, 0].mean(), color='orange', linestyle='--',
                   label=f'Transformed mean: {transformed_sims[:, 0].mean():.3f}')
    axes[1].set_xlabel('Similarity Score', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Top-1 Similarity Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved retrieval comparison: {save_path}")
    plt.close()
    
    # Print numerical results
    print("\n" + "="*80)
    print("RETRIEVAL METRICS")
    print("="*80)
    print(f"{'K':<10} {'Original':<15} {'Transformed':<15} {'Improvement':<15}")
    print("-"*80)
    for k, orig, trans in zip(k_values, original_recalls, transformed_recalls):
        improvement = (trans - orig) * 100
        print(f"{k:<10} {orig*100:>6.2f}%{'':<7} {trans*100:>6.2f}%{'':<7} {improvement:>+6.2f}%")
    print("="*80)


def format_key_display(key: str, max_length: int = 40) -> str:
    """Format key for display - extract meaningful info."""
    # Remove caption index suffix
    display = key.split('_cap')[0]
    
    # If it's a COCO key like "train_000000057395"
    # Extract just the image ID
    if 'train_' in display or 'val_' in display:
        parts = display.split('_')
        if len(parts) >= 2:
            display = f"COCO ID: {parts[-1]}"
    
    # Truncate if too long
    if len(display) > max_length:
        display = display[:max_length-3] + "..."
    
    return display


def plot_example_retrievals(
    query_idx: int,
    original_indices: torch.Tensor,
    transformed_indices: torch.Tensor,
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    keys: List[str],
    coco_images_dir: str,
    captions_map: Dict,
    save_path: str,
    k: int = 5,
    dataset_type: str = 'coco'
):
    """
    Visualize top-k retrievals for a specific query.
    Shows original CLIP retrieval vs transformed retrieval side-by-side.
    Supports both COCO and Flickr30k datasets.
    """
    if coco_images_dir is None:
        print("Skipping image visualization (no image directory provided)")
        return
    
    images_path = Path(coco_images_dir)
    
    # Select appropriate image finder function
    if dataset_type == 'flickr30k':
        find_image_func = find_flickr30k_image
        dataset_name = "Flickr30k"
    else:
        find_image_func = find_coco_image
        dataset_name = "COCO"
    
    # Debug: print first query key to help diagnose
    if query_idx == 0:
        query_key = keys[query_idx] if keys else f"query_{query_idx}"
        print(f"\nDebug info for image loading:")
        print(f"  {dataset_name} directory: {images_path}")
        print(f"  Sample key: {query_key}")
        test_path = find_image_func(images_path, query_key)
        if test_path:
            print(f"  ✓ Found image at: {test_path}")
        else:
            print(f"  ✗ Image not found for key: {query_key}")
    
    # Get top-k indices
    orig_k = original_indices[query_idx, :k]
    trans_k = transformed_indices[query_idx, :k]
    
    # Get caption or formatted key for display
    query_key_raw = keys[query_idx] if keys else f"query_{query_idx}"
    
    if captions_map:
        query_text = get_caption_from_key(query_key_raw, captions_map, dataset_type)
    else:
        query_text = format_key_display(query_key_raw)
    
    fig, axes = plt.subplots(2, k+1, figsize=(3*(k+1), 6))
    
    # Query info (first column) - show caption text
    axes[0, 0].text(0.5, 0.7, f"Query", 
                   ha='center', va='center', fontsize=11, fontweight='bold')
    axes[0, 0].text(0.5, 0.4, f'"{query_text}"', 
                   ha='center', va='center', fontsize=8, wrap=True, style='italic')
    axes[0, 0].text(0.5, 0.1, f"Original CLIP", 
                   ha='center', va='center', fontsize=8, color='gray')
    axes[0, 0].axis('off')
    
    axes[1, 0].text(0.5, 0.7, f"Query", 
                   ha='center', va='center', fontsize=11, fontweight='bold')
    axes[1, 0].text(0.5, 0.4, f'"{query_text}"', 
                   ha='center', va='center', fontsize=8, wrap=True, style='italic')
    axes[1, 0].text(0.5, 0.1, f"Transformed", 
                   ha='center', va='center', fontsize=8, color='gray')
    axes[1, 0].axis('off')
    
    # Original retrievals (row 0)
    for i, idx in enumerate(orig_k):
        img_key = keys[idx] if keys else f"{idx}"
        
        # Try to load image
        img_path = find_image_func(images_path, img_key)
        
        if img_path and img_path.exists():
            img = Image.open(img_path).convert('RGB')
            axes[0, i+1].imshow(img)
        else:
            axes[0, i+1].text(0.5, 0.5, 'Image\nnot found', ha='center', va='center')
        
        # Check if this is correct (diagonal)
        is_correct = (idx == query_idx)
        color = 'green' if is_correct else 'black'
        axes[0, i+1].set_title(f"#{i+1}" + (" ✓" if is_correct else ""), 
                              fontsize=10, color=color, fontweight='bold' if is_correct else 'normal')
        axes[0, i+1].axis('off')
    
    # Transformed retrievals (row 1)
    for i, idx in enumerate(trans_k):
        img_key = keys[idx] if keys else f"{idx}"
        
        # Try to load image
        img_path = find_image_func(images_path, img_key)
        
        if img_path and img_path.exists():
            img = Image.open(img_path).convert('RGB')
            axes[1, i+1].imshow(img)
        else:
            axes[1, i+1].text(0.5, 0.5, 'Image\nnot found', ha='center', va='center')
        
        # Check if this is correct
        is_correct = (idx == query_idx)
        color = 'green' if is_correct else 'black'
        axes[1, i+1].set_title(f"#{i+1}" + (" ✓" if is_correct else ""), 
                              fontsize=10, color=color, fontweight='bold' if is_correct else 'normal')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved example retrieval: {save_path}")
    plt.close()


def find_coco_image(coco_dir: Path, key: str) -> Path:
    """
    Find COCO image file from key.
    Handles different naming conventions.
    """
    # Try different patterns
    # Remove _cap suffix if present (from caption expansion)
    base_key = key.split('_cap')[0]
    
    # COCO images are named like: 000000000009.jpg (12 digits)
    # Keys might be: "000000000009" or "9" or "coco_train-000000-000009"
    
    # Extract numeric ID
    import re
    numeric_match = re.search(r'(\d+)', base_key)
    if numeric_match:
        numeric_id = numeric_match.group(1)
    else:
        return None
    
    # Try different filename patterns
    patterns = [
        f"{numeric_id.zfill(12)}.jpg",  # 000000000009.jpg
        f"{numeric_id}.jpg",             # 9.jpg
        f"{base_key}.jpg",               # Original key
        f"COCO_train2017_{numeric_id.zfill(12)}.jpg",
        f"COCO_val2017_{numeric_id.zfill(12)}.jpg",
    ]
    
    # Try different subdirectories
    subdirs = ['train2017', 'val2017', 'train2014', 'val2014', 'test2017', '']
    
    for pattern in patterns:
        for subdir in subdirs:
            if subdir:
                img_path = coco_dir / subdir / pattern
            else:
                img_path = coco_dir / pattern
            
            if img_path.exists():
                return img_path
    
    return None


def find_flickr30k_image(flickr_dir: Path, key: str) -> Path:
    """
    Find Flickr30k image file from key.
    Flickr30k keys are simple numeric IDs like "1000092795".
    """
    # Remove _cap suffix if present (from caption expansion)
    base_key = key.split('_cap')[0]
    
    # Flickr30k images are named like: 1000092795.jpg
    filename = f"{base_key}.jpg"
    
    # Try different subdirectories
    # Images are typically in flickr30k_images/flickr30k_images/
    subdirs = [
        'flickr30k_images/flickr30k_images',
        'flickr30k_images',
        'images',
        ''
    ]
    
    for subdir in subdirs:
        if subdir:
            img_path = flickr_dir / subdir / filename
        else:
            img_path = flickr_dir / filename
        
        if img_path.exists():
            return img_path
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize retrieval before/after transformation")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--embedding_dir", type=str, required=True,
                       help="Directory with embeddings to visualize")
    parser.add_argument("--backbone", type=str, default="openai/clip-vit-base-patch16",
                       help="Backbone model name (must match training)")
    parser.add_argument("--dataset_type", type=str, default="auto",
                       choices=["auto", "coco", "flickr30k"],
                       help="Dataset type (auto-detect from embedding_dir if not specified)")
    
    # COCO-specific arguments
    parser.add_argument("--coco_images", type=str, default=None,
                       help="Path to COCO images directory (for visualization)")
    parser.add_argument("--coco_annotations", type=str, default=None,
                       help="Path to COCO annotations JSON (e.g., captions_train2017.json)")
    
    # Flickr30k-specific arguments
    parser.add_argument("--flickr30k_images", type=str, default=None,
                       help="Path to Flickr30k images directory (for visualization)")
    parser.add_argument("--flickr30k_captions", type=str, default=None,
                       help="Path to Flickr30k captions.txt file")
    
    parser.add_argument("--n_samples", type=int, default=10000,
                       help="Number of samples to use (default: 10000)")
    parser.add_argument("--n_queries", type=int, default=20,
                       help="Number of example queries to visualize (default: 20)")
    parser.add_argument("--k", type=int, default=10,
                       help="Top-k for retrieval (default: 10)")
    
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                       help="Output directory for plots")
    parser.add_argument("--projection_method", type=str, default="tsne",
                       choices=["tsne", "umap"],
                       help="Method for 2D projection")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Auto-detect dataset type if not specified
    if args.dataset_type == "auto":
        if "flickr" in args.embedding_dir.lower():
            args.dataset_type = "flickr30k"
        else:
            args.dataset_type = "coco"
        print(f"Auto-detected dataset type: {args.dataset_type}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("RETRIEVAL VISUALIZATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Embeddings: {args.embedding_dir}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    # Load model and data
    model, text_embeds, image_embeds, keys = load_model_and_data(
        args.checkpoint, args.embedding_dir, args.backbone, args.n_samples
    )
    
    print(f"Loaded {len(text_embeds):,} embedding pairs")
    
    # Load captions based on dataset type
    captions_map = {}
    images_dir = None
    
    if args.dataset_type == "flickr30k":
        # Handle Flickr30k
        images_dir = args.flickr30k_images
        
        # Try to auto-locate captions file
        if args.flickr30k_captions:
            print(f"Loading Flickr30k captions from: {args.flickr30k_captions}")
            captions_map = load_flickr30k_captions(args.flickr30k_captions)
        elif images_dir:
            # Try to find captions.txt in Flickr30k directory
            flickr_dir = Path(images_dir)
            for captions_file in ['captions.txt', 
                                 'flickr30k_images/captions.txt',
                                 '../captions.txt']:
                cap_path = flickr_dir / captions_file
                if cap_path.exists():
                    print(f"Found Flickr30k captions: {cap_path}")
                    captions_map = load_flickr30k_captions(str(cap_path))
                    break
        
        # Auto-locate images directory if not specified
        if not images_dir:
            # Try common locations
            for base_dir in ['./data/flickr30k', '../data/flickr30k', '/home/exing/repr_learning/data/flickr30k']:
                if Path(base_dir).exists():
                    images_dir = base_dir
                    print(f"Auto-located Flickr30k images: {images_dir}")
                    break
    else:
        # Handle COCO
        images_dir = args.coco_images
        
        if args.coco_annotations:
            print(f"Loading COCO captions from: {args.coco_annotations}")
            captions_map = load_coco_captions(args.coco_annotations)
        elif args.coco_images:
            # Try to find annotations in COCO directory
            coco_dir = Path(args.coco_images)
            for ann_file in ['annotations/captions_train2017.json', 
                            'annotations/captions_val2017.json',
                            'captions_train2017.json',
                            'captions_val2017.json']:
                ann_path = coco_dir / ann_file
                if ann_path.exists():
                    print(f"Found COCO annotations: {ann_path}")
                    captions_map = load_coco_captions(str(ann_path))
                    break
    
    # Compare retrieval
    (orig_sims, orig_indices), (trans_sims, trans_indices), refined_queries, transformed_images = compare_retrieval(
        model, text_embeds, image_embeds, args.batch_size, args.k
    )
    
    # Plot retrieval comparison
    plot_retrieval_comparison(
        (orig_sims, orig_indices),
        (trans_sims, trans_indices),
        output_dir / "retrieval_comparison.png"
    )
    

    # Visualize example retrievals with images
    if images_dir:
        print(f"\nGenerating {args.n_queries} example retrieval visualizations...")
        for i in range(min(args.n_queries, len(text_embeds))):
            plot_example_retrievals(
                query_idx=i,
                original_indices=orig_indices,
                transformed_indices=trans_indices,
                text_embeds=text_embeds,
                image_embeds=image_embeds,
                keys=keys,
                coco_images_dir=images_dir,
                captions_map=captions_map,
                save_path=output_dir / f"retrieval_example_{i:03d}.png",
                k=args.k,
                dataset_type=args.dataset_type
            )
            if (i+1) % 5 == 0:
                print(f"  Generated {i+1}/{args.n_queries} examples")

if __name__ == "__main__":
    main()


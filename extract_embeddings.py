import os
import argparse
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Any

from feature_extractors import FeatureExtractorFactory
from datasets import SimpleTextImageDataset
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract and save embeddings for later use")
    
    # Dataset arguments
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON file with text-image pairs")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--split", type=str, required=True, help="Data split name (e.g., train, val, test)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloaders")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    
    # Feature extractor arguments
    parser.add_argument("--extractor_type", type=str, default="clip", choices=["clip", "siglip", "siglip2"], 
                        help="Type of feature extractor")
    parser.add_argument("--extractor_model", type=str, default="ViT-B/32", 
                        help="Model name for feature extractor")
    
    return parser.parse_args()


def main():
    """Main function for extracting and saving embeddings."""
    args = parse_args()
    
    # Create output directory named after the model
    output_dir = f"{args.extractor_type}_{args.extractor_model.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create feature extractor
    feature_extractor = FeatureExtractorFactory.create_extractor(
        extractor_type=args.extractor_type,
        model_name=args.extractor_model,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load dataset for the specified split
    dataset = SimpleTextImageDataset(
        json_path=args.json_path,
        image_dir=args.image_dir,
        max_samples=args.max_samples
    )
    
    print(f"\nProcessing {args.split} split ({len(dataset)} samples)...")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Collect all data for this split
    all_query_texts = []
    all_text_embeddings = []
    all_image_embeddings = []
    
    for batch in tqdm(dataloader, desc=f"Extracting embeddings for {args.split} data"):
        all_query_texts.extend(batch["query_text"])
        
        # Extract text embeddings
        with torch.no_grad():
            text_embs = feature_extractor.extract_text_features(batch["query_text"])
            all_text_embeddings.append(text_embs)
        
        # Extract image embeddings
        with torch.no_grad():
            image_embs = feature_extractor.extract_image_features(batch["query_image"])
            all_image_embeddings.append(image_embs)
    
    # Concatenate embeddings
    all_text_embeddings = torch.cat(all_text_embeddings)
    all_image_embeddings = torch.cat(all_image_embeddings)
    
    # Save embeddings
    save_path = os.path.join(output_dir, f"{args.split}_embeddings.pt")
    
    torch.save({
        'text_embeddings': all_text_embeddings,
        'image_embeddings': all_image_embeddings,
        'query_texts': all_query_texts
    }, save_path)
    
    print(f"Saved {args.split} embeddings to {save_path}")
    print(f"Processed {len(all_query_texts)} samples")
    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
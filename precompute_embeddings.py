"""
Precompute embeddings from a dataset for efficient training.
"""
import argparse
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from feature_extractors import FeatureExtractorFactory
from datasets import SimpleTextImageDataset


def precompute_embeddings(
    extractor,
    json_path,
    image_dir,
    output_path,
    batch_size=256,
    num_workers=10,
    max_samples=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)
    
    dataset = SimpleTextImageDataset(
        json_path=json_path,
        image_dir=image_dir,
        max_samples=max_samples,
        resize_resolution=extractor.input_resolution
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    text_embeddings = []
    image_embeddings = []
    query_texts = []
    
    print(f"Extracting features from {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader):
            texts = batch["query_text"]
            images = batch["target_image"]
            
            text_feats = extractor.extract_text_features(texts)
            image_feats = extractor.extract_image_features(images)
            
            text_embeddings.append(text_feats.cpu())
            image_embeddings.append(image_feats.cpu())
            query_texts.extend(texts)
    
    text_embeddings = torch.cat(text_embeddings, dim=0)
    image_embeddings = torch.cat(image_embeddings, dim=0)
    
    data = {
        "text_embeddings": text_embeddings,
        "image_embeddings": image_embeddings,
        "query_texts": query_texts
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print(f"Saved {len(query_texts)} embeddings to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", required=True, help="Model name for feature extraction")
    parser.add_argument("--json_path", required=True, help="Path to dataset JSON")
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_path", required=True, help="Path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()
    
    extractor = FeatureExtractorFactory.create_extractor(
        model_name=args.extractor,
        device=args.device
    )
    
    precompute_embeddings(
        extractor=extractor,
        json_path=args.json_path,
        image_dir=args.image_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()


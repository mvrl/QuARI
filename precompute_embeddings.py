"""
Precompute embeddings from a WebDataset of (image, caption) pairs.
"""

import argparse
import os
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm

from feature_extractors import FeatureExtractorFactory
from datasets import create_pair_dataloader


def build_shard_list(tar_dir: str, tar_regex: str):
    """
    Build a sorted list of shard paths in tar_dir whose filenames match tar_regex.
    """
    tar_dir = Path(tar_dir)
    pattern = re.compile(tar_regex)

    shard_paths = []
    for p in sorted(tar_dir.iterdir()):
        if not p.is_file():
            continue
        if pattern.search(p.name):
            shard_paths.append(str(p))

    if not shard_paths:
        raise ValueError(
            f"No tarfiles in '{tar_dir}' matched regex '{tar_regex}'."
        )

    return shard_paths


def precompute_embeddings(
    extractor,
    json_path,
    image_dir,
    output_path,
    batch_size=256,
    num_workers=10,
    max_samples=None,
    tar_regex=r".*\.tar$",
):
    """
    Precompute embeddings from a WebDataset of image-caption pairs.

    Args:
        extractor: Feature extractor with .extract_text_features and .extract_image_features
        json_path: Kept for backward compatibility (unused here)
        image_dir: Directory containing WebDataset shard tarfiles
        output_path: Destination .pt file
        batch_size: Batch size over (image, caption) pairs
        num_workers: DataLoader workers
        max_samples: Optional cap on number of pairs to process
        tar_regex: Regex to select shard tarfiles inside image_dir
    """
    # Choose device as in original script
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)

    # Build explicit list of shards using regex
    shard_urls = build_shard_list(image_dir, tar_regex)

    # Create a WebDataset-based loader that yields (images, captions)
    loader = create_pair_dataloader(
        dataset_pattern=shard_urls,
        batch_size=batch_size,
        shuffle=False,  # don't repeat / infinite-loop when precomputing
        num_workers=num_workers,
        image_size=extractor.input_resolution,
        expand_pairs=True,
    )

    text_embeddings = []
    image_embeddings = []
    query_texts = []

    # tqdm total is known if max_samples is set; otherwise unknown
    pbar = tqdm(total=max_samples, desc="Extracting features", dynamic_ncols=True)

    processed = 0
    with torch.no_grad():
        for images, captions in loader:
            # captions is typically a list of strings
            texts = list(captions)

            batch_size_curr = len(texts)

            # If max_samples is set, maybe need to trim this batch
            if max_samples is not None:
                remaining = max_samples - processed
                if remaining <= 0:
                    break
                if batch_size_curr > remaining:
                    images = images[:remaining]
                    texts = texts[:remaining]
                    batch_size_curr = remaining

            # Extract features
            text_feats = extractor.extract_text_features(texts)
            image_feats = extractor.extract_image_features(images)

            text_embeddings.append(text_feats.cpu())
            image_embeddings.append(image_feats.cpu())
            query_texts.extend(texts)

            processed += batch_size_curr
            pbar.update(batch_size_curr)

            if max_samples is not None and processed >= max_samples:
                break

    pbar.close()

    if not text_embeddings:
        raise RuntimeError("No samples were processed; check your tar_regex and shards.")

    text_embeddings = torch.cat(text_embeddings, dim=0)
    image_embeddings = torch.cat(image_embeddings, dim=0)

    data = {
        "text_embeddings": text_embeddings,
        "image_embeddings": image_embeddings,
        "query_texts": query_texts,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print(f"Saved {len(query_texts)} embeddings to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", required=True, help="Model name for feature extraction")
    parser.add_argument("--json_path", required=True, help="legacy")
    parser.add_argument("--image_dir", required=True, help="Directory containing WebDataset shard tarfiles")
    parser.add_argument("--output_path", required=True, help="Path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers")
    parser.add_argument("--max_samples", type=int, default=None, help="Max (image, caption) pairs to process")
    parser.add_argument(
        "--tar_regex",
        required=True,
        help="Regex to match tarfiles inside image_dir (e.g. 'train-.*\\\\.tar$')",
    )
    parser.add_argument("--device", default="cuda", help="Device to use (passed to FeatureExtractorFactory)")
    args = parser.parse_args()

    extractor = FeatureExtractorFactory.create_extractor(
        model_name=args.extractor,
        device=args.device,
    )

    precompute_embeddings(
        extractor=extractor,
        json_path=args.json_path,
        image_dir=args.image_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        tar_regex=args.tar_regex,
    )


if __name__ == "__main__":
    main()

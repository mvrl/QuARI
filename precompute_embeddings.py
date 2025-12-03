"""
Precompute embeddings from a WebDataset of (image, caption) pairs.
Each file (single or chunk) has:
    {
        "text_embeddings": [N, D_txt],
        "image_embeddings": [N, D_img],
        "query_texts": list[str] length N
    }
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


def _save_single_file(output_path: str, text_embeddings, image_embeddings, query_texts):
    data = {
        "text_embeddings": text_embeddings,
        "image_embeddings": image_embeddings,
        "query_texts": query_texts,
    }
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data, output_path)
    print(f"Saved {len(query_texts)} embeddings to {output_path}")
    return output_path


def _save_chunk(base_dir: Path, chunk_idx: int, text_buf, image_buf, text_str_buf):
    text_cat = torch.cat(text_buf, dim=0)
    image_cat = torch.cat(image_buf, dim=0)
    data = {
        "text_embeddings": text_cat,
        "image_embeddings": image_cat,
        "query_texts": text_str_buf,
    }
    chunk_path = base_dir / f"chunk_{chunk_idx:05d}.pt"
    torch.save(data, chunk_path)
    print(f"  Saved chunk {chunk_idx:05d} with {len(text_str_buf)} samples to {chunk_path}")
    return len(text_str_buf)


def precompute_embeddings(
    extractor,
    image_dir,
    output_path,
    batch_size=256,
    num_workers=10,
    max_samples=None,
    tar_regex=r".*\.tar$",
    chunk_size=None,
):
    """
    Precompute embeddings from a WebDataset of image-caption pairs.

    Args:
        extractor: Feature extractor with .extract_text_features and .extract_image_features
        image_dir: Directory containing WebDataset shard tarfiles
        output_path: If chunk_size is None -> single .pt file.
                     If chunk_size is set -> directory to hold chunk_XXXXX.pt files.
        batch_size: Batch size over (image, caption) pairs
        num_workers: DataLoader workers
        max_samples: Optional cap on number of pairs to process
        tar_regex: Regex to select shard tarfiles inside image_dir
        chunk_size: If set, maximum number of samples per chunk file.
    """
    # Match original behavior: shove extractor to CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)

    # Build explicit list of shards using regex
    shard_urls = build_shard_list(image_dir, tar_regex)

    # Create a WebDataset-based loader that yields (images, captions)
    loader = create_pair_dataloader(
        dataset_pattern=shard_urls,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        image_size=extractor.input_resolution,
        expand_pairs=True,
    )

    # SINGLE-FILE MODE (no chunking)
    if chunk_size is None:
        text_embeddings = []
        image_embeddings = []
        query_texts = []

        processed = 0
        pbar = tqdm(desc="Extracting features", dynamic_ncols=True)

        with torch.no_grad():
            for images, captions in loader:
                texts = list(captions)
                bs = len(texts)

                if max_samples is not None:
                    remaining = max_samples - processed
                    if remaining <= 0:
                        break
                    if bs > remaining:
                        images = images[:remaining]
                        texts = texts[:remaining]
                        bs = remaining

                text_feats = extractor.extract_text_features(texts)
                image_feats = extractor.extract_image_features(images)

                text_embeddings.append(text_feats.cpu())
                image_embeddings.append(image_feats.cpu())
                query_texts.extend(texts)

                processed += bs
                pbar.update(bs)

                if max_samples is not None and processed >= max_samples:
                    break

        pbar.close()

        if not text_embeddings:
            raise RuntimeError("No samples were processed; check your tar_regex and shards.")

        text_embeddings = torch.cat(text_embeddings, dim=0)
        image_embeddings = torch.cat(image_embeddings, dim=0)

        return _save_single_file(output_path, text_embeddings, image_embeddings, query_texts)
        
    base_dir = Path(output_path)
    base_dir.mkdir(parents=True, exist_ok=True)

    text_buf = []
    image_buf = []
    text_str_buf = []

    processed = 0
    total_saved = 0
    chunk_idx = 0

    pbar = tqdm(desc="Extracting features (chunked)", dynamic_ncols=True)

    with torch.no_grad():
        for images, captions in loader:
            texts = list(captions)
            bs = len(texts)

            if max_samples is not None:
                remaining = max_samples - processed
                if remaining <= 0:
                    break
                if bs > remaining:
                    images = images[:remaining]
                    texts = texts[:remaining]
                    bs = remaining

            # If adding this batch would overflow the desired chunk_size,
            # flush the current buffer first (if it's non-empty).
            if text_str_buf and len(text_str_buf) + bs > chunk_size:
                saved = _save_chunk(base_dir, chunk_idx, text_buf, image_buf, text_str_buf)
                total_saved += saved
                chunk_idx += 1

                text_buf = []
                image_buf = []
                text_str_buf = []

            # Now add this batch to the buffer
            text_feats = extractor.extract_text_features(texts)
            image_feats = extractor.extract_image_features(images)

            text_buf.append(text_feats.cpu())
            image_buf.append(image_feats.cpu())
            text_str_buf.extend(texts)

            processed += bs
            pbar.update(bs)

            if max_samples is not None and processed >= max_samples:
                break

    # Flush tail
    if text_str_buf:
        saved = _save_chunk(base_dir, chunk_idx, text_buf, image_buf, text_str_buf)
        total_saved += saved
        chunk_idx += 1

    pbar.close()

    if total_saved == 0:
        raise RuntimeError("No samples were processed; check your tar_regex and shards.")

    print(f"Saved total of {total_saved} embeddings in {chunk_idx} chunks to {base_dir}")
    return str(base_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", required=True, help="Model name for feature extraction")
    parser.add_argument("--image_dir", required=True, help="Directory containing WebDataset shard tarfiles")
    parser.add_argument("--output_path", required=True, help="Path to save embeddings (file or directory)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers")
    parser.add_argument("--max_samples", type=int, default=None, help="Max (image, caption) pairs to process")
    parser.add_argument(
        "--tar_regex",
        required=True,
        help="Regex to match tarfiles inside image_dir (e.g. 'train-.*\\\\.tar$')",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="If set, write chunked output with at most this many samples per chunk to a directory at --output_path",
    )
    parser.add_argument("--device", default="cuda", help="Device to use (passed to FeatureExtractorFactory)")
    args = parser.parse_args()

    extractor = FeatureExtractorFactory.create_extractor(
        model_name=args.extractor,
        device=args.device,
    )

    precompute_embeddings(
        extractor=extractor,
        image_dir=args.image_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        tar_regex=args.tar_regex,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()

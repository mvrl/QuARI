#!/usr/bin/env python
"""
Download the ILIAS evaluation dataset from Hugging Face.

ILIAS (Identification-focused Large-scale Image Attribute Search) is a
fine-grained retrieval benchmark that pairs text queries with target images
drawn from a 15 M-image YFCC subset.

What this script downloads
--------------------------
* **ILIAS-core** — the ~1 000 query images and ground-truth annotations,
  stored as WebDataset ``.tar`` shards.  After download the shards can be
  passed directly to ``precompute_embeddings.py``.

What you need to obtain separately
-----------------------------------
* **YFCC15M distractors** — a 15-million-image subset of Yahoo's YFCC100M
  dataset.  These are large (~2 TB) and must be requested from the
  YFCC100M official download page:
      https://multimediacommons.wordpress.com/yfcc100m-core-dataset/
  Once you have the YFCC `.tar` shards, embed them with ``compute_embeds.py``
  and pass the output directory to ``eval_retrieval.py --distractor_dirs``.

Usage
-----
    python downloading/download_ilias.py --local-dir ./data/ilias

After downloading, run the evaluation pipeline::

    # 1. Compute paired (query-text, target-image) embeddings
    python precompute_embeddings.py \\
        --extractor google/siglip2-large-patch16-512 \\
        --image_dir ./data/ilias/ilias-core \\
        --output_path ./embeds/ilias_pairs.pt \\
        --tar_regex '.*\\.tar$'

    # 2. Evaluate (add --distractor_dirs if you have YFCC embeddings)
    python eval_retrieval.py \\
        --embeddings_dir ./embeds/ilias_pairs.pt \\
        --checkpoint_path ./ckpts/siglip2-large-patch16-512 \\
        --eval_baseline
"""

import argparse
import os

from huggingface_hub import snapshot_download


def parse_args():
    p = argparse.ArgumentParser(
        description="Download ILIAS evaluation data from Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repo-id",
        default="MVRL/ILIAS",
        help="Hugging Face dataset repo id for ILIAS",
    )
    p.add_argument(
        "--local-dir",
        default=os.environ.get("DATASET_ROOT", "./data/ilias"),
        help="Local directory to download into ($DATASET_ROOT/ilias if set)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Concurrent download workers",
    )
    p.add_argument(
        "--revision",
        default="main",
        help="Branch/revision to download from",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    print(f"Downloading ILIAS from {args.repo_id} → {args.local_dir}")
    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=args.num_workers,
    )
    print(f"Done. Local mirror at: {local_path}")
    print(
        "\nNext step — compute paired embeddings:\n"
        "  python precompute_embeddings.py \\\n"
        f"      --extractor google/siglip2-large-patch16-512 \\\n"
        f"      --image_dir {local_path}/ilias-core \\\n"
        "      --output_path ./embeds/ilias_pairs.pt \\\n"
        "      --tar_regex '.*\\.tar$'\n"
    )


if __name__ == "__main__":
    main()

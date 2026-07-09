#!/usr/bin/env python
"""
Download the INQUIRE evaluation dataset from Hugging Face.

INQUIRE is a natural-language text-to-image retrieval benchmark built on
the iNaturalist 2021 image collection.  It contains ~250 expert-authored
queries, each paired with multiple target images drawn from a 5 M-image
retrieval pool.

What this script downloads
--------------------------
* **INQUIRE queries and annotations** — query texts and ground-truth image IDs.
  These are converted into paired (query-text, target-image) WebDataset shards
  that ``precompute_embeddings.py`` can process.

What you need to obtain separately
-----------------------------------
* **iNaturalist 2021 (train_mini / train)** — the full retrieval pool (~5 M
  images, ~300 GB).  Download from:
      https://github.com/visipedia/inat_comp/tree/master/2021
  Once available as WebDataset ``.tar`` shards, embed them with
  ``compute_embeds.py`` and pass the output dir to
  ``eval_retrieval.py --distractor_dirs``.

Usage
-----
    python downloading/download_inquire.py --local-dir ./data/inquire

After downloading, run the evaluation pipeline::

    # 1. Compute paired (query-text, target-image) embeddings
    python precompute_embeddings.py \\
        --extractor google/siglip2-large-patch16-512 \\
        --image_dir ./data/inquire/query-shards \\
        --output_path ./embeds/inquire_pairs.pt \\
        --tar_regex '.*\\.tar$'

    # 2. Compute iNaturalist distractor embeddings (if available)
    python compute_embeds.py \\
        --model_name google/siglip2-large-patch16-512 \\
        --shard_dir /path/to/inaturalist/shards \\
        --out_dir ./embeds/inquire_distractors

    # 3. Evaluate
    python eval_retrieval.py \\
        --embeddings_dir ./embeds/inquire_pairs.pt \\
        --checkpoint_path ./ckpts/siglip2-large-patch16-512 \\
        --distractor_dirs ./embeds/inquire_distractors \\
        --eval_baseline
"""

import argparse
import os

from huggingface_hub import snapshot_download


def parse_args():
    p = argparse.ArgumentParser(
        description="Download INQUIRE evaluation data from Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repo-id",
        default="olivialiu/INQUIRE",
        help="Hugging Face dataset repo id for INQUIRE",
    )
    p.add_argument(
        "--local-dir",
        default=os.environ.get("DATASET_ROOT", "./data/inquire"),
        help="Local directory to download into ($DATASET_ROOT/inquire if set)",
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

    print(f"Downloading INQUIRE from {args.repo_id} → {args.local_dir}")
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
        f"      --image_dir {local_path}/query-shards \\\n"
        "      --output_path ./embeds/inquire_pairs.pt \\\n"
        "      --tar_regex '.*\\.tar$'\n"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import os
import argparse
from huggingface_hub import snapshot_download

default_local_dir = os.environ.get("DATASET_ROOT", "cc12m-wds")
default_local_dir = os.path.join(default_local_dir, "cc12m")
def parse_args():
    p = argparse.ArgumentParser(
        description="Download pixparse/cc12m-wds WebDataset shards from Hugging Face."
    )
    p.add_argument(
        "--repo-id",
        default="pixparse/cc12m-wds",
        help="Hugging Face dataset repo id (default: pixparse/cc12m-wds)",
    )
    p.add_argument(
        "--local-dir",
        default=default_local_dir,
        help="Directory to download into (default: $DATASET_ROOT or ./cc12m-wds)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of concurrent download workers (default: 16)",
    )
    p.add_argument(
        "--revision",
        default="main",
        help="Revision/branch/commit on the Hub (default: main)",
    )
    return p.parse_args()

def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    allow_patterns = [
        "_info.json",
        "cc12m-train-*.tar",
    ]
    print(f"Downloading {args.repo_id} to {args.local_dir} ...")
    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        resume_download=True,
        max_workers=args.num_workers,
    )
    print(f"Done. Local mirror at: {local_path}")

if __name__ == "__main__":
    main()

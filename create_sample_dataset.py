"""
Create small sample datasets for testing ILIAS/INQUIRE evaluation workflows.

This script produces two kinds of outputs:

1. **Distractor shards** (``--mode distractors``) — WebDataset tar files where
   each sample has a ``jpg`` key and a ``__key__`` identifier.  These mimic the
   YFCC100M shards used as distractors for ILIAS, and the iNaturalist shards
   used for INQUIRE.  Pass the resulting directory to ``compute_embeds.py`` to
   get distractor embeddings.

2. **Query-target pairs** (``--mode pairs``) — WebDataset tar files where each
   sample has a ``jpg`` key AND a ``json`` key that contains
   ``{"image_id": <int>, "captions": [<str>, ...]}``.  This matches the format
   expected by ``precompute_embeddings.py``, which produces the paired
   (query-text, target-image) embeddings used during evaluation.

Generated images are small random RGB patches so the files are lightweight and
suitable only for end-to-end smoke tests, not for benchmarking quality.

Example — create a distractor shard set::

    python create_sample_dataset.py \\
        --mode distractors \\
        --out_dir ./sample_data/ilias_distractors \\
        --n_images 200 \\
        --n_shards 2

Example — create a query-target pair set::

    python create_sample_dataset.py \\
        --mode pairs \\
        --out_dir ./sample_data/inquire_pairs \\
        --n_images 50 \\
        --captions_per_image 3 \\
        --n_shards 1

After generating the sample data you can run the full evaluation pipeline::

    # Step 1 — distractor embeddings
    python compute_embeds.py \\
        --model_name openai/clip-vit-base-patch16 \\
        --shard_dir ./sample_data/ilias_distractors \\
        --out_dir ./sample_embeds/distractors \\
        --device cpu

    # Step 2 — paired query/target embeddings
    python precompute_embeddings.py \\
        --extractor openai/clip-vit-base-patch16 \\
        --image_dir ./sample_data/inquire_pairs \\
        --output_path ./sample_embeds/pairs.pt \\
        --tar_regex '.*\\.tar$' \\
        --device cpu

    # Step 3 — retrieval evaluation with QuARI
    python eval_retrieval.py \\
        --embeddings_dir ./sample_embeds \\
        --checkpoint_path ./ckpts/<model>/ \\
        --distractor_dirs ./sample_embeds/distractors \\
        --eval_baseline
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import tarfile
import tempfile
from pathlib import Path
from typing import List

from PIL import Image


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _random_image(width: int = 64, height: int = 64) -> bytes:
    """Return JPEG bytes for a random RGB image."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    img = Image.new("RGB", (width, height), color=(r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _random_caption() -> str:
    subjects = ["a dog", "a cat", "a bird", "a red car", "a mountain",
                "a beach", "a forest", "a city street", "a flower", "a person"]
    predicates = ["sitting on grass", "near a tree", "under bright lights",
                  "in the rain", "during sunset", "with blue sky background",
                  "close-up view", "from above"]
    return f"{random.choice(subjects)} {random.choice(predicates)}"


# --------------------------------------------------------------------------- #
# Shard writers
# --------------------------------------------------------------------------- #

def _write_distractor_shard(shard_path: Path, keys: List[str]) -> None:
    """Write a WebDataset shard with only image data (no captions)."""
    with tarfile.open(shard_path, "w") as tf:
        for key in keys:
            jpg_bytes = _random_image()
            info = tarfile.TarInfo(name=f"{key}.jpg")
            info.size = len(jpg_bytes)
            tf.addfile(info, io.BytesIO(jpg_bytes))


def _write_pairs_shard(
    shard_path: Path,
    image_ids: List[int],
    captions_per_image: int,
) -> None:
    """Write a WebDataset shard with image + JSON caption data."""
    with tarfile.open(shard_path, "w") as tf:
        for image_id in image_ids:
            key = f"{image_id:08d}"
            jpg_bytes = _random_image()
            captions = [_random_caption() for _ in range(captions_per_image)]
            meta = {"image_id": image_id, "captions": captions}
            meta_bytes = json.dumps(meta).encode("utf-8")

            jpg_info = tarfile.TarInfo(name=f"{key}.jpg")
            jpg_info.size = len(jpg_bytes)
            tf.addfile(jpg_info, io.BytesIO(jpg_bytes))

            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(meta_bytes)
            tf.addfile(json_info, io.BytesIO(meta_bytes))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def create_distractor_shards(
    out_dir: str | Path,
    n_images: int = 200,
    n_shards: int = 2,
) -> None:
    """
    Create ``n_shards`` WebDataset tar files containing ``n_images`` random
    images total.  Each sample only has a ``jpg`` key (plus implicit
    ``__key__``); there is no caption.

    Args:
        out_dir:   Output directory.
        n_images:  Total number of images across all shards.
        n_shards:  Number of shard files.
    """
    if n_shards > n_images:
        raise ValueError(
            f"n_shards ({n_shards}) must not exceed n_images ({n_images})"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_shard = [n_images // n_shards] * n_shards
    per_shard[-1] += n_images - sum(per_shard)  # remainder goes to last shard

    idx = 0
    for shard_i, count in enumerate(per_shard):
        keys = [f"img{idx + j:07d}" for j in range(count)]
        shard_path = out_dir / f"shard-{shard_i:04d}.tar"
        _write_distractor_shard(shard_path, keys)
        print(f"  Wrote {shard_path.name}  ({count} images)")
        idx += count

    print(f"Created {n_images} distractor images in {n_shards} shards → {out_dir}")


def create_pairs_shards(
    out_dir: str | Path,
    n_images: int = 50,
    captions_per_image: int = 3,
    n_shards: int = 1,
) -> None:
    """
    Create ``n_shards`` WebDataset tar files where each sample contains a
    ``jpg`` key (image) and a ``json`` key (caption metadata).

    The JSON format matches what ``datasets.expand_to_pairs`` expects::

        {"image_id": <int>, "captions": [<str>, ...]}

    Args:
        out_dir:             Output directory.
        n_images:            Total number of images across all shards.
        captions_per_image:  Number of captions per image.
        n_shards:            Number of shard files.
    """
    if n_shards > n_images:
        raise ValueError(
            f"n_shards ({n_shards}) must not exceed n_images ({n_images})"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_shard = [n_images // n_shards] * n_shards
    per_shard[-1] += n_images - sum(per_shard)

    image_id_start = 0
    for shard_i, count in enumerate(per_shard):
        image_ids = list(range(image_id_start, image_id_start + count))
        shard_path = out_dir / f"shard-{shard_i:04d}.tar"
        _write_pairs_shard(shard_path, image_ids, captions_per_image)
        total_pairs = count * captions_per_image
        print(f"  Wrote {shard_path.name}  ({count} images, {total_pairs} pairs)")
        image_id_start += count

    total_pairs = n_images * captions_per_image
    print(
        f"Created {n_images} images ({total_pairs} query-target pairs) "
        f"in {n_shards} shards → {out_dir}"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic WebDataset shards for smoke-testing "
                    "the ILIAS/INQUIRE evaluation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["distractors", "pairs"],
        help=(
            "'distractors' — image-only shards (for compute_embeds.py); "
            "'pairs' — image+caption shards (for precompute_embeddings.py)"
        ),
    )
    parser.add_argument("--out_dir", required=True,
                        help="Directory to write output tar shards into")
    parser.add_argument("--n_images", type=int, default=200,
                        help="Total number of images across all shards")
    parser.add_argument("--n_shards", type=int, default=2,
                        help="Number of output shard files")
    parser.add_argument(
        "--captions_per_image",
        type=int,
        default=3,
        help="(pairs mode only) Number of captions per image",
    )

    args = parser.parse_args()

    if args.mode == "distractors":
        create_distractor_shards(
            out_dir=args.out_dir,
            n_images=args.n_images,
            n_shards=args.n_shards,
        )
    else:
        create_pairs_shards(
            out_dir=args.out_dir,
            n_images=args.n_images,
            captions_per_image=args.captions_per_image,
            n_shards=args.n_shards,
        )


if __name__ == "__main__":
    main()

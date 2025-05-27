import argparse
import os
from pathlib import Path
from typing import List

import torch
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from feature_extractors import FeatureExtractorFactory   # <-- import the file above


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True,
                   help='e.g. "openai/clip-vit-large-patch14" or "ViT-B-32"')
    p.add_argument("--shard_dir", default="/u/ericx003/data/ilias/yfcc100m")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16",
                   help="Storage dtype on disk.")
    p.add_argument("--chunk_size", type=int, default=5000000,
                   help="#embeddings per saved file.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    # -------- fixed output dir & safe filename stem ----------------------- #
    out_dir = Path("./yfcc_embeds")
    out_dir.mkdir(exist_ok=True, parents=True)
    stem = args.model_name.replace("/", "-")

    # -------- CUDA / matmul settings -------------------------------------- #
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # -------- create extractor ------------------------------------------- #
    extractor = FeatureExtractorFactory.create_extractor(
        args.model_name, device=args.device
    )
    try:
        extractor = torch.compile(extractor)          # PyTorch 2.x kernel fusion
    except Exception:
        pass

    # -------- build WebDataset pipeline ---------------------------------- #
    shards: List[str] = sorted(
        str(Path(args.shard_dir, f))
        for f in os.listdir(args.shard_dir) if f.endswith(".tar")
    )
    dataset = (
        wds.WebDataset(shards, repeat=False)
        .decode("pil")
        .to_tuple("__key__", "jpg")      # keep original key + PIL image
    )

    def collate_pil(samples):
        keys, imgs = zip(*samples)           # tuple‑of‑tuples → two lists
        return list(keys), list(imgs)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_pil,
    )

    # -------- extraction loop with chunked writes ------------------------ #
    emb_buf, key_buf, chunk_idx = [], [], 0
    target_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    pbar = tqdm(loader, desc="Embedding batches", unit="batch")
    for keys, imgs in pbar:                     # imgs = list[PIL]
        with torch.no_grad():
            embs = extractor.extract_image_features(imgs)  # extractor handles device
        emb_buf.append(embs.cpu())
        key_buf.extend(keys)

        if len(key_buf) >= args.chunk_size:
            save_chunk(out_dir, stem, chunk_idx, key_buf, emb_buf, target_dtype)
            emb_buf, key_buf, chunk_idx = [], [], chunk_idx + 1

    if key_buf:  # leftovers
        save_chunk(out_dir, stem, chunk_idx, key_buf, emb_buf, target_dtype)

    print("All embeddings written to", out_dir.resolve())


# --------------------------------------------------------------------------- #
def save_chunk(out_dir: Path, stem: str, idx: int,
               keys: List[str], tensors: List[torch.Tensor],
               dtype) -> None:
    path = out_dir / f"{stem}_{idx:04d}.pt"
    torch.save(
        {"keys": keys, "embeddings": torch.cat(tensors).to(dtype)},
        path
    )
    print(f"Saved {path.name}  ({len(keys):,} embeddings)")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create a tiny WebDataset-format tar file for smoke-testing precompute/eval.

Creates a tar with two image entries (000000.jpg, 000001.jpg) and matching
JSON entries (000000.json, 000001.json) where each JSON contains a
`captions` list and an `image_id` field — the format expected by
`precompute_embeddings.py` with `expand_pairs=True`.

Usage:
    python scripts/create_synthetic_webdataset.py --out ./dataset/val_small.tar

This script only depends on Pillow (for image generation) and the standard library.
"""
import argparse
import io
import json
import tarfile
from PIL import Image


def make_rgb_image_bytes(color, size=(64, 64)):
    im = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


def build_tar(path: str):
    samples = [
        {
            "key": "000000",
            "image_bytes": make_rgb_image_bytes((200, 30, 30)),
            "json": {"image_id": 42, "captions": ["a red square", "an object red"]},
        },
        {
            "key": "000001",
            "image_bytes": make_rgb_image_bytes((30, 200, 30)),
            "json": {"image_id": 43, "captions": ["a green square", "an object green"]},
        },
    ]

    with tarfile.open(path, "w") as tf:
        for s in samples:
            # jpg entry
            img_name = f"{s['key']}.jpg"
            ti = tarfile.TarInfo(img_name)
            ti.size = len(s["image_bytes"])
            tf.addfile(ti, io.BytesIO(s["image_bytes"]))

            # json entry
            json_bytes = json.dumps(s["json"]).encode("utf-8")
            json_name = f"{s['key']}.json"
            tj = tarfile.TarInfo(json_name)
            tj.size = len(json_bytes)
            tf.addfile(tj, io.BytesIO(json_bytes))

    print(f"Wrote synthetic tar to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output tar path")
    args = parser.parse_args()
    build_tar(args.out)


if __name__ == "__main__":
    main()

import os
import json
import tarfile
import io
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--captions-json", required=True)
    p.add_argument("--out-tar", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.captions_json, "r") as f:
        data = json.load(f)
    id_to_fname = {img["id"]: img["file_name"] for img in data["images"]}
    anns = data["annotations"]
    with tarfile.open(args.out_tar, "w") as tar:
        for ann in anns:
            img_id = ann["image_id"]
            caption = ann["caption"]
            fname = id_to_fname[img_id]
            img_path = os.path.join(args.images_dir, fname)
            key = f"{os.path.splitext(fname)[0]}_{ann['id']}"
            with open(img_path, "rb") as fimg:
                img_bytes = fimg.read()
            img_info = tarfile.TarInfo(name=f"{key}.jpg")
            img_info.size = len(img_bytes)
            tar.addfile(img_info, io.BytesIO(img_bytes))
            cap_bytes = caption.strip().encode("utf-8")
            cap_info = tarfile.TarInfo(name=f"{key}.txt")
            cap_info.size = len(cap_bytes)
            tar.addfile(cap_info, io.BytesIO(cap_bytes))

if __name__ == "__main__":
    main()

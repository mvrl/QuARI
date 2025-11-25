#!/usr/bin/env bash
set -euo pipefail

ROOT="${DATASET_ROOT:?DATASET_ROOT must be set}"
DATA_DIR="${ROOT}/coco2017"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

mkdir -p downloads
cd downloads

if [ ! -f val2017.zip ]; then
  wget http://images.cocodataset.org/zips/val2017.zip
fi

if [ ! -f annotations_trainval2017.zip ]; then
  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi

cd ..
if [ ! -d val2017 ]; then
  unzip -q downloads/val2017.zip
fi

if [ ! -d annotations ]; then
  unzip -q downloads/annotations_trainval2017.zip
fi

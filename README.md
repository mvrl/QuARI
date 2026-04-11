# QuARI: Query Adaptive Retrieval Improvement: NeurIPS 2025

<div align="center">

[![Static Badge](https://img.shields.io/badge/2502.19781-red?label=arxiv)]([https://arxiv.org/abs/2502.19781](https://arxiv.org/abs/2505.21647))
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://ericx003.github.io/proj/quari/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow
)](https://huggingface.co/collections/MVRL/quari)

</center>
</div>

## Overview
Current multimodal embedding models are widely used for image-to-image and text-to-image retrieval, but their global embeddings often miss the fine-grained cues needed for challenging retrieval tasks. QuARI tackles this by learning a query-specific linear projection of a frozen backbone embedding space. A transformer hypernetwork maps each query to both an adapted query embedding and a low-rank projection matrix that is applied to all gallery embeddings, making the adaptation cheap enough to run over millions of items. Trained with a symmetric contrastive loss and additional “semi-positive” neighbors, QuARI emphasizes subspaces that are relevant to the current query while down-weighting irrelevant directions. Experiments on ILIAS and INQUIRE show that this simple query-conditioned adaptation consistently outperforms strong baselines, including static task-adapted encoders and heavyweight re-rankers, while remaining highly efficient at inference time.

## Setup

```bash
conda env create -f env.yml
conda activate vis-lang
```

## Data Setup
Set the appropriate download directory `downloading/setup_download.sh`
```bash
bash download_cc12m.sh
bash download_coco.sh
python cocototar.py \
    --images-dir /path/to/coco/images \
    --captions-json /path/to/coco/captions \
    --out-tar /path/to/output/tarfile
```

## Training

### Step 1: Precompute embeddings
```bash
python precompute_embeddings.py \
    --extractor openai/clip-vit-base-patch32 \
    --output_path ./precomputed/train_chunks \
    --image_dir ./data/images \
    --tar_regex '.*\.tar$' \
    --chunk_size 50000

```

### Step 2: Mine semi-positives
```bash
python mine_semipositives.py \
    --embeddings_path ./precomputed/train_embeds.pt \
    --output_path ./semipositives/train_semipos.pt \
    --k 100 \
    --top_n 2
```

### Step 3: Train QuARI
```bash
python train.py \
    --json_path ./data/train.json \
    --image_dir ./data/images \
    --extractor openai/clip-vit-base-patch32 \
    --use_precomputed \
    --precomputed_dir ./precomputed \
    --train_semipositives_path ./semipositives/train_semipos.pt \
    --batch_size 512 \
    --max_epochs 10 \
    --freeze_extractors \
    --output_dir ./outputs
```

## Evaluation

### Evaluating on COCO / Flickr (standard pairs)

```bash
python eval_retrieval.py \
    --embeddings_dir ./precomputed/val \
    --checkpoint_path ./outputs/checkpoints/best.ckpt \
    --eval_baseline
```

### Evaluating on ILIAS and INQUIRE

ILIAS and INQUIRE use a large pool of distractor images (YFCC100M and
iNaturalist respectively).  The evaluation workflow has two stages:

**Stage 1 — compute distractor embeddings** with `compute_embeds.py`.  The
distractor shards must be WebDataset `.tar` files where each sample contains a
`jpg` key (image bytes) and a `__key__` identifier.

```bash
# ILIAS: YFCC100M distractors
python compute_embeds.py \
    --model_name openai/clip-vit-base-patch16 \
    --shard_dir /path/to/yfcc100m/shards \
    --out_dir ./embeds/ilias_distractors

# INQUIRE: iNaturalist distractors
python compute_embeds.py \
    --model_name openai/clip-vit-base-patch16 \
    --shard_dir /path/to/inaturalist/shards \
    --out_dir ./embeds/inquire_distractors
```

**Stage 2 — compute paired query/target embeddings** for the evaluation set
(ILIAS-core or INQUIRE queries) with `precompute_embeddings.py`.  The shards
must contain a `jpg` key (target image) and a `json` key with
`{"image_id": <int>, "captions": [<str>, ...]}`.

```bash
python precompute_embeddings.py \
    --extractor openai/clip-vit-base-patch16 \
    --image_dir /path/to/ilias_core/shards \
    --output_path ./embeds/ilias_pairs.pt \
    --tar_regex '.*\.tar$'
```

**Stage 3 — run retrieval evaluation**

```bash
python eval_retrieval.py \
    --embeddings_dir ./embeds/ilias_pairs.pt \
    --checkpoint_path ./ckpts/clip-vit-base-patch16/ \
    --distractor_dirs ./embeds/ilias_distractors \
    --eval_baseline
```

### Testing the pipeline with synthetic sample data

`create_sample_dataset.py` generates small synthetic datasets in the correct
formats so you can smoke-test the full pipeline without downloading large
datasets:

```bash
# Create synthetic distractor shards (image-only, mimics YFCC100M / iNaturalist)
python create_sample_dataset.py \
    --mode distractors \
    --out_dir ./sample_data/distractors \
    --n_images 200 \
    --n_shards 2

# Create synthetic query/target pair shards (image + captions, mimics ILIAS-core / INQUIRE)
python create_sample_dataset.py \
    --mode pairs \
    --out_dir ./sample_data/pairs \
    --n_images 50 \
    --captions_per_image 3

# Compute distractor embeddings
python compute_embeds.py \
    --model_name openai/clip-vit-base-patch16 \
    --shard_dir ./sample_data/distractors \
    --out_dir ./sample_embeds/distractors \
    --device cpu

# Compute paired embeddings
python precompute_embeddings.py \
    --extractor openai/clip-vit-base-patch16 \
    --image_dir ./sample_data/pairs \
    --output_path ./sample_embeds/pairs.pt \
    --tar_regex '.*\.tar$' \
    --device cpu

# Run retrieval evaluation
python eval_retrieval.py \
    --embeddings_dir ./sample_embeds \
    --checkpoint_path ./ckpts/<model>/ \
    --distractor_dirs ./sample_embeds/distractors \
    --eval_baseline
```

## Using Pretrained Models
Get pretrained weights by running `download_ckpts.py`.


## Citation

```bibtex
@inproceedings{xing2025quari,
  title={QuARI: Query Adaptive Retrieval Improvement},
  author={Xing, Eric and Stylianou, Abby and Pless, Robert and Jacobs, Nathan},
  booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```


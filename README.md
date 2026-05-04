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

### Training data (CC12M + COCO)
Set the download directory in `downloading/setup_download.sh`, then:
```bash
python downloading/download_cc12m.py
bash downloading/download_coco.sh
python downloading/cocototar.py \
    --images-dir /path/to/coco/images \
    --captions-json /path/to/coco/captions \
    --out-tar /path/to/output/tarfile
```

### Evaluation data

#### ILIAS
ILIAS pairs ~1 000 text queries with fine-grained target images drawn from a
15 M-image YFCC subset.

```bash
# Download ILIAS-core query shards
python downloading/download_ilias.py --local-dir ./data/ilias
```

The YFCC15M distractor images (~2 TB) must be requested separately from the
[YFCC100M official page](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/).

#### INQUIRE
INQUIRE contains ~250 natural-language queries over a 5 M-image iNaturalist
retrieval pool.

```bash
# Download INQUIRE query annotations
python downloading/download_inquire.py --local-dir ./data/inquire
```

The iNaturalist 2021 distractor images must be downloaded separately from the
[iNaturalist 2021 competition page](https://github.com/visipedia/inat_comp/tree/master/2021).

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

## Using Pretrained Models

Download all four pretrained QuARI checkpoints from HuggingFace:

```bash
python download_ckpts.py
# Downloads to ./ckpts/
#   clip-vit-base-patch16/
#   clip-vit-large-patch14/
#   siglip2-base-patch16-512/
#   siglip2-large-patch16-512/
```

## Evaluation

### Evaluating on COCO / Flickr (standard pairs)

```bash
python eval_retrieval.py \
    --embeddings_dir ./precomputed/val \
    --checkpoint_path ./ckpts/siglip2-large-patch16-512 \
    --eval_baseline
```

### Evaluating on ILIAS

Full replication of ILIAS results (SigLIP2-L, which Vladan reproduced in issue #1):

```bash
# Step 1 — download ILIAS-core query shards (if not already done)
python downloading/download_ilias.py --local-dir ./data/ilias

# Step 2 — compute paired (query-text, target-image) embeddings
python precompute_embeddings.py \
    --extractor google/siglip2-large-patch16-512 \
    --image_dir ./data/ilias/ilias-core \
    --output_path ./embeds/ilias_pairs.pt \
    --tar_regex '.*\.tar$'

# Step 3 — embed YFCC15M distractors (requires the ~2 TB YFCC data)
python compute_embeds.py \
    --model_name google/siglip2-large-patch16-512 \
    --shard_dir /path/to/yfcc15m/shards \
    --out_dir ./embeds/ilias_distractors

# Step 4 — run retrieval evaluation
python eval_retrieval.py \
    --embeddings_dir ./embeds/ilias_pairs.pt \
    --checkpoint_path ./ckpts/siglip2-large-patch16-512 \
    --distractor_dirs ./embeds/ilias_distractors \
    --eval_baseline
```

To replicate other ILIAS model variants replace `siglip2-large-patch16-512` with
one of the other pretrained model names above (e.g. `clip-vit-large-patch14`).

### Evaluating on INQUIRE

```bash
# Step 1 — download INQUIRE query shards (if not already done)
python downloading/download_inquire.py --local-dir ./data/inquire

# Step 2 — compute paired embeddings
python precompute_embeddings.py \
    --extractor google/siglip2-large-patch16-512 \
    --image_dir ./data/inquire/query-shards \
    --output_path ./embeds/inquire_pairs.pt \
    --tar_regex '.*\.tar$'

# Step 3 — embed iNaturalist 2021 distractors (~5 M images)
python compute_embeds.py \
    --model_name google/siglip2-large-patch16-512 \
    --shard_dir /path/to/inaturalist2021/shards \
    --out_dir ./embeds/inquire_distractors

# Step 4 — run retrieval evaluation
python eval_retrieval.py \
    --embeddings_dir ./embeds/inquire_pairs.pt \
    --checkpoint_path ./ckpts/siglip2-large-patch16-512 \
    --distractor_dirs ./embeds/inquire_distractors \
    --eval_baseline
```

### Smoke test — verify the pipeline without large datasets

`create_sample_dataset.py` generates tiny synthetic datasets in the correct
WebDataset format so you can verify the full pipeline end-to-end before
committing to hours of embedding computation.  The smoke test uses
`--baseline_only` so no pretrained checkpoint is required.

```bash
# 1. Create synthetic data
python create_sample_dataset.py \
    --mode distractors \
    --out_dir ./sample_data/distractors \
    --n_images 200 \
    --n_shards 2

python create_sample_dataset.py \
    --mode pairs \
    --out_dir ./sample_data/pairs \
    --n_images 50 \
    --captions_per_image 3

# 2. Compute distractor embeddings
python compute_embeds.py \
    --model_name openai/clip-vit-base-patch16 \
    --shard_dir ./sample_data/distractors \
    --out_dir ./sample_embeds/distractors \
    --device cpu

# 3. Compute paired embeddings
python precompute_embeddings.py \
    --extractor openai/clip-vit-base-patch16 \
    --image_dir ./sample_data/pairs \
    --output_path ./sample_embeds/pairs.pt \
    --tar_regex '.*\.tar$' \
    --device cpu

# 4. Baseline-only evaluation (no checkpoint needed)
python eval_retrieval.py \
    --embeddings_dir ./sample_embeds \
    --distractor_dirs ./sample_embeds/distractors \
    --baseline_only \
    --k_values 1 5 10

# 5. Full QuARI evaluation (requires a downloaded checkpoint)
python eval_retrieval.py \
    --embeddings_dir ./sample_embeds \
    --checkpoint_path ./ckpts/clip-vit-base-patch16 \
    --distractor_dirs ./sample_embeds/distractors \
    --eval_baseline \
    --k_values 1 5 10 \
    --device cpu
```


## Citation

```bibtex
@inproceedings{xing2025quari,
  title={QuARI: Query Adaptive Retrieval Improvement},
  author={Xing, Eric and Stylianou, Abby and Pless, Robert and Jacobs, Nathan},
  booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```


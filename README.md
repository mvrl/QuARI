# QuARI: Query Adaptive Retrieval Improvement

Official implementation of QuARI from NeurIPS 2025.

## Setup

```bash
pip install torch torchvision pytorch-lightning
pip install transformers open_clip_torch webdataset
pip install faiss-gpu pillow tqdm
```

## Training

### Step 1: Precompute embeddings
```bash
python precompute_embeddings.py \
    --extractor openai/clip-vit-base-patch32 \
    --json_path ./data/train.json \
    --image_dir ./data/images \
    --output_path ./precomputed/train_embeds.pt
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

```bash
python eval_retrieval.py \
    --embeddings_dir ./precomputed/val \
    --checkpoint_path ./outputs/checkpoints/best.ckpt \
    --distractor_dirs ./distractors/yfcc \
    --eval_baseline
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


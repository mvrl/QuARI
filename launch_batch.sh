#!/bin/bash
#SBATCH --job-name=./slurm_outputs/column_wise_transformer_vit_large
#SBATCH --output=./slurm_outputs/column_wise_transformer_vit_large.log
#SBATCH --error=./slurm_outputs/column_wise_transformer_vit_large.log
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account bebk-tgirails 
#SBATCH --gpu-bind=closest

source activate /u/ericx003/conda/envs/vis-lang
cd /u/ericx003/code/e2e-hypernet

srun python train.py \
    --json_path /u/ericx003/data/coco/coco_dataset/annotations/train_captions.json \
    --image_dir /u/ericx003/data/coco/coco_dataset/ \
    --extractor_type clip \
    --extractor_model openai/clip-vit-large-patch14 \
    --batch_size 320 \
    --num_denoising_steps 4 \
    --learning_rate 5e-4 \
    --weight_decay 1e-2 \
    --temperature 0.07 \
    --output_dir ./clip-vit-large-column-wise \
    --experiment_name column_wise_transformer_vit_large \
    --num_encoder_layers 4 \
    --nhead 8 \
    --hidden_dim 768 \
    --matrix_sequence_len 4 \
    --low_rank_dim 64 \
    --use_precomputed \
    --precomputed_dir ./precomputed_embeddings \
    --freeze_extractors \
    --gradient_clip_val 1.0 \
    --accumulate_grad_batches 2 \
    --column_wise
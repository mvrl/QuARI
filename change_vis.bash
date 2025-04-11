python vis_change.py \
    --checkpoint_path /u/ericx003/code/e2e-hypernet/clip-vit-large-column-wise/column_wise_transformer_vit_large/checkpoints/model-epoch=00-val_loss=6.319e-04.ckpt \
    --extractor_type clip \
    --extractor_model openai/clip-vit-large-patch14 \
    --class1 dog \
    --class2 cat \
    --images_per_class 100 \
    --coco_dir /u/ericx003/data/coco/coco_dataset \
    --output_dir ./class_visualization_results \
    --use_precomputed \
    --precomputed_dir ./precomputed_embeddings

# --class1 person --class2 horse
# --class1 car --class2 truck
# --class1 boat --class2 airplane
# --class1 chair --class2 couch
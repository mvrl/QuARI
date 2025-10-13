#!/bin/bash

# Download ILIAS dataset
mkdir -p ilias
cd ilias
wget https://vrg.fel.cvut.cz/ilias_data/ilias_core.tar
tar -xvf ilias_core.tar
rm ilias_core.tar

# Compute embeddings
cd ..
python compute_features.py

# Download checkpoints
python download_ckpts.py

# Evaluate
python eval_ilias.py
"""
Mine semi-positive samples using FAISS for efficient nearest neighbor search.
Implements the semi-positive mining strategy from Section 3.3 of the QuARI paper.
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import numpy as np
import faiss
from tqdm.auto import tqdm


def load_embeddings_from_pt(pt_path: str) -> torch.Tensor:
    data = torch.load(pt_path, map_location='cpu')
    if 'embeddings' in data:
        return data['embeddings']
    elif 'image_embeddings' in data:
        return data['image_embeddings']
    else:
        raise ValueError("Unknown embedding file format")


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    dim = embeddings.shape[1]
    
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        index = faiss.IndexFlatIP(dim)
    
    index.add(embeddings)
    return index


def mine_semipositives(
    embeddings: np.ndarray,
    k: int = 100,
    top_n: int = 2,
    use_gpu: bool = True,
    batch_size: int = 10000,
    temperature: float = 0.07
) -> Dict[str, np.ndarray]:
    n_samples = embeddings.shape[0]
    
    index = build_faiss_index(embeddings, use_gpu=use_gpu)
    
    all_semipos_embeddings = []
    all_semipos_weights = []
    
    for start_idx in tqdm(range(0, n_samples, batch_size), desc="Mining semi-positives"):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = embeddings[start_idx:end_idx]
        
        similarities, indices = index.search(batch, k + 1)
        
        indices = indices[:, 1:]
        similarities = similarities[:, 1:]
        
        logits = similarities / temperature
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        top_indices = indices[:, :top_n]
        top_weights = weights[:, :top_n]
        
        top_embeddings = embeddings[top_indices]
        
        all_semipos_embeddings.append(top_embeddings)
        all_semipos_weights.append(top_weights)
    
    semipos_embeddings = np.concatenate(all_semipos_embeddings, axis=0)
    semipos_weights = np.concatenate(all_semipos_weights, axis=0)
    
    return {
        'semipositive_embeddings': semipos_embeddings,
        'semipositive_weights': semipos_weights,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_path', required=True, help='Path to embeddings .pt file')
    parser.add_argument('--output_path', required=True, help='Path to save semi-positive data')
    parser.add_argument('--k', type=int, default=100, help='Number of nearest neighbors')
    parser.add_argument('--top_n', type=int, default=2, help='Number of semi-positives per sample')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU for FAISS')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for mining')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for softmax weights')
    args = parser.parse_args()
    
    print(f"Loading embeddings from {args.embeddings_path}")
    embeddings = load_embeddings_from_pt(args.embeddings_path)
    print(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    embeddings_np = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)
    
    print("Mining semi-positives...")
    semipos_data = mine_semipositives(
        embeddings_np,
        k=args.k,
        top_n=args.top_n,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size,
        temperature=args.temperature
    )
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(semipos_data, args.output_path)
    print(f"Saved semi-positive data to {args.output_path}")
    print(f"Semi-positive embeddings shape: {semipos_data['semipositive_embeddings'].shape}")
    print(f"Semi-positive weights shape: {semipos_data['semipositive_weights'].shape}")


if __name__ == '__main__':
    main()


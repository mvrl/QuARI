import argparse
import os
from pathlib import Path
from typing import List, Dict
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

from feature_extractors import FeatureExtractorFactory
from transformer_hypernetwork import load_model


def load_embeddings_from_dir(embed_dir: str, pattern: str = "*.pt") -> Dict[str, torch.Tensor]:
    embed_dir = Path(embed_dir)
    files = sorted(embed_dir.glob(pattern))
    
    all_text = []
    all_image = []
    
    for f in files:
        data = torch.load(f, map_location='cpu')
        all_text.append(data['text_embeddings'])
        all_image.append(data['image_embeddings'])
    
    return {
        'text_embeddings': torch.cat(all_text, dim=0),
        'image_embeddings': torch.cat(all_image, dim=0)
    }


def load_distractor_embeddings_from_dir(embed_dir: str, pattern: str = "*.pt") -> torch.Tensor:
    """Load image-only embeddings produced by ``compute_embeds.py``.

    Each ``.pt`` file in *embed_dir* must contain a dict with either:
    * ``"embeddings"`` key (format written by ``compute_embeds.py``), or
    * ``"image_embeddings"`` key (format written by ``precompute_embeddings.py``).

    Returns a single concatenated tensor of shape ``[N, D]``.
    """
    embed_dir = Path(embed_dir)
    files = sorted(embed_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {embed_dir}")

    chunks = []
    for f in files:
        data = torch.load(f, map_location='cpu')
        if 'embeddings' in data:
            chunks.append(data['embeddings'])
        elif 'image_embeddings' in data:
            chunks.append(data['image_embeddings'])
        else:
            raise KeyError(
                f"{f}: expected 'embeddings' or 'image_embeddings' key, "
                f"got {list(data.keys())}"
            )
    return torch.cat(chunks, dim=0)


def compute_recall_at_k(similarities: torch.Tensor, k_values: List[int]) -> Dict[str, float]:
    n_queries = similarities.shape[0]
    
    sorted_indices = torch.argsort(similarities, dim=1, descending=True)
    
    recalls = {}
    for k in k_values:
        correct = 0
        for i in range(n_queries):
            if i in sorted_indices[i, :k]:
                correct += 1
        recalls[f'R@{k}'] = correct / n_queries
    
    return recalls


def evaluate_retrieval(
    hypernetwork,
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    distractor_embeddings: torch.Tensor = None,
    batch_size: int = 256,
    k_values: List[int] = [1, 5, 10, 50],
    device: str = 'cuda'
):
    hypernetwork = hypernetwork.to(device)
    hypernetwork.eval()
    
    if distractor_embeddings is not None:
        image_embeddings = torch.cat([image_embeddings, distractor_embeddings], dim=0)
    
    n_queries = text_embeddings.shape[0]
    n_images = image_embeddings.shape[0]
    
    all_similarities = torch.zeros(n_queries, n_images)
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_queries, batch_size), desc="Computing similarities"):
            end_idx = min(start_idx + batch_size, n_queries)
            batch_text = text_embeddings[start_idx:end_idx].to(device)
            
            out = hypernetwork.forward(batch_text)
            refined_query = out['refined_query']
            W_image = out['W_image']
            
            img_transformed = torch.bmm(
                image_embeddings.unsqueeze(0).expand(refined_query.shape[0], -1, -1).to(device),
                W_image
            )
            img_transformed = F.normalize(img_transformed, dim=-1)
            
            similarities = torch.einsum('be,bme->bm', refined_query, img_transformed)
            all_similarities[start_idx:end_idx] = similarities.cpu()
    
    recalls = compute_recall_at_k(all_similarities, k_values)
    
    return recalls


def evaluate_baseline(
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    distractor_embeddings: torch.Tensor = None,
    k_values: List[int] = [1, 5, 10, 50]
):
    if distractor_embeddings is not None:
        image_embeddings = torch.cat([image_embeddings, distractor_embeddings], dim=0)
    
    similarities = torch.matmul(text_embeddings, image_embeddings.t())
    recalls = compute_recall_at_k(similarities, k_values)
    
    return recalls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_dir', required=True, help='Directory with paired embeddings')
    parser.add_argument('--checkpoint_path', default=None, help='Path to QuARI checkpoint (not required with --baseline_only)')
    parser.add_argument('--distractor_dirs', nargs='*', default=None, help='Directories with distractor embeddings')
    parser.add_argument('--pattern', default='*.pt', help='File pattern for embeddings')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1, 5, 10, 50], help='K values for recall')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--eval_baseline', action='store_true', help='Also evaluate baseline CLIP similarity')
    parser.add_argument('--baseline_only', action='store_true',
                        help='Run only the baseline evaluation (no QuARI model required)')
    args = parser.parse_args()

    if not args.baseline_only and args.checkpoint_path is None:
        parser.error('--checkpoint_path is required unless --baseline_only is set')

    print(f"Loading embeddings from {args.embeddings_dir}")
    data = load_embeddings_from_dir(args.embeddings_dir, args.pattern)
    text_embeddings = F.normalize(data['text_embeddings'], dim=-1)
    image_embeddings = F.normalize(data['image_embeddings'], dim=-1)
    
    print(f"Loaded {text_embeddings.shape[0]} query-image pairs")
    
    distractor_embeddings = None
    if args.distractor_dirs:
        all_distractors = []
        for dist_dir in args.distractor_dirs:
            print(f"Loading distractors from {dist_dir}")
            dist_embeds = load_distractor_embeddings_from_dir(dist_dir, args.pattern)
            # Cast to float32: compute_embeds.py saves fp16 by default, and F.normalize
            # requires consistent precision with the paired embeddings (fp32).
            all_distractors.append(F.normalize(dist_embeds.float(), dim=-1))
        distractor_embeddings = torch.cat(all_distractors, dim=0)
        print(f"Loaded {distractor_embeddings.shape[0]} distractor images")

    if args.baseline_only:
        print("\nEvaluating baseline (CLIP similarity)...")
        baseline_recalls = evaluate_baseline(
            text_embeddings,
            image_embeddings,
            distractor_embeddings,
            k_values=args.k_values
        )
        print("\n=== Baseline Results ===")
        for metric, value in sorted(baseline_recalls.items()):
            print(f"{metric}: {value:.4f}")
        return

    print(f"\nLoading QuARI model from {args.checkpoint_path}")
    hypernetwork, metadata = load_model(args.checkpoint_path)
    
    print("\nEvaluating QuARI...")
    quari_recalls = evaluate_retrieval(
        hypernetwork,
        text_embeddings,
        image_embeddings,
        distractor_embeddings,
        batch_size=args.batch_size,
        k_values=args.k_values,
        device=args.device
    )
    
    print("\n=== QuARI Results ===")
    for metric, value in sorted(quari_recalls.items()):
        print(f"{metric}: {value:.4f}")
    
    if args.eval_baseline:
        print("\nEvaluating baseline (CLIP similarity)...")
        baseline_recalls = evaluate_baseline(
            text_embeddings,
            image_embeddings,
            distractor_embeddings,
            k_values=args.k_values
        )
        
        print("\n=== Baseline Results ===")
        for metric, value in sorted(baseline_recalls.items()):
            print(f"{metric}: {value:.4f}")
        
        print("\n=== Improvement ===")
        for metric in sorted(quari_recalls.keys()):
            improvement = quari_recalls[metric] - baseline_recalls[metric]
            print(f"{metric}: {improvement:+.4f}")


if __name__ == '__main__':
    main()


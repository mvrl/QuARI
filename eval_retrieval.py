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
    all_image_ids = []
    has_image_ids = True
    
    for f in files:
        data = torch.load(f, map_location='cpu')
        all_text.append(data['text_embeddings'])
        all_image.append(data['image_embeddings'])
        if has_image_ids and 'image_ids' in data:
            all_image_ids.extend(list(data['image_ids']))
        else:
            has_image_ids = False
            all_image_ids = []

    output = {
        'text_embeddings': torch.cat(all_text, dim=0),
        'image_embeddings': torch.cat(all_image, dim=0)
    }
    if has_image_ids:
        output['image_ids'] = all_image_ids
    return output


def compute_recall_at_k(
    similarities: torch.Tensor,
    k_values: List[int],
    query_image_ids: List = None,
    gallery_image_ids: List = None,
) -> Dict[str, float]:
    n_queries = similarities.shape[0]
    
    sorted_indices = torch.argsort(similarities, dim=1, descending=True)
    
    use_image_ids = (
        query_image_ids is not None
        and gallery_image_ids is not None
        and len(query_image_ids) == n_queries
        and len(gallery_image_ids) == similarities.shape[1]
    )
    positive_lookup = None
    if use_image_ids:
        positive_lookup = {}
        for idx, image_id in enumerate(gallery_image_ids):
            positive_lookup.setdefault(image_id, set()).add(idx)

    recalls = {}
    for k in k_values:
        correct = 0
        for i in range(n_queries):
            if use_image_ids:
                positives = positive_lookup.get(query_image_ids[i], set())
                top_k = sorted_indices[i, :k].tolist()
                if any(idx in positives for idx in top_k):
                    correct += 1
            elif i in sorted_indices[i, :k]:
                correct += 1
        recalls[f'R@{k}'] = correct / n_queries
    
    return recalls


def evaluate_retrieval(
    hypernetwork,
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    query_image_ids: List = None,
    gallery_image_ids: List = None,
    distractor_embeddings: torch.Tensor = None,
    batch_size: int = 256,
    k_values: List[int] = [1, 5, 10, 50],
    device: str = 'cuda'
):
    hypernetwork = hypernetwork.to(device)
    hypernetwork.eval()
    
    if distractor_embeddings is not None:
        image_embeddings = torch.cat([image_embeddings, distractor_embeddings], dim=0)
        if gallery_image_ids is not None:
            distractor_sentinel = object()
            gallery_image_ids = list(gallery_image_ids) + [distractor_sentinel] * distractor_embeddings.shape[0]
    
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
    
    recalls = compute_recall_at_k(all_similarities, k_values, query_image_ids, gallery_image_ids)
    
    return recalls


def evaluate_baseline(
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    query_image_ids: List = None,
    gallery_image_ids: List = None,
    distractor_embeddings: torch.Tensor = None,
    k_values: List[int] = [1, 5, 10, 50]
):
    if distractor_embeddings is not None:
        image_embeddings = torch.cat([image_embeddings, distractor_embeddings], dim=0)
        if gallery_image_ids is not None:
            distractor_sentinel = object()
            gallery_image_ids = list(gallery_image_ids) + [distractor_sentinel] * distractor_embeddings.shape[0]
    
    similarities = torch.matmul(text_embeddings, image_embeddings.t())
    recalls = compute_recall_at_k(similarities, k_values, query_image_ids, gallery_image_ids)
    
    return recalls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_dir', required=True, help='Directory with paired embeddings')
    parser.add_argument('--checkpoint_path', required=True, help='Path to QuARI checkpoint')
    parser.add_argument('--distractor_dirs', nargs='*', default=None, help='Directories with distractor embeddings')
    parser.add_argument('--pattern', default='*.pt', help='File pattern for embeddings')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1, 5, 10, 50], help='K values for recall')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--eval_baseline', action='store_true', help='Also evaluate baseline')
    args = parser.parse_args()
    
    print(f"Loading embeddings from {args.embeddings_dir}")
    data = load_embeddings_from_dir(args.embeddings_dir, args.pattern)
    text_embeddings = F.normalize(data['text_embeddings'], dim=-1)
    image_embeddings = F.normalize(data['image_embeddings'], dim=-1)
    query_image_ids = data.get('image_ids')
    gallery_image_ids = data.get('image_ids')
    
    print(f"Loaded {text_embeddings.shape[0]} query-image pairs")
    
    distractor_embeddings = None
    if args.distractor_dirs:
        all_distractors = []
        for dist_dir in args.distractor_dirs:
            print(f"Loading distractors from {dist_dir}")
            dist_data = load_embeddings_from_dir(dist_dir, args.pattern)
            all_distractors.append(F.normalize(dist_data['image_embeddings'], dim=-1))
        distractor_embeddings = torch.cat(all_distractors, dim=0)
        print(f"Loaded {distractor_embeddings.shape[0]} distractor images")
    
    print(f"\nLoading QuARI model from {args.checkpoint_path}")
    hypernetwork, metadata = load_model(args.checkpoint_path)
    
    print("\nEvaluating QuARI...")
    quari_recalls = evaluate_retrieval(
        hypernetwork,
        text_embeddings,
        image_embeddings,
        query_image_ids,
        gallery_image_ids,
        distractor_embeddings,
        batch_size=args.batch_size,
        k_values=args.k_values,
        device=args.device
    )
    
    print("\n=== QuARI Results ===")
    for metric, value in sorted(quari_recalls.items()):
        print(f"{metric}: {value:.4f}")
    
    if args.eval_baseline:
        print("\nEvaluating baseline...")
        baseline_recalls = evaluate_baseline(
            text_embeddings,
            image_embeddings,
            query_image_ids,
            gallery_image_ids,
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

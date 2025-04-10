import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

from feature_extractors import FeatureExtractorFactory
from transformer_hypernetwork import TransformerHypernetwork


class PersonalizedRetrievalModule(pl.LightningModule):
    """
    PyTorch Lightning module for training and evaluating the personalized text-to-image retrieval system.
    """
    
    def __init__(
        self,
        extractor_type: str = "clip",
        extractor_model: str = "ViT-B/32",
        embedding_dim: int = 512,
        hidden_dim: int = 768,
        num_denoising_steps: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        use_residual: bool = False,
        freeze_extractors: bool = True,
        # Transformer hypernetwork parameters
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dropout: float = 0.1,
        # Diffusion parameters
        scale_factor: float = 0.1
    ):
        """
        Initialize the personalized retrieval module.
        
        Args:
            extractor_type: Type of feature extractor ('clip' or 'siglip')
            extractor_model: Model name for the feature extractor
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension for the hypernetwork
            num_denoising_steps: Number of iterative refinement steps
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            temperature: Temperature parameter for contrastive loss
            use_residual: Whether to use residual connection in personalization
            freeze_extractors: Whether to freeze the feature extractors
            nhead: Number of attention heads in transformer
            num_encoder_layers: Number of transformer encoder layers
            dropout: Dropout rate
            scale_factor: Scale factor for diffusion noise and updates
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create feature extractor
        self.feature_extractor = FeatureExtractorFactory.create_extractor(
            extractor_type=extractor_type,
            model_name=extractor_model,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.feature_extractor.model.to(self.device)
        
        # Update embedding dimension based on feature extractor
        self.embedding_dim = self.feature_extractor.feature_dim
        
        # Create transformer hypernetwork for personalized transformation
        self.hypernetwork = TransformerHypernetwork(
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            num_denoising_steps=num_denoising_steps,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            use_residual=use_residual,
            scale_factor=scale_factor
        )
        
        # Freeze feature extractors if specified
        if freeze_extractors:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                
        # Temperature parameter for contrastive loss
        self.temperature = temperature
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            batch: Batch dictionary containing 'query_text', 'query_image', and 'negative_images'
            
        Returns:
            Dictionary containing computed embeddings and matrices
        """
        # Extract features
        if "text_features" in batch and "query_image_features" in batch:
            # Use precomputed features if available
            text_features = batch["text_features"]
            query_image_features = batch["query_image_features"]
            neg_image_features = batch.get("neg_image_features", None)
        else:
            # Extract features from raw inputs
            query_text = batch["query_text"]
            query_image = batch["query_image"]
            negative_images = batch.get("negative_images", None)
            
            # Extract features
            text_features = self.feature_extractor.extract_text_features(query_text)
            query_image_features = self.feature_extractor.extract_image_features(query_image)
            
            # Process negative images if present
            neg_image_features = None
            if negative_images is not None:
                if isinstance(negative_images, list):
                    # Handle batch of lists of varying lengths
                    all_neg_features = []
                    for neg_batch in negative_images:
                        neg_features = self.feature_extractor.extract_image_features(neg_batch)
                        all_neg_features.append(neg_features)
                        
                    # Pad to same length for batching
                    max_negs = max(len(neg) for neg in all_neg_features)
                    padded_neg_features = []
                    for neg_features in all_neg_features:
                        if len(neg_features) < max_negs:
                            padding = torch.zeros(
                                max_negs - len(neg_features),
                                neg_features.shape[1],
                                device=neg_features.device
                            )
                            neg_features = torch.cat([neg_features, padding], dim=0)
                        padded_neg_features.append(neg_features)
                        
                    neg_image_features = torch.stack(padded_neg_features)
                else:
                    # Handle tensor input
                    neg_image_features = self.feature_extractor.extract_image_features(negative_images)
        
        # Generate personalization matrix
        personalization_matrix, intermediate_matrices = self.hypernetwork(
            text_features, return_all_steps=True
        )
        
        # Apply personalization to text features
        if self.hypernetwork.use_residual:
            personalized_text = text_features + torch.bmm(
                text_features.unsqueeze(1),
                personalization_matrix
            ).squeeze(1)
        else:
            personalized_text = torch.bmm(
                text_features.unsqueeze(1),
                personalization_matrix
            ).squeeze(1)
        
        # Normalize
        personalized_text = F.normalize(personalized_text, dim=-1)
        
        return {
            "text_features": text_features,
            "personalized_text": personalized_text,
            "query_image_features": query_image_features,
            "neg_image_features": neg_image_features,
            "personalization_matrix": personalization_matrix,
            "intermediate_matrices": intermediate_matrices
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss for the model.
        
        Args:
            outputs: Outputs from the forward pass
            
        Returns:
            Dictionary of computed losses
        """
        personalized_text = outputs["personalized_text"]
        query_image_features = outputs["query_image_features"]
        neg_image_features = outputs["neg_image_features"]
        personalization_matrix = outputs["personalization_matrix"]
        
        losses = {}
        
        # Compute similarity between personalized text and positive image
        pos_similarity = F.cosine_similarity(personalized_text, query_image_features, dim=1)
        
        # Compute similarity between personalized text and negative images
        if neg_image_features is not None:
            if neg_image_features.dim() == 3:
                # Multiple negative images per query
                batch_size, num_negs, feat_dim = neg_image_features.shape
                neg_similarity = torch.bmm(
                    personalized_text.unsqueeze(1),
                    neg_image_features.view(batch_size, num_negs, feat_dim).transpose(1, 2)
                ).squeeze(1)
            else:
                # One negative image per query
                neg_similarity = F.cosine_similarity(personalized_text, neg_image_features, dim=1)
        
            # Scale by temperature
            pos_similarity = pos_similarity / self.temperature
            neg_similarity = neg_similarity / self.temperature
            
            # Compute contrastive loss
            if neg_similarity.dim() == 2:
                # Multiple negatives per query
                all_similarities = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)
                labels = torch.zeros(all_similarities.shape[0], dtype=torch.long, device=all_similarities.device)
                contrastive_loss = F.cross_entropy(all_similarities, labels)
            else:
                # One negative per query
                all_similarities = torch.stack([pos_similarity, neg_similarity], dim=1)
                labels = torch.zeros(all_similarities.shape[0], dtype=torch.long, device=all_similarities.device)
                contrastive_loss = F.cross_entropy(all_similarities, labels)
        else:
            # Only positive similarity available, use MSE loss to increase it
            contrastive_loss = F.mse_loss(pos_similarity, torch.ones_like(pos_similarity))
        
        losses["contrastive_loss"] = contrastive_loss
        losses["total_loss"] = contrastive_loss
        
        # Add similarity metrics
        if neg_image_features is not None:
            losses["pos_similarity"] = pos_similarity.mean()
            losses["neg_similarity"] = neg_similarity.mean() if neg_similarity.dim() == 1 else neg_similarity.mean(dim=1).mean()
        
        # Add orthogonality metric (for monitoring purposes only)
        batch_size = personalization_matrix.shape[0]
        identity = torch.eye(personalization_matrix.shape[1], device=personalization_matrix.device)
        identity = identity.unsqueeze(0).expand_as(personalization_matrix)
        
        # Compute orthogonality metric (how close W^T W is to identity)
        WtW = torch.bmm(personalization_matrix.transpose(1, 2), personalization_matrix)
        ortho_metric = F.mse_loss(WtW, identity)
        losses["orthogonality"] = ortho_metric
        
        return losses
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Index of the batch
            
        Returns:
            Loss tensor
        """
        outputs = self.forward(batch)
        losses = self.compute_loss(outputs)
        
        # Log losses
        self.log("train_loss", losses["total_loss"], prog_bar=True)
        self.log("train_contrastive_loss", losses["contrastive_loss"])
        self.log("train_orthogonality", losses["orthogonality"])
        
        if "pos_similarity" in losses:
            self.log("train_pos_sim", losses["pos_similarity"])
            self.log("train_neg_sim", losses["neg_similarity"])
        
        return losses["total_loss"]
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        Validation step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Index of the batch
        """
        outputs = self.forward(batch)
        losses = self.compute_loss(outputs)
        
        # Log losses
        self.log("val_loss", losses["total_loss"], prog_bar=True)
        self.log("val_contrastive_loss", losses["contrastive_loss"])
        self.log("val_orthogonality", losses["orthogonality"])
        
        if "pos_similarity" in losses:
            self.log("val_pos_sim", losses["pos_similarity"])
            self.log("val_neg_sim", losses["neg_similarity"])
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer for training.
        
        Returns:
            Configured optimizer
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
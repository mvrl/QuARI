import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from feature_extractors import FeatureExtractorFactory
from transformer_hypernetwork import TransformerHypernetwork, ColumnWiseTransformerHypernetwork


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
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-2,
        temperature: float = 0.07,
        use_residual: bool = False,
        freeze_extractors: bool = True,
        # Transformer hypernetwork parameters
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dropout: float = 0.1,
        # Diffusion parameters
        scale_factor: float = 0.1,
        # Learning rate schedule parameters
        warmup_pct: float = 0.1,
        min_lr_factor: float = 0.05,
        # Column-wise transformer parameters
        column_wise: bool = True,
        matrix_sequence_len: int = 4,
        low_rank_dim: int = 64,
        using_precomputed_features: bool = True
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
            warmup_pct: Percentage of training steps for warmup
            min_lr_factor: Minimum learning rate as a factor of max learning rate
            column_wise: Whether to use column-wise transformer
            matrix_sequence_len: Length of sequence for matrix representation
            low_rank_dim: Dimension for low-rank matrix factorization
        """
        super().__init__()
        self.save_hyperparameters()
        

        if not using_precomputed_features:
            # Create feature extractor
            self.feature_extractor = FeatureExtractorFactory.create_extractor(
                extractor_type=extractor_type,
                model_name=extractor_model,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.feature_extractor.model.to(self.device)
        else:
            # Use precomputed features
            self.feature_extractor = None
        
        # Update embedding dimension based on feature extractor
        self.embedding_dim = self.feature_extractor.feature_dim if self.feature_extractor else 768
        
        # Create transformer hypernetwork for personalized transformation
        # Use column-wise transformer if specified
        if column_wise:
            print("Using column-wise transformer hypernetwork")
            self.hypernetwork = ColumnWiseTransformerHypernetwork(
                embedding_dim=self.embedding_dim,
                hidden_dim=hidden_dim,
                num_denoising_steps=num_denoising_steps,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dropout=dropout,
                use_residual=use_residual,
                scale_factor=scale_factor,
                matrix_sequence_len=matrix_sequence_len,
                low_rank_dim=low_rank_dim
            )
        else:
            print("Using row-wise transformer hypernetwork")
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
        if freeze_extractors and self.feature_extractor is not None:
            self.feature_extractor.model.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                
        # Temperature parameter for contrastive loss
        self.temperature = temperature
        
        # Learning rate schedule parameters
        self.warmup_pct = warmup_pct
        self.min_lr_factor = min_lr_factor
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            batch: Batch dictionary containing 'query_text', 'target_image'
            
        Returns:
            Dictionary containing computed embeddings and matrices
        """

        if "negative_images" in batch:
            raise ValueError("wrong")

        # Extract features
        if "text_features" in batch and "target_image_features" in batch:
            # Use precomputed features if available
            text_features = batch["text_features"]
            target_image_features = batch["target_image_features"]
        else:
            # Extract features from raw inputs
            query_text = batch["query_text"]
            target_image = batch["target_image"]
            
            # Extract features
            text_features = self.feature_extractor.extract_text_features(query_text)
            target_image_features = self.feature_extractor.extract_image_features(query_image)
        
        
        # Generate personalization matrix
        personalization_matrix = self.hypernetwork(
            text_features, return_all_steps=False
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


        #compute personalized image features
        personalized_image_features = torch.bmm(target_image_features.unsqueeze(1), personalization_matrix).squeeze(1)
        personalized_image_features = F.normalize(personalized_image_features, dim=-1)

        return {
            "personalized_text": personalized_text,
            "personalized_image_features": personalized_image_features,
            "personalization_matrix": personalization_matrix
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss for the model.
        
        Args:
            outputs: Outputs from the forward pass
            
        Returns:
            Dictionary of computed losses
        """

        personalized_text_features = outputs["personalized_text"]
        personalized_image_features = outputs["personalized_image_features"]
        personalization_matrix = outputs["personalization_matrix"]
        
        losses = {}

        # Compute contrastive loss
        batch_size = personalized_text_features.shape[0]
        logits = torch.matmul(personalized_text_features, personalized_image_features.t()) / self.temperature
        labels = torch.arange(batch_size, device=logits.device)
        
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.t(), labels)
        
        contrastive_loss = (loss_t2i + loss_i2t) / 2.0
        losses["contrastive_loss"] = contrastive_loss
        losses["total_loss"] = contrastive_loss
        
        # Compute orthogonality loss for monitoring
        batch_size = personalization_matrix.shape[0]
        identity = torch.eye(personalization_matrix.shape[1], device=personalization_matrix.device)
        identity = identity.unsqueeze(0).expand_as(personalization_matrix)
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
        
        # Log learning rate
        if self.trainer.is_global_zero:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log("learning_rate", lr, prog_bar=True)
        
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
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Get total number of training steps
        if self.trainer.max_steps > 0:
            max_steps = self.trainer.max_steps
        else:
            # Calculate from epochs
            if hasattr(self.trainer, 'estimated_stepping_batches'):
                max_steps = self.trainer.estimated_stepping_batches
            else:
                # Fallback
                max_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
        
        # Create a PyTorch built-in scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=max_steps,
            pct_start=self.warmup_pct,
            div_factor=25,
            final_div_factor=1/(self.min_lr_factor),
            three_phase=False
        )
        
        # Return the configuration with proper scheduler dict
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr"
            }
        }
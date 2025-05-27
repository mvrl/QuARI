import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from feature_extractors import FeatureExtractorFactory, BaseFeatureExtractor
from transformer_hypernetwork import ColumnWiseTransformerHypernetwork


class PersonalizedRetrievalModule(pl.LightningModule):
    """
    PyTorch Lightning module for training and evaluating the personalized text-to-image retrieval system.
    """
    def __init__(
        self,
        feature_extractor: BaseFeatureExtractor,
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
        # Learning rate schedule parameters
        warmup_pct: float = 0.01,
        min_lr_factor: float = 0.05,
        # Column-wise transformer parameters
        low_rank_dim: int = 64,
        using_precomputed_features: bool = True,
        use_separate_decoders: bool = True,
        train_noise_scale: float = 1.0,
    ):
        """
        Initialize the personalized retrieval module.
        
        Args:
            extractor_type: Type of feature extractor ('clip' or 'siglip')
            feature_extractor: Model name for the feature extractor
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
            low_rank_dim: Dimension for low-rank matrix factorization
        """

        super().__init__()
        self.save_hyperparameters()

        self.backbone_extractor = feature_extractor
        # Temperature parameter for contrastive loss
        self.temperature = temperature
        
        # Learning rate schedule parameters
        self.warmup_pct = warmup_pct
        self.min_lr_factor = min_lr_factor

        self.using_precomputed_features = using_precomputed_features
        self.low_rank_dim = low_rank_dim
        self.num_denoising_steps = num_denoising_steps
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.freeze_extractors = freeze_extractors
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.use_separate_decoders = use_separate_decoders
        
        if self.using_precomputed_features:
            # Forcibly send to CPU since it isn't used so it can be stored
            self.backbone_extractor.model.to("cpu")
        else:
            self.backbone_extractor.model.to(self.device)
        
        # Update embedding dimension based on feature extractor
        self.embedding_dim = self.backbone_extractor.feature_dim
        
        # Create transformer hypernetwork for personalized transformation
        # Use column-wise transformer if specified

        self.hypernetwork = ColumnWiseTransformerHypernetwork(
                                embedding_dim=self.embedding_dim,
                                low_rank_dim = low_rank_dim,
                                hidden_dim=hidden_dim,
                                num_steps=num_denoising_steps,
                                nhead=nhead,
                                num_layers=num_encoder_layers,
                                dropout=dropout,
                                use_separate_decoders=use_separate_decoders
                            )

        # Freeze feature extractors if specified
        if freeze_extractors and self.backbone_extractor is not None:
            self.backbone_extractor.model.eval()
            for param in self.backbone_extractor.parameters():
                param.requires_grad = False
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            batch: Batch dictionary containing 'query_text', 'target_image'
            
        Returns:
            Dictionary containing computed embeddings and matrices
        """
        
        text_features = batch["text_features"]
        target_image_features = batch["target_image_features"]

        #check for nan or inf
        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            raise ValueError("Text features contain NaN or Inf values.")

        # Generate personalization matrix
        outputs = self.hypernetwork(
            text_features, return_all=False
        )

        W_text = outputs["W_text"]
        W_image = outputs["W_image"]


        # Check for NaN or Inf in personalization matrices
        if torch.isnan(W_text).any() or torch.isinf(W_text).any():
            raise ValueError("W_text contains NaN or Inf values.")
        if torch.isnan(W_image).any() or torch.isinf(W_image).any():
            raise ValueError("W_image contains NaN or Inf values.")
        
        # Apply personalization to text features
        if self.use_residual:
            personalized_text = text_features + torch.bmm(
                text_features.unsqueeze(1),
                W_text
            ).squeeze(1)
        else:
            personalized_text = torch.bmm(
                text_features.unsqueeze(1),
                W_text
            ).squeeze(1)
        
        # Normalize
        personalized_text = F.normalize(personalized_text, dim=-1)

        #compute personalized image features
        personalized_image_features = torch.bmm(target_image_features.unsqueeze(1), W_image).squeeze(1)
        personalized_image_features = F.normalize(personalized_image_features, dim=-1)

        return {
            "personalized_text": personalized_text,
            "personalized_image_features": personalized_image_features,
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
        
        losses = {}

        # Compute contrastive loss
        batch_size = personalized_text_features.shape[0]
        logits = torch.matmul(personalized_text_features, personalized_image_features.t()) / self.temperature
        labels = torch.arange(batch_size, device=logits.device)
        
        # Text-to-image loss
        loss_text_to_image = F.cross_entropy(logits, labels)

        # Image-to-text loss
        loss_image_to_text = F.cross_entropy(logits.t(), labels)

        # Total contrastive loss (symmetric)
        contrastive_loss = (loss_text_to_image + loss_image_to_text) / 2

        
        losses["contrastive_loss"] = contrastive_loss
        losses["total_loss"] = contrastive_loss
        
        # # Compute orthogonality loss for monitoring
        # batch_size = personalization_matrix.shape[0]
        # identity = torch.eye(personalization_matrix.shape[1], device=personalization_matrix.device)
        # identity = identity.unsqueeze(0).expand_as(personalization_matrix)
        # WtW = torch.bmm(personalization_matrix.transpose(1, 2), personalization_matrix)
        # ortho_metric = F.mse_loss(WtW, identity)
        # losses["orthogonality"] = ortho_metric
        
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

        # Extract features
        if not ("text_features" in batch and "target_image_features" in batch):
            query_text = batch["query_text"]
            target_image = batch["target_image"]
            
            # Extract features
            batch["text_features"] = self.backbone_extractor.extract_text_features(query_text)
            batch["target_image_features"] = self.backbone_extractor.extract_image_features(query_image)

        # Add noise. From LinCIR paper: https://github.com/navervision/lincir/tree/master
        batch["text_features"] = batch["text_features"] + \
            self.train_noise_scale * torch.randn_like(batch["text_features"]) * torch.randlike(batch["text_features"])

        outputs = self.forward(batch)
        losses = self.compute_loss(outputs)
        
        # Log losses
        self.log("train_loss", losses["total_loss"], prog_bar=True)
        # self.log("train_contrastive_loss", losses["contrastive_loss"])
        # self.log("train_orthogonality", losses["orthogonality"])
        
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

        # Extract features
        if not ("text_features" in batch and "target_image_features" in batch):
            query_text = batch["query_text"]
            target_image = batch["target_image"]
            
            # Extract features
            batch["text_features"] = self.backbone_extractor.extract_text_features(query_text)
            batch["target_image_features"] = self.backbone_extractor.extract_image_features(query_image)

        outputs = self.forward(batch)
        losses = self.compute_loss(outputs)
        
        # Log losses
        self.log("val_loss", losses["total_loss"], prog_bar=True)
    
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
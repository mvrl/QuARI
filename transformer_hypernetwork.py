import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Any, Union


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    From original transformer paper.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
            
        Returns:
            Positional encoding added to input
        """
        return x + self.pe[:x.size(0)]


class TransformerHypernetwork(nn.Module):
    """
    Transformer-based hypernetwork that generates a personalized transformation matrix
    through an iterative denoising process.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 768,
        num_denoising_steps: int = 4,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dropout: float = 0.1,
        use_residual: bool = False,
        scale_factor: float = 0.1,
        matrix_sequence_len: int = 64,  # Controls how we reshape the matrix into a sequence
    ):
        """
        Initialize the transformer hypernetwork.
        
        Args:
            embedding_dim: Dimension of text/image embeddings
            hidden_dim: Hidden dimension for transformer
            num_denoising_steps: Number of denoising steps
            nhead: Number of attention heads in transformer
            num_encoder_layers: Number of transformer encoder layers
            dropout: Dropout rate
            use_residual: Whether to use residual connection in final output
            scale_factor: Scale factor for noise and updates
            matrix_sequence_len: Length of sequence for matrix reshaping
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_denoising_steps
        self.use_residual = use_residual
        self.scale_factor = scale_factor
        self.matrix_sequence_len = matrix_sequence_len
        
        # Calculate matrix item dim (each matrix element is represented as a vector)
        self.matrix_item_dim = hidden_dim
        
        # Query encoder (text features → context embedding)
        self.query_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Step embedding
        self.step_embedding = nn.Embedding(num_denoising_steps + 1, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Matrix projection layers
        self.matrix_to_sequence = nn.Linear(embedding_dim, matrix_sequence_len * self.matrix_item_dim)
        self.sequence_to_matrix = nn.Linear(self.matrix_item_dim, embedding_dim)
        
        # Noise scheduler (similar to diffusion models)
        # Beta schedule for gradually decreasing noise level
        self.register_buffer(
            'betas', 
            torch.linspace(0.1, 0.01, num_denoising_steps)
        )
        alphas = 1.0 - self.betas
        self.register_buffer(
            'alphas_cumprod', 
            torch.cumprod(alphas, dim=0)
        )
    
    def _reshape_matrix_to_sequence(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Reshape a batch of matrices into a sequence for transformer processing.
        
        Args:
            matrix: Tensor of shape [batch_size, embedding_dim, embedding_dim]
            
        Returns:
            Sequence tensor of shape [batch_size, matrix_sequence_len, matrix_item_dim]
        """
        batch_size = matrix.shape[0]
        
        # Project each row of the matrix to a sequence
        reshaped = []
        for i in range(self.embedding_dim):
            # Take the i-th row from each matrix in the batch
            row = matrix[:, i, :]  # [batch_size, embedding_dim]
            # Project to sequence representation
            projected = self.matrix_to_sequence(row)  # [batch_size, matrix_sequence_len * matrix_item_dim]
            # Reshape to sequence
            projected = projected.view(batch_size, self.matrix_sequence_len, self.matrix_item_dim)
            reshaped.append(projected)
        
        # Stack along new dimension
        sequence = torch.stack(reshaped, dim=1)  # [batch_size, embedding_dim, matrix_sequence_len, matrix_item_dim]
        
        # Reshape to [batch_size, embedding_dim * matrix_sequence_len, matrix_item_dim]
        sequence = sequence.view(batch_size, self.embedding_dim * self.matrix_sequence_len, self.matrix_item_dim)
        
        return sequence
    
    def _reshape_sequence_to_matrix(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Reshape a sequence back to a batch of matrices.
        
        Args:
            sequence: Tensor of shape [batch_size, embedding_dim * matrix_sequence_len, matrix_item_dim]
            
        Returns:
            Matrix tensor of shape [batch_size, embedding_dim, embedding_dim]
        """
        batch_size = sequence.shape[0]
        
        # Reshape to [batch_size, embedding_dim, matrix_sequence_len, matrix_item_dim]
        reshaped = sequence.view(batch_size, self.embedding_dim, self.matrix_sequence_len, self.matrix_item_dim)
        
        # Process each row
        matrix_rows = []
        for i in range(self.embedding_dim):
            # Get the sequence for row i
            row_seq = reshaped[:, i]  # [batch_size, matrix_sequence_len, matrix_item_dim]
            # Project back to embedding_dim
            row = self.sequence_to_matrix(row_seq)  # [batch_size, matrix_sequence_len, embedding_dim]
            # Average across sequence dimension
            row = row.mean(dim=1)  # [batch_size, embedding_dim]
            matrix_rows.append(row)
        
        # Stack rows to form matrices
        matrices = torch.stack(matrix_rows, dim=1)  # [batch_size, embedding_dim, embedding_dim]
        
        return matrices
    
    def _add_noise(self, matrices: torch.Tensor, step_idx: int) -> torch.Tensor:
        """
        Add noise to matrices based on diffusion timestep.
        
        Args:
            matrices: Tensor of shape [batch_size, embedding_dim, embedding_dim]
            step_idx: Current denoising step index
            
        Returns:
            Noised matrices
        """
        batch_size = matrices.shape[0]
        device = matrices.device
        
        # Get noise scale for this step
        alpha_cumprod = self.alphas_cumprod[step_idx]
        
        # Generate noise
        noise = torch.randn_like(matrices) * self.scale_factor
        
        # Mix original and noise based on schedule
        noised_matrices = torch.sqrt(alpha_cumprod) * matrices + torch.sqrt(1 - alpha_cumprod) * noise
        
        return noised_matrices
    
    def forward(
        self, 
        query_emb: torch.Tensor,
        initial_matrix: Optional[torch.Tensor] = None,
        return_all_steps: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate a personalized transformation matrix through iterative denoising.
        
        Args:
            query_emb: Query embedding tensor of shape [batch_size, embedding_dim]
            initial_matrix: Optional initial matrix of shape [batch_size, embedding_dim, embedding_dim]
                           If None, an identity matrix will be used
            return_all_steps: Whether to return all intermediate matrices
            
        Returns:
            Either:
             - Final transformation matrix of shape [batch_size, embedding_dim, embedding_dim]
             - Tuple of (final_matrix, list_of_all_matrices)
        """
        batch_size = query_emb.shape[0]
        device = query_emb.device
        
        # Encode query context
        query_context = self.query_encoder(query_emb)  # [batch_size, hidden_dim]
        
        # Initialize with identity if no initial matrix provided
        if initial_matrix is None:
            current_matrix = torch.eye(self.embedding_dim, device=device)
            current_matrix = current_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            current_matrix = initial_matrix
        
        # Store intermediate matrices
        all_matrices = [current_matrix.clone()]
        
        # Iterative denoising
        for step in range(self.num_steps, 0, -1):
            # Add noise to current matrix (reverse diffusion process)
            noised_matrix = self._add_noise(current_matrix, step-1)
            
            # Reshape matrix to sequence for transformer
            sequence = self._reshape_matrix_to_sequence(noised_matrix)  # [batch_size, seq_len, hidden_dim]
            
            # Get step embedding
            step_emb = self.step_embedding(torch.tensor([step-1], device=device))  # [1, hidden_dim]
            step_emb = step_emb.expand(batch_size, -1)  # [batch_size, hidden_dim]
            
            # Create conditioning by adding query context and step embedding
            # Prepend as a special token
            cond_token = query_context + step_emb  # [batch_size, hidden_dim]
            cond_token = cond_token.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Concatenate conditioning token with sequence
            sequence_with_cond = torch.cat([cond_token, sequence], dim=1)  # [batch_size, seq_len+1, hidden_dim]
            
            # Apply transformer encoder
            output_sequence = self.transformer_encoder(sequence_with_cond)  # [batch_size, seq_len+1, hidden_dim]
            
            # Remove conditioning token
            output_sequence = output_sequence[:, 1:, :]  # [batch_size, seq_len, hidden_dim]
            
            # Reshape back to matrix
            output_matrix = self._reshape_sequence_to_matrix(output_sequence)  # [batch_size, embedding_dim, embedding_dim]
            
            # Update current matrix
            current_matrix = output_matrix
            all_matrices.append(current_matrix.clone())
        
        # Return the final matrix and optionally all intermediate matrices
        if return_all_steps:
            return current_matrix, all_matrices
        else:
            return current_matrix
    
    def apply_personalization(self, query_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply the personalized transformation to the query embedding and compute similarity with image.
        
        Args:
            query_emb: Query embedding of shape [batch_size, embedding_dim]
            image_emb: Image embedding of shape [batch_size, num_images, embedding_dim] or [batch_size, embedding_dim]
            
        Returns:
            Similarity scores of shape [batch_size, num_images] or [batch_size]
        """
        # Generate transformation matrix
        W = self.forward(query_emb)
        
        # Apply transformation with optional residual connection
        if self.use_residual:
            personalized_query = query_emb + torch.bmm(
                query_emb.unsqueeze(1), 
                W
            ).squeeze(1)
        else:
            # Direct transformation without residual
            personalized_query = torch.bmm(
                query_emb.unsqueeze(1), 
                W
            ).squeeze(1)
        
        # Normalize
        personalized_query = F.normalize(personalized_query, dim=-1)
        
        # Reshape for batch matrix multiplication if needed
        if image_emb.dim() == 3:
            # Multiple images per query
            sim = torch.bmm(
                personalized_query.unsqueeze(1),
                image_emb.transpose(1, 2)
            ).squeeze(1)
        else:
            # One image per query
            sim = F.cosine_similarity(personalized_query, image_emb, dim=1)
            
        return sim
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


class ColumnWiseTransformerHypernetwork(nn.Module):
    """
    Column-wise transformer-based hypernetwork that generates a personalized transformation matrix
    through an iterative denoising process. Processes the matrix column by column to better respect
    the semantic structure of embedding transformations.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        num_denoising_steps: int = 4,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = False,
        scale_factor: float = 0.1,
        matrix_sequence_len: int = 16,  # Tokens per column
        low_rank_dim: int = 64,         # Dimension for low-rank factorization
    ):
        """
        Initialize the column-wise transformer hypernetwork.
        
        Args:
            embedding_dim: Dimension of text/image embeddings
            hidden_dim: Hidden dimension for transformer
            num_denoising_steps: Number of denoising steps
            nhead: Number of attention heads in transformer
            num_encoder_layers: Number of transformer encoder layers
            dropout: Dropout rate
            use_residual: Whether to use residual connection in final output
            scale_factor: Scale factor for noise and updates
            matrix_sequence_len: Length of sequence for column encoding
            low_rank_dim: Dimension for low-rank matrix factorization
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_denoising_steps
        self.use_residual = use_residual
        self.scale_factor = scale_factor
        self.matrix_sequence_len = matrix_sequence_len
        self.low_rank_dim = low_rank_dim
        
        # Matrix item dim (each matrix element is represented as a vector)
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
        
        # COLUMN-WISE PROCESSING: Project columns to sequence tokens
        # We'll project each column of U and V (which are embedding_dim-dimensional)
        self.column_to_sequence = nn.Linear(embedding_dim, matrix_sequence_len * hidden_dim)
        
        # Project from sequence back to column vectors
        self.sequence_to_u_column = nn.Linear(hidden_dim, embedding_dim)
        self.sequence_to_v_column = nn.Linear(hidden_dim, embedding_dim)
        
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
    
    def _factors_to_matrix(self, u_factors: torch.Tensor, v_factors: torch.Tensor) -> torch.Tensor:
        """
        Construct a matrix from its low-rank factors U and V.
        
        Args:
            u_factors: Tensor of shape [batch_size, embedding_dim, low_rank_dim]
            v_factors: Tensor of shape [batch_size, embedding_dim, low_rank_dim]
            
        Returns:
            Matrix tensor of shape [batch_size, embedding_dim, embedding_dim]
        """
        # W = U @ V.T
        matrices = torch.bmm(u_factors, v_factors.transpose(1, 2))
        return matrices
    
    def _add_noise(self, u_factors: torch.Tensor, v_factors: torch.Tensor, step_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to matrix factors based on diffusion timestep.
        
        Args:
            u_factors: Tensor of shape [batch_size, embedding_dim, low_rank_dim]
            v_factors: Tensor of shape [batch_size, embedding_dim, low_rank_dim]
            step_idx: Current denoising step index
            
        Returns:
            Noised matrix factors
        """
        # Get noise scale for this step
        alpha_cumprod = self.alphas_cumprod[step_idx]
        
        # Generate noise for both factors
        u_noise = torch.randn_like(u_factors) * self.scale_factor
        v_noise = torch.randn_like(v_factors) * self.scale_factor
        
        # Mix original and noise based on schedule
        noised_u = torch.sqrt(alpha_cumprod) * u_factors + torch.sqrt(1 - alpha_cumprod) * u_noise
        noised_v = torch.sqrt(alpha_cumprod) * v_factors + torch.sqrt(1 - alpha_cumprod) * v_noise
        
        return noised_u, noised_v
    
    def forward(
        self, 
        query_emb: torch.Tensor,
        initial_matrix: Optional[torch.Tensor] = None,
        return_all_steps: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate a personalized transformation matrix through iterative denoising,
        processing the matrix column by column.
        
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
        
        # Initialize matrix factors
        if initial_matrix is None:
            # Initialize with scaled identity matrix factors
            # For identity matrix, both U and V are diagonal matrices with sqrt(1/low_rank_dim) on the diagonal
            scale = math.sqrt(1.0 / self.low_rank_dim)
            u_factors = torch.zeros(batch_size, self.embedding_dim, self.low_rank_dim, device=device)
            v_factors = torch.zeros(batch_size, self.embedding_dim, self.low_rank_dim, device=device)
            
            # Set diagonal elements with a small value
            for i in range(min(self.embedding_dim, self.low_rank_dim)):
                u_factors[:, i, i] = scale
                v_factors[:, i, i] = scale
        else:
            # If an initial matrix is provided, decompose it with SVD for the low-rank factors
            # This is a simplified version, proper implementation would use SVD
            u_factors = torch.randn(batch_size, self.embedding_dim, self.low_rank_dim, device=device) * 0.01
            v_factors = torch.randn(batch_size, self.embedding_dim, self.low_rank_dim, device=device) * 0.01
        
        # Store intermediate matrices
        all_matrices = [self._factors_to_matrix(u_factors, v_factors).clone()] if return_all_steps else []
        
        # Iterative denoising
        for step in range(self.num_steps, 0, -1):
            # Add noise to current matrix factors (reverse diffusion process)
            noised_u, noised_v = self._add_noise(u_factors, v_factors, step-1)
            
            # Get step embedding
            step_emb = self.step_embedding(torch.tensor([step-1], device=device))  # [1, hidden_dim]
            step_emb = step_emb.expand(batch_size, -1)  # [batch_size, hidden_dim]
            
            # Create conditioning token with query context and step embedding
            cond_token = query_context + step_emb  # [batch_size, hidden_dim]
            cond_token = cond_token.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Process in chunks of columns to reduce memory footprint
            # COLUMN-WISE PROCESSING: This is the key difference
            chunk_size = 16  # Process 16 columns at a time
            num_chunks = (self.low_rank_dim + chunk_size - 1) // chunk_size  # Ceiling division
            
            new_u_columns = []
            new_v_columns = []
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, self.low_rank_dim)
                chunk_len = end_idx - start_idx
                
                if chunk_len <= 0:
                    continue
                
                # Extract chunks of the matrix factors - now we extract columns
                # We transpose the factors to get a shape that's easier to work with
                # [batch_size, low_rank_dim(chunk), embedding_dim]
                u_chunk_t = noised_u[:, :, start_idx:end_idx].transpose(1, 2)
                v_chunk_t = noised_v[:, :, start_idx:end_idx].transpose(1, 2)
                
                # Process each column in the chunk
                u_column_tokens = []
                v_column_tokens = []
                
                for i in range(chunk_len):
                    # Get column i from each factor
                    # These are embedding_dim-dimensional vectors
                    u_column = u_chunk_t[:, i, :]  # [batch_size, embedding_dim]
                    v_column = v_chunk_t[:, i, :]  # [batch_size, embedding_dim]
                    
                    # Project each column to sequence tokens
                    u_tokens = self.column_to_sequence(u_column)  # [batch_size, matrix_sequence_len * hidden_dim]
                    u_tokens = u_tokens.view(batch_size, self.matrix_sequence_len, self.hidden_dim)
                    
                    v_tokens = self.column_to_sequence(v_column)
                    v_tokens = v_tokens.view(batch_size, self.matrix_sequence_len, self.hidden_dim)
                    
                    u_column_tokens.append(u_tokens)
                    v_column_tokens.append(v_tokens)
                
                # Process U columns in this chunk
                # Stack tokens for all columns
                u_sequence = torch.cat(u_column_tokens, dim=1)  # [batch_size, chunk_len*matrix_sequence_len, hidden_dim]
                
                # Prepend conditioning token
                sequence_with_cond = torch.cat([cond_token, u_sequence], dim=1)
                
                # Apply transformer encoder
                u_output_sequence = self.transformer_encoder(sequence_with_cond)
                
                # Remove conditioning token
                u_output_sequence = u_output_sequence[:, 1:, :]
                
                # Now do the same for V columns
                v_sequence = torch.cat(v_column_tokens, dim=1)
                
                # Prepend conditioning token
                sequence_with_cond = torch.cat([cond_token, v_sequence], dim=1)
                
                # Apply transformer encoder
                v_output_sequence = self.transformer_encoder(sequence_with_cond)
                
                # Remove conditioning token
                v_output_sequence = v_output_sequence[:, 1:, :]
                
                # Convert back to columns
                for i in range(chunk_len):
                    start = i * self.matrix_sequence_len
                    end = start + self.matrix_sequence_len
                    
                    # Extract sequence for this column
                    u_col_sequence = u_output_sequence[:, start:end, :]  # [batch_size, matrix_sequence_len, hidden_dim]
                    v_col_sequence = v_output_sequence[:, start:end, :]
                    
                    # Aggregate the sequence to a single vector (mean pooling)
                    u_col_embedding = u_col_sequence.mean(dim=1)  # [batch_size, hidden_dim]
                    v_col_embedding = v_col_sequence.mean(dim=1)
                    
                    # Project back to column vectors (embedding_dim-dimensional)
                    new_u_col = self.sequence_to_u_column(u_col_embedding)  # [batch_size, embedding_dim]
                    new_v_col = self.sequence_to_v_column(v_col_embedding)
                    
                    new_u_columns.append(new_u_col)
                    new_v_columns.append(new_v_col)
            
            # Stack all columns to form new factors
            # We need to transpose back to the original format [batch_size, embedding_dim, low_rank_dim]
            u_columns_stacked = torch.stack(new_u_columns, dim=1)  # [batch_size, low_rank_dim, embedding_dim]
            v_columns_stacked = torch.stack(new_v_columns, dim=1)
            
            u_factors = u_columns_stacked.transpose(1, 2)  # [batch_size, embedding_dim, low_rank_dim]
            v_factors = v_columns_stacked.transpose(1, 2)
            
            # Generate matrix from factors and store if needed
            if return_all_steps:
                current_matrix = self._factors_to_matrix(u_factors, v_factors)
                all_matrices.append(current_matrix.clone())
        
        # Generate final matrix from factors
        final_matrix = self._factors_to_matrix(u_factors, v_factors)
        
        # Return the final matrix and optionally all intermediate matrices
        if return_all_steps:
            return final_matrix, all_matrices
        else:
            return final_matrix
    
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

#######
#LEGACY
#######
class TransformerHypernetwork(nn.Module):
    """
    Memory-optimized transformer-based hypernetwork that generates a personalized transformation matrix
    through an iterative denoising process.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_denoising_steps: int,
        nhead: int,
        num_encoder_layers: int,
        dropout: float,
        use_residual: bool,
        scale_factor: float,
        matrix_sequence_len: int,
        low_rank_dim: int,
    ):
        """
        Initialize the memory-optimized transformer hypernetwork.
        
        Args:
            embedding_dim: Dimension of text/image embeddings
            hidden_dim: Hidden dimension for transformer
            num_denoising_steps: Number of denoising steps
            nhead: Number of attention heads in transformer
            num_encoder_layers: Number of transformer encoder layers
            dropout: Dropout rate
            use_residual: Whether to use residual connection in final output
            scale_factor: Scale factor for noise and updates
            matrix_sequence_len: Length of sequence for matrix reshaping (REDUCED)
            low_rank_dim: Dimension for low-rank matrix factorization (NEW)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_denoising_steps
        self.use_residual = use_residual
        self.scale_factor = scale_factor
        self.matrix_sequence_len = matrix_sequence_len
        self.low_rank_dim = low_rank_dim
        
        # Matrix item dim (each matrix element is represented as a vector)
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
        
        # MEMORY OPTIMIZATION: Use low-rank parameterization instead of full matrix
        # Instead of projecting to a full matrix, we'll project to two low-rank factors
        # Matrix W = U @ V.T where U, V are both of shape [embedding_dim, low_rank_dim]
        
        # Matrix projection layers (OPTIMIZED)
        # Project to sequence tokens that will generate U and V factors
        self.to_sequence = nn.Linear(embedding_dim, matrix_sequence_len * hidden_dim)
        
        # Project from sequence back to U and V factors
        self.u_projector = nn.Linear(hidden_dim, low_rank_dim)
        self.v_projector = nn.Linear(hidden_dim, low_rank_dim)
        
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
    
    def _factors_to_matrix(self, u_factors: torch.Tensor, v_factors: torch.Tensor) -> torch.Tensor:
        """
        Construct a matrix from its low-rank factors U and V.
        
        Args:
            u_factors: Tensor of shape [batch_size, embedding_dim, low_rank_dim]
            v_factors: Tensor of shape [batch_size, embedding_dim, low_rank_dim]
            
        Returns:
            Matrix tensor of shape [batch_size, embedding_dim, embedding_dim]
        """
        # W = U @ V.T
        matrices = torch.bmm(u_factors, v_factors.transpose(1, 2))
        return matrices
    
    def _add_noise(self, u_factors: torch.Tensor, v_factors: torch.Tensor, step_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to matrix factors based on diffusion timestep.
        
        Args:
            u_factors: Tensor of shape [batch_size, embedding_dim, low_rank_dim]
            v_factors: Tensor of shape [batch_size, embedding_dim, low_rank_dim]
            step_idx: Current denoising step index
            
        Returns:
            Noised matrix factors
        """
        # Get noise scale for this step
        alpha_cumprod = self.alphas_cumprod[step_idx]
        
        # Generate noise for both factors
        u_noise = torch.randn_like(u_factors) * self.scale_factor
        v_noise = torch.randn_like(v_factors) * self.scale_factor
        
        # Mix original and noise based on schedule
        noised_u = torch.sqrt(alpha_cumprod) * u_factors + torch.sqrt(1 - alpha_cumprod) * u_noise
        noised_v = torch.sqrt(alpha_cumprod) * v_factors + torch.sqrt(1 - alpha_cumprod) * v_noise
        
        return noised_u, noised_v
    
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
        
        # Initialize matrix factors
        if initial_matrix is None:
            # Initialize with scaled identity matrix factors
            # For identity matrix, both U and V are diagonal matrices with sqrt(1/low_rank_dim) on the diagonal
            scale = math.sqrt(1.0 / self.low_rank_dim)
            u_factors = torch.zeros(batch_size, self.embedding_dim, self.low_rank_dim, device=device)
            v_factors = torch.zeros(batch_size, self.embedding_dim, self.low_rank_dim, device=device)
            
            # Set diagonal elements with a small value
            for i in range(min(self.embedding_dim, self.low_rank_dim)):
                u_factors[:, i, i] = scale
                v_factors[:, i, i] = scale
        else:
            # If an initial matrix is provided, decompose it with SVD
            # This is a simplified version, proper implementation would use SVD
            # This is just a placeholder
            # TODO: Implement proper matrix factorization if initial matrix is provided
            u_factors = torch.randn(batch_size, self.embedding_dim, self.low_rank_dim, device=device) * 0.01
            v_factors = torch.randn(batch_size, self.embedding_dim, self.low_rank_dim, device=device) * 0.01
        
        # Store intermediate matrices
        all_matrices = [self._factors_to_matrix(u_factors, v_factors).clone()] if return_all_steps else []
        
        # Iterative denoising
        for step in range(self.num_steps, 0, -1):
            # Add noise to current matrix factors (reverse diffusion process)
            noised_u, noised_v = self._add_noise(u_factors, v_factors, step-1)
            
            # MEMORY-EFFICIENT APPROACH: Process each chunk of the matrix separately
            # Instead of reshaping the entire matrix at once, we'll process it in chunks
            
            # Get step embedding
            step_emb = self.step_embedding(torch.tensor([step-1], device=device))  # [1, hidden_dim]
            step_emb = step_emb.expand(batch_size, -1)  # [batch_size, hidden_dim]
            
            # Create conditioning token with query context and step embedding
            cond_token = query_context + step_emb  # [batch_size, hidden_dim]
            cond_token = cond_token.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Process in chunks to reduce memory footprint
            chunk_size = 32  # Process 32 rows at a time
            num_chunks = (self.embedding_dim + chunk_size - 1) // chunk_size  # Ceiling division
            
            new_u_rows = []
            new_v_rows = []
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, self.embedding_dim)
                chunk_len = end_idx - start_idx
                
                if chunk_len <= 0:
                    continue
                
                # Extract chunks of the matrix factors
                u_chunk = noised_u[:, start_idx:end_idx, :]  # [batch_size, chunk_len, low_rank_dim]
                v_chunk = noised_v[:, start_idx:end_idx, :]  # [batch_size, chunk_len, low_rank_dim]
                
                # Project chunks to sequence tokens
                u_tokens = []
                v_tokens = []
                
                for i in range(chunk_len):
                    # Project row i from each factor in the batch
                    u_row = u_chunk[:, i, :]  # [batch_size, low_rank_dim]
                    v_row = v_chunk[:, i, :]  # [batch_size, low_rank_dim]
                    
                    # Combine the two factor rows and project to sequence tokens
                    combined = torch.cat([u_row, v_row], dim=1)  # [batch_size, 2*low_rank_dim]
                    
                    # Pad if needed
                    if combined.shape[1] < self.embedding_dim:
                        padding = torch.zeros(batch_size, self.embedding_dim - combined.shape[1], device=device)
                        combined = torch.cat([combined, padding], dim=1)
                    
                    # Project to sequence
                    tokens = self.to_sequence(combined)  # [batch_size, matrix_sequence_len * hidden_dim]
                    tokens = tokens.view(batch_size, self.matrix_sequence_len, self.hidden_dim)  # [batch_size, matrix_sequence_len, hidden_dim]
                    
                    u_tokens.append(tokens)
                    v_tokens.append(tokens)  # We'll use the same tokens but project them differently later
                
                # Stack tokens for all rows in the chunk
                u_sequence = torch.cat(u_tokens, dim=1)  # [batch_size, chunk_len*matrix_sequence_len, hidden_dim]
                
                # Prepend conditioning token
                sequence_with_cond = torch.cat([cond_token, u_sequence], dim=1)  # [batch_size, 1+chunk_len*matrix_sequence_len, hidden_dim]
                
                # Apply transformer encoder
                output_sequence = self.transformer_encoder(sequence_with_cond)  # [batch_size, 1+chunk_len*matrix_sequence_len, hidden_dim]
                
                # Remove conditioning token
                output_sequence = output_sequence[:, 1:, :]  # [batch_size, chunk_len*matrix_sequence_len, hidden_dim]
                
                # Reshape back and generate new U and V rows
                for i in range(chunk_len):
                    start = i * self.matrix_sequence_len
                    end = start + self.matrix_sequence_len
                    
                    # Extract sequence for this row
                    row_sequence = output_sequence[:, start:end, :]  # [batch_size, matrix_sequence_len, hidden_dim]
                    
                    # Pool the sequence for this row
                    row_embedding = row_sequence.mean(dim=1)  # [batch_size, hidden_dim]
                    
                    # Project to U and V factors
                    new_u_row = self.u_projector(row_embedding)  # [batch_size, low_rank_dim]
                    new_v_row = self.v_projector(row_embedding)  # [batch_size, low_rank_dim]
                    
                    new_u_rows.append(new_u_row)
                    new_v_rows.append(new_v_row)
            
            # Stack all rows to form new factors
            u_factors = torch.stack(new_u_rows, dim=1)  # [batch_size, embedding_dim, low_rank_dim]
            v_factors = torch.stack(new_v_rows, dim=1)  # [batch_size, embedding_dim, low_rank_dim]
            
            # Generate matrix from factors and store if needed
            if return_all_steps:
                current_matrix = self._factors_to_matrix(u_factors, v_factors)
                all_matrices.append(current_matrix.clone())
        
        # Generate final matrix from factors
        final_matrix = self._factors_to_matrix(u_factors, v_factors)
        
        # Return the final matrix and optionally all intermediate matrices
        if return_all_steps:
            return final_matrix, all_matrices
        else:
            return final_matrix
    
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
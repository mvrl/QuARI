import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple
import math
import yaml

def load_model(path):
    metadata_path = path + "config.yml"
    weights_path = path + "weights.pt"
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)
    model = TransformerHypernetwork(**metadata['hparams'])
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model, metadata

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings (from denoising diffusion models).
    
    Args:
        timesteps: 1D tensor of timestep indices [B] or scalar
        embedding_dim: Dimension of the embedding
        max_period: Maximum period for sinusoidal encoding
    
    Returns:
        Embedding tensor [B, embedding_dim]
    """
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def scaled_positional_encoding(batch_seq: torch.Tensor, pe: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Adds positional encoding to a batched sequence with optional scaling.
    
    Args:
        batch_seq: [batch, seq_len, hidden_dim]
        pe: [max_len, hidden_dim]
        scale: Scaling factor for positional encoding
    
    Returns:
        Sequence with positional encoding added
    """
    seq_len = batch_seq.size(1)
    return batch_seq + scale * pe[:seq_len].unsqueeze(0)


class TransformerHypernetwork(nn.Module):
    """
    Hypernetwork that:
    - Generates only W_image (more efficient for text-to-image retrieval)
    - Refines the query embedding through the condition token
    - Uses sinusoidal step embeddings for better gradient flow
    
    Architecture:
        Input: query_emb [B, E]
        ↓
        Query Encoder → ctx [B, H]
        ↓
        Iterative Refinement (T steps):
            - ctx refined through transformer
            - U/V tokens refined through transformer
        ↓
        Outputs:
            - refined_query [B, E]: Decoded from final ctx
            - W_image [B, E, E]: Low-rank matrix for image transformation
    """
    
    def __init__(
        self,
        embedding_dim: int,
        low_rank_dim: int = 64,
        hidden_dim: int = 512,
        num_steps: int = 4,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        query_residual_weight: float = 0.5,
        use_fro_norm: bool = True,
        anchor_scale: float = 0.8,
    ):
        """
        Args:
            embedding_dim: Dimension of input embeddings (e.g., 1024 for SigLIP2-large)
            low_rank_dim: Rank for low-rank factorization (default: 64)
            hidden_dim: Hidden dimension for transformer (default: 512)
            num_steps: Number of iterative refinement steps (default: 4)
            nhead: Number of attention heads (default: 8)
            num_layers: Number of transformer encoder layers (default: 4)
            dropout: Dropout rate (default: 0.1)
            query_residual_weight: Weight for residual connection in query refinement (default: 0.5)
        """
        super().__init__()
        self.E = embedding_dim
        self.r = low_rank_dim
        self.H = hidden_dim
        self.num_steps = num_steps
        self.query_residual_weight = query_residual_weight
        self.use_fro_norm = use_fro_norm
        self.anchor_scale = anchor_scale
        
        # Query encoder: Projects query embedding to hidden space
        self.query_encoder = nn.Sequential(
            nn.Linear(self.E, self.H),
            nn.LayerNorm(self.H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.H, self.H),
            nn.LayerNorm(self.H)
        )
        
        # Step embedding: Maps step index to hidden dimension
        # Using sinusoidal encoding for better gradient flow
        self.step_proj = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.GELU(),
            nn.Linear(self.H, self.H)
        )
        
        # Positional encodings for U/V tokens
        seq_len = 2 * self.r + 1  # condition + u_tokens + v_tokens
        pe = torch.zeros(seq_len, self.H)
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.H, 2, dtype=torch.float) * (-math.log(10000.0) / self.H))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_emb', pe)

        self.register_buffer('pos_scale', torch.ones(1) * 1.0) 
        
        # Transformer encoder: Refines all tokens through self-attention
        layer = nn.TransformerEncoderLayer(
            d_model=self.H,
            nhead=nhead,
            dim_feedforward=self.H * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable)
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        
        # Decoder for refined query embedding from condition token
        self.query_decoder = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.H, self.E)
        )
        
        # Decoders for U and V matrices (image transformation only)
        self.dec_u_img = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.H, self.E)
        )
        
        self.dec_v_img = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.H, self.E)
        )
    
    def forward(
        self,
        query_emb: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through the hypernetwork.
        
        Args:
            query_emb: Query embeddings [B, E]
            return_all: If True, return intermediate steps
        
        Returns:
            Dictionary containing:
                - 'refined_query': Refined query embedding [B, E]
                - 'W_image': Image transformation matrix [B, E, E]
                - 'all': (optional) List of intermediate outputs
        """
        B, E = query_emb.shape
        device = query_emb.device
        
        # Encode query to hidden space
        ctx = self.query_encoder(query_emb)  # [B, H]
        
        # Initialize U and V tokens with zeros
        u_tok = torch.zeros(B, self.r, self.H, device=device)
        v_tok = torch.zeros(B, self.r, self.H, device=device)

        # Store intermediate results if requested
        all_steps = [] if return_all else None
        if return_all:
            initial_query = self.query_decoder(ctx)
            initial_query = (1.0 - self.query_residual_weight) * initial_query + self.query_residual_weight * query_emb
            initial_query = F.normalize(initial_query, dim=-1)
            all_steps.append({
                'refined_query': initial_query,
                'W_image': torch.eye(E, device=device).unsqueeze(0).expand(B, -1, -1)
            })
        
        # Iterative refinement loop
        for t in range(self.num_steps):
            # Create step embedding (sinusoidal encoding)
            step_idx = torch.full((B,), t, device=device, dtype=torch.long)
            step_emb = get_timestep_embedding(step_idx, self.H)
            step_emb = self.step_proj(step_emb)
            
            # Update condition token with step information
            cond = (ctx + step_emb).unsqueeze(1)  # [B, 1, H]
            
            # Add positional encodings to U/V tokens with learned scaling
            u_seq = scaled_positional_encoding(u_tok, self.pos_emb, scale=self.pos_scale)
            v_seq = scaled_positional_encoding(v_tok, self.pos_emb, scale=self.pos_scale)
            
            # Build sequence: [cond, U-tokens, V-tokens]
            seq = torch.cat([cond, u_seq, v_seq], dim=1)  # [B, 1+2r, H]
            
            # Transform through transformer (all tokens attend to each other)
            out = self.transformer(seq)  # [B, 1+2r, H]
            
            # Extract outputs
            ctx_out = out[:, 0, :]  # Updated condition token
            delta = out[:, 1:, :]   # Updates for U/V tokens
            
            # Split deltas for U and V
            d_u = delta[:, :self.r, :]
            d_v = delta[:, self.r:, :]
            
            # Update tokens with residual connection
            ctx = ctx + ctx_out  # Refine context
            u_tok = u_tok + d_u  # Refine U tokens
            v_tok = v_tok + d_v  # Refine V tokens
            

            if self.training:
                u_dropout_mask = torch.rand(B, self.r, 1, device=device) > 0.05
                v_dropout_mask = torch.rand(B, self.r, 1, device=device) > 0.05
                u_tok = u_tok * u_dropout_mask.float()
                v_tok = v_tok * v_dropout_mask.float()
            
            # Store intermediate results if requested
            if return_all:
                query_refined = self.query_decoder(ctx)
                query_refined = (1.0 - self.query_residual_weight) * query_refined + self.query_residual_weight * query_emb
                query_refined = F.normalize(query_refined, dim=-1)
                
                W_img = self._decode_and_form_matrix(u_tok, v_tok)
                all_steps.append({
                    'refined_query': query_refined,
                    'W_image': W_img
                })
        
        # Final decoding
        query_delta = self.query_decoder(ctx)
        refined_query = (1.0 - self.query_residual_weight) * query_delta + self.query_residual_weight * query_emb
        refined_query = F.normalize(refined_query, dim=-1)  # Ensure unit norm
        
        # Form image transformation matrix from U/V tokens
        W_image = self._decode_and_form_matrix(u_tok, v_tok)
        
        # Prepare output
        output = {
            'refined_query': refined_query,
            'W_image': W_image
        }
        
        if return_all:
            output['all'] = all_steps
        
        return output
    
    def _decode_and_form_matrix(
        self,
        u_tok: torch.Tensor,
        v_tok: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode U/V tokens and form transformation matrix via low-rank factorization.

        Args:
            u_tok: U tokens [B, r, H]
            v_tok: V tokens [B, r, H]
        
        Returns:
            W_image: Transformation matrix [B, E, E]
        """
        # Decode tokens to embedding dimension
        u_cols = self.dec_u_img(u_tok)  # [B, r, E]
        v_cols = self.dec_v_img(v_tok)  # [B, r, E]
        
        # Transpose to [B, E, r]
        U = u_cols.transpose(1, 2)
        V = v_cols.transpose(1, 2)
        
        if self.use_fro_norm:
            U_norm = torch.norm(U, p='fro', dim=(1, 2), keepdim=True)
            V_norm = torch.norm(V, p='fro', dim=(1, 2), keepdim=True)
            
            U = U / (U_norm + 1e-8)
            V = V / (V_norm + 1e-8)
            
        # Form low-rank perturbation: Delta = U @ V^T
        Delta = torch.bmm(U, V.transpose(1, 2))  # [B, E, E]
        B, E, _ = Delta.shape

        I = torch.eye(E, device=Delta.device, dtype=Delta.dtype).unsqueeze(0).expand(B, -1, -1)
        W_image = I + self.anchor_scale * Delta
        
        return W_image
    
    def project_and_score(
        self,
        query_emb: torch.Tensor,
        img_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply transformations and compute similarity scores.
        
        Args:
            query_emb: Query embeddings [B, E] or [B, N, E]
            img_emb: Image embeddings [B, E] or [B, M, E]
        
        Returns:
            Similarity scores
        """
        # Forward pass
        out = self.forward(query_emb)
        
        # Get refined query and transformation matrix
        refined_query = out['refined_query']  # [B, E]
        W_image = out['W_image']              # [B, E, E]
        
        # Transform images
        if img_emb.dim() == 2:
            # Single image per query: [B, E]
            img_transformed = torch.bmm(
                img_emb.unsqueeze(1),
                W_image
            ).squeeze(1)  # [B, E]
            img_transformed = F.normalize(img_transformed, dim=-1)
            
            # Compute similarity
            similarity = F.cosine_similarity(refined_query, img_transformed, dim=-1)
        else:
            # Multiple images per query: [B, M, E]
            img_transformed = torch.bmm(img_emb, W_image)  # [B, M, E]
            img_transformed = F.normalize(img_transformed, dim=-1)
            
            # Compute similarity
            similarity = torch.einsum('be,bme->bm', refined_query, img_transformed)
        
        return similarity




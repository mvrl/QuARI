import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, List, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Any, Optional


def scaled_positional_encoding(batch_seq: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """
    Adds positional encoding to a batched sequence.
    batch_seq: [batch, seq_len, hidden_dim]
    pe: [max_len, hidden_dim]
    """
    seq_len = batch_seq.size(1)
    return batch_seq + pe[:seq_len].unsqueeze(0)


class ColumnWiseTransformerHypernetwork(nn.Module):
    """
    Predicts square low-rank projectors W_text and W_img via shared U/V tokens
    and separate two-layer MLP decoders.

    Inputs:
      - query_emb: [B, E] text/query embeddings
    Outputs (dict):
      - 'W_text': [B, E, E]
      - 'W_img' : [B, E, E]
    If return_all, also returns 'all': List[Dict[str, Tensor]]
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
        use_separate_decoders: bool = True,
    ):
        super().__init__()
        self.E = embedding_dim
        self.r = low_rank_dim
        self.H = hidden_dim
        self.num_steps = num_steps
        self.use_separate_decoders = use_separate_decoders

        # Query conditioning
        self.query_encoder = nn.Sequential(
            nn.Linear(self.E, self.H),
            nn.LayerNorm(self.H),
            nn.GELU(),
            nn.Linear(self.H, self.H),
            nn.LayerNorm(self.H)
        )

        # Step embedding
        self.step_embedding = nn.Embedding(num_steps, self.H)

        # Positional encodings
        seq_len = 2 * self.r + 1
        pe = torch.zeros(seq_len, self.H)
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.H, 2) * (-math.log(10000.0) / self.H))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_emb', pe)

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=self.H,
            nhead=nhead,
            dim_feedforward=self.H * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)

        # Two-layer MLP decoders
        def make_decoder():
            return nn.Sequential(
                nn.Linear(self.H, self.H),
                nn.GELU(),
                nn.Linear(self.H, self.E)
            )
        if self.use_separate_decoders:
            self.dec_u_text = make_decoder()
            self.dec_v_text = make_decoder()
            self.dec_u_img  = make_decoder()
            self.dec_v_img  = make_decoder()
        else:
            self.dec_u = make_decoder()
            self.dec_v = make_decoder()

    def forward(
        self,
        query_emb: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, Any]:
        B, E = query_emb.shape
        device = query_emb.device

        # Encode query
        ctx = self.query_encoder(query_emb)  # [B, H]

        # Initialize shared tokens (zeros)
        u_tok = torch.zeros(B, self.r, self.H, device=device)
        v_tok = torch.zeros(B, self.r, self.H, device=device)

        all_steps: Optional[List[Dict[str, torch.Tensor]]] = [] if return_all else None
        if return_all:
            zero = torch.zeros(B, E, E, device=device)
            all_steps.append({'W_text': zero.clone(), 'W_img': zero.clone()})

        # Iterative refinement
        for t in range(self.num_steps):
            # Condition token
            step_idx = torch.full((B,), t, device=device, dtype=torch.long)
            cond = (ctx + self.step_embedding(step_idx)).unsqueeze(1)  # [B,1,H]

            # Add positional encoding
            u_seq = scaled_positional_encoding(u_tok, self.pos_emb)
            v_seq = scaled_positional_encoding(v_tok, self.pos_emb)

            # Build sequence: [cond, U-tokens, V-tokens]
            seq = torch.cat([cond, u_seq, v_seq], dim=1)  # [B,1+2r,H]
            out = self.transformer(seq)  # [B,1+2r,H]
            delta = out[:, 1:, :]

            # Split and update tokens
            d_u = delta[:, :self.r, :]
            d_v = delta[:, self.r:, :]
            u_tok = u_tok + d_u
            v_tok = v_tok + d_v

            if return_all:
                Wt, Wi = self._decode_and_proj(u_tok, v_tok)
                all_steps.append({'W_text': Wt, 'W_image': Wi})

        # Final decode & projection
        W_text, W_img = self._decode_and_proj(u_tok, v_tok)
        out: Dict[str, Any] = {'W_text': W_text, 'W_image': W_img}
        if return_all:
            out['all'] = all_steps
        return out

    def _decode_and_proj(
        self,
        u_tok: torch.Tensor,
        v_tok: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode tokens into U_text, V_text and U_img, V_img, then form W = U @ V^T.
        u_tok, v_tok: [B, r, H]
        Returns:
          W_text: [B, E, E], W_img: [B, E, E]
        """
        # Decode columns

        if self.use_separate_decoders:
            ut_cols = self.dec_u_text(u_tok)  # [B, r, E]
            vt_cols = self.dec_v_text(v_tok)
            ui_cols = self.dec_u_img(u_tok)
            vi_cols = self.dec_v_img(v_tok)

            # Transpose to [B, E, r]
            U_t = ut_cols.transpose(1, 2)
            V_t = vt_cols.transpose(1, 2)
            U_i = ui_cols.transpose(1, 2)
            V_i = vi_cols.transpose(1, 2)

            # Compute W = U @ V^T
            W_text = torch.bmm(U_t, V_t.transpose(1, 2))
            W_img  = torch.bmm(U_i, V_i.transpose(1, 2))
        else:
            u_cols = self.dec_u(u_tok)
            v_cols = self.dec_v(v_tok)

            # Transpose to [B, E, r]
            U = u_cols.transpose(1, 2)
            V = v_cols.transpose(1, 2)

            # Compute W = U @ V^T
            W = torch.bmm(U, V.transpose(1, 2))
            
            W_text = W
            W_img = W

        return W_text, W_img

    def project_and_score(
        self,
        query_emb: torch.Tensor,
        img_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies W_text and W_img, computes cosine similarities.
        """
        out = self.forward(query_emb)
        q_proj = torch.bmm(query_emb.unsqueeze(1), out['W_text']).squeeze(1)
        q_proj = F.normalize(q_proj, dim=-1)
        if img_emb.dim() == 3:
            img_proj = torch.matmul(img_emb, out['W_img'])
            sim = F.cosine_similarity(q_proj.unsqueeze(1), img_proj, dim=-1)
        else:
            img_proj = torch.matmul(img_emb, out['W_img']).squeeze(1)
            sim = F.cosine_similarity(q_proj, img_proj, dim=-1)
        return sim
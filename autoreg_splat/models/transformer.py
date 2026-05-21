"""
Autoregressive transformer for 3D Gaussian sequence prediction.

Decoder-only transformer with cross-attention to top-down conditioning.
Predicts the next Gaussian token given all previous tokens + top-down features.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv_proj = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        M = context.shape[1]

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, M, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = hidden_dim * mlp_ratio
        self.w1 = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.w2 = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, inner_dim, bias=False)  # SwiGLU gate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        has_cross_attn: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.self_attn = CausalSelfAttention(hidden_dim, num_heads, dropout)

        self.has_cross_attn = has_cross_attn
        if has_cross_attn:
            self.norm_cross = RMSNorm(hidden_dim)
            self.cross_attn = CrossAttention(hidden_dim, num_heads, dropout)

        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))

        if self.has_cross_attn and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context)

        x = x + self.ffn(self.norm2(x))
        return x


# ─── Special token IDs (offsets above the RVQ vocab) ───
BOS_TOKEN = 0
EOS_TOKEN = 1
SEP_TOKEN = 2
NUM_SPECIAL_TOKENS = 3


class AutoregSplatTransformer(nn.Module):
    """
    Autoregressive transformer that predicts 3D Gaussian tokens.

    The vocabulary is: [BOS, EOS, SEP] + [codebook_1 tokens] + [codebook_2 tokens] + ...
    Each Gaussian is represented as 4 RVQ tokens separated by SEP tokens.

    Sequence format:
        [BOS] g1_q1 g1_q2 g1_q3 g1_q4 [SEP] g2_q1 g2_q2 g2_q3 g2_q4 [SEP] ... [EOS]
    """

    def __init__(
        self,
        codebook_size: int = 1024,
        num_codebooks: int = 4,
        num_layers: int = 24,
        hidden_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        cross_attn_every_n: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.max_seq_len = max_seq_len

        # Total vocabulary: special tokens + codebook_size * num_codebooks
        # Each codebook gets its own token range to distinguish RVQ levels
        self.vocab_size = NUM_SPECIAL_TOKENS + codebook_size * num_codebooks

        self.token_embedding = nn.Embedding(self.vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Intra-Gaussian position: which RVQ level (0-3) or special token
        self.intra_gaussian_embedding = nn.Embedding(num_codebooks + 1, hidden_dim)

        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                has_cross_attn=(i % cross_attn_every_n == 0),
            )
            for i in range(num_layers)
        ])

        self.norm_out = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, self.vocab_size, bias=False)

        # tie input/output embeddings
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def rvq_indices_to_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert RVQ codebook indices to global token IDs.

        Args:
            indices: (B, N, num_codebooks) — raw codebook indices in [0, codebook_size)
        Returns:
            tokens: (B, seq_len) — flattened token sequence with special tokens
        """
        B, N, Q = indices.shape
        device = indices.device

        # offset each codebook: level_i tokens start at NUM_SPECIAL + i * codebook_size
        offsets = torch.arange(Q, device=device) * self.codebook_size + NUM_SPECIAL_TOKENS
        offset_indices = indices + offsets.unsqueeze(0).unsqueeze(0)  # (B, N, Q)

        # build sequence: [BOS] [g1_q1..g1_qQ] [SEP] [g2_q1..g2_qQ] [SEP] ... [EOS]
        parts = [torch.full((B, 1), BOS_TOKEN, device=device, dtype=torch.long)]
        for i in range(N):
            parts.append(offset_indices[:, i, :])  # Q tokens
            if i < N - 1:
                parts.append(torch.full((B, 1), SEP_TOKEN, device=device, dtype=torch.long))
        parts.append(torch.full((B, 1), EOS_TOKEN, device=device, dtype=torch.long))

        return torch.cat(parts, dim=1)

    def tokens_to_rvq_indices(self, tokens: torch.Tensor) -> torch.Tensor:
        """Inverse of rvq_indices_to_tokens — extract codebook indices from token sequence."""
        Q = self.num_codebooks
        # strip BOS and EOS
        inner = tokens[:, 1:-1]

        # remove SEP tokens
        mask = inner != SEP_TOKEN
        # each Gaussian is Q consecutive non-SEP tokens
        non_sep = inner[mask].reshape(-1, Q)

        offsets = torch.arange(Q, device=tokens.device) * self.codebook_size + NUM_SPECIAL_TOKENS
        raw_indices = non_sep - offsets.unsqueeze(0)
        return raw_indices

    def _get_intra_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute the intra-Gaussian position for each token in the sequence.

        BOS/EOS/SEP → position num_codebooks (special)
        RVQ level j → position j
        """
        Q = self.num_codebooks
        group_len = Q + 1  # Q tokens + 1 SEP
        positions = torch.full((seq_len,), Q, device=device, dtype=torch.long)  # default: special

        # BOS is at index 0 → special (already set)
        # tokens at index 1..(seq_len-1) before EOS:
        for i in range(1, seq_len - 1):
            pos_in_group = (i - 1) % group_len
            if pos_in_group < Q:
                positions[i] = pos_in_group
            # else it's a SEP → stays as special

        return positions

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B, T) token IDs
            context: (B, M, hidden_dim) conditioning from top-down encoder
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = tokens.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        positions = torch.arange(T, device=tokens.device)
        intra_positions = self._get_intra_positions(T, tokens.device)

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
            + self.intra_gaussian_embedding(intra_positions)
        )
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, context)

        x = self.norm_out(x)
        return self.lm_head(x)

    def compute_loss(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Standard next-token prediction loss."""
        logits = self.forward(tokens[:, :-1], context)
        targets = tokens[:, 1:]

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=-1,
        )

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == targets).float().mean()

        metrics = {"ce_loss": loss.item(), "token_accuracy": accuracy.item()}
        return loss, metrics

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor | None = None,
        max_gaussians: int = 512,
        temperature: float = 0.9,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
    ) -> torch.Tensor:
        """Autoregressively generate a sequence of Gaussian tokens.

        Returns:
            tokens: (1, T) generated token sequence
        """
        self.eval()
        device = next(self.parameters()).device

        tokens = torch.tensor([[BOS_TOKEN]], device=device, dtype=torch.long)
        Q = self.num_codebooks
        max_tokens = 1 + max_gaussians * (Q + 1) + 1  # BOS + N*(Q+SEP) + EOS

        for _ in range(max_tokens):
            if tokens.shape[1] >= self.max_seq_len:
                break

            logits = self.forward(tokens, context)
            next_logits = logits[:, -1, :] / temperature

            # mask out invalid tokens based on position in current Gaussian
            next_logits = self._apply_grammar_mask(tokens, next_logits)

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove_mask] = float("-inf")
                next_logits.scatter_(1, sorted_indices, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == EOS_TOKEN:
                break

        return tokens

    def _apply_grammar_mask(
        self, tokens: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """Enforce valid token structure: after Q RVQ tokens, must emit SEP or EOS."""
        Q = self.num_codebooks
        seq_len = tokens.shape[1]

        # count tokens since last BOS or SEP
        last_special = 0
        for i in range(seq_len - 1, -1, -1):
            t = tokens[0, i].item()
            if t in (BOS_TOKEN, SEP_TOKEN):
                last_special = i
                break

        tokens_since_special = seq_len - last_special - 1

        if tokens_since_special == Q:
            # must emit SEP or EOS
            mask = torch.full_like(logits, float("-inf"))
            mask[:, SEP_TOKEN] = 0
            mask[:, EOS_TOKEN] = 0
            logits = logits + mask
        elif tokens_since_special < Q:
            # must emit an RVQ token from the correct codebook level
            level = tokens_since_special
            valid_start = NUM_SPECIAL_TOKENS + level * self.codebook_size
            valid_end = valid_start + self.codebook_size
            mask = torch.full_like(logits, float("-inf"))
            mask[:, valid_start:valid_end] = 0
            logits = logits + mask

        return logits

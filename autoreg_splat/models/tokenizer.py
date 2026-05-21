"""
Residual Vector Quantization VAE for 3D Gaussians.

Each Gaussian (59 continuous params) → 4 discrete tokens via RVQ.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantizer(nn.Module):
    """Single-level vector quantizer with EMA codebook updates."""

    def __init__(self, codebook_size: int, code_dim: int, ema_decay: float = 0.99):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.ema_decay = ema_decay

        self.embedding = nn.Embedding(codebook_size, code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", self.embedding.weight.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = z.reshape(-1, self.code_dim)

        # distances: (B, K)
        dist = (
            flat.pow(2).sum(dim=-1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=-1, keepdim=True).t()
        )
        indices = dist.argmin(dim=-1)
        quantized = self.embedding(indices).reshape_as(z)

        if self.training:
            self._ema_update(flat, indices)

        # straight-through estimator
        commitment_loss = F.mse_loss(z.detach(), quantized) + F.mse_loss(z, quantized.detach())
        quantized_st = z + (quantized - z).detach()

        return quantized_st, indices.reshape(z.shape[:-1]), commitment_loss

    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor):
        cluster_size = torch.zeros(self.codebook_size, device=flat.device)
        cluster_size.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float))

        embed_sum = torch.zeros_like(self.ema_embed_sum)
        embed_sum.scatter_add_(0, indices.unsqueeze(1).expand_as(flat), flat)

        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        n = self.ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
        )
        embed_normalized = self.ema_embed_sum / cluster_size_smoothed.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)


class ResidualVQ(nn.Module):
    """Residual Vector Quantization — stacks multiple VQ layers."""

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        code_dim: int,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.layers = nn.ModuleList(
            [VectorQuantizer(codebook_size, code_dim, ema_decay) for _ in range(num_codebooks)]
        )

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, code_dim) latent vectors
        Returns:
            quantized: (B, code_dim) sum of all quantized residuals
            indices: (B, num_codebooks) token indices per level
            total_loss: scalar commitment loss
        """
        quantized_out = torch.zeros_like(z)
        residual = z
        all_indices = []
        total_loss = torch.tensor(0.0, device=z.device)

        for layer in self.layers:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            total_loss = total_loss + loss

        all_indices = torch.stack(all_indices, dim=-1)  # (B, num_codebooks)
        return quantized_out, all_indices, total_loss / self.num_codebooks

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (B, num_codebooks) or (B, N, num_codebooks)
        Returns:
            quantized: sum of embeddings from each codebook level
        """
        quantized = torch.zeros(
            *indices.shape[:-1], self.layers[0].code_dim, device=indices.device
        )
        for i, layer in enumerate(self.layers):
            quantized = quantized + layer.decode_indices(indices[..., i])
        return quantized


class GaussianRVQVAE(nn.Module):
    """
    Tokenizes 3D Gaussians into discrete codes via RVQ-VAE.

    Gaussian params (59D) → Encoder → Latent (256D) → RVQ → 4 codes → Decoder → Reconstructed params
    """

    def __init__(
        self,
        input_dim: int = 59,
        latent_dim: int = 256,
        num_codebooks: int = 4,
        codebook_size: int = 1024,
        code_dim: int = 64,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.commitment_weight = commitment_weight

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, code_dim),
        )

        self.rvq = ResidualVQ(num_codebooks, codebook_size, code_dim, ema_decay)

        self.decoder = nn.Sequential(
            nn.Linear(code_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, input_dim),
        )

    def encode(self, gaussians: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode Gaussians to discrete tokens.

        Args:
            gaussians: (B, 59) raw Gaussian parameters
        Returns:
            indices: (B, num_codebooks) discrete token indices
            commitment_loss: scalar
        """
        z = self.encoder(gaussians)
        quantized, indices, commitment_loss = self.rvq(z)
        return indices, commitment_loss

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens back to Gaussian parameters.

        Args:
            indices: (B, num_codebooks) token indices
        Returns:
            gaussians: (B, 59) reconstructed parameters
        """
        quantized = self.rvq.decode_indices(indices)
        return self.decoder(quantized)

    def forward(
        self, gaussians: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        z = self.encoder(gaussians)
        quantized, indices, commitment_loss = self.rvq(z)
        reconstructed = self.decoder(quantized)

        recon_loss = F.l1_loss(reconstructed, gaussians) + F.mse_loss(reconstructed, gaussians)
        total_loss = recon_loss + self.commitment_weight * commitment_loss

        metrics = {
            "recon_loss": recon_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "total_loss": total_loss.item(),
        }
        return reconstructed, total_loss, metrics

    def get_vocab_size(self) -> int:
        return self.rvq.layers[0].codebook_size

    def get_num_codebooks(self) -> int:
        return self.rvq.num_codebooks

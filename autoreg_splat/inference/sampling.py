"""
Sampling / decoding strategies for autoregressive Gaussian generation.

Supports: greedy, temperature, top-k, nucleus (top-p), classifier-free guidance.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from autoreg_splat.models.transformer import (
    AutoregSplatTransformer,
    BOS_TOKEN,
    EOS_TOKEN,
)


class AutoregSplatSampler:
    """High-level sampler that wraps generation + detokenization."""

    def __init__(
        self,
        transformer: AutoregSplatTransformer,
        tokenizer,
        encoder=None,
    ):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.encoder = encoder

    @torch.no_grad()
    def sample(
        self,
        top_down_image: torch.Tensor | None = None,
        max_gaussians: int = 512,
        temperature: float = 0.9,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        cfg_scale: float = 1.0,
        uncond_context: torch.Tensor | None = None,
    ) -> dict:
        """Generate a 3D Gaussian scene from a top-down image.

        Args:
            top_down_image: (1, 3, H, W) preprocessed image, or None for unconditional
            max_gaussians: maximum number of Gaussians to generate
            temperature: sampling temperature (higher = more diverse)
            top_k: top-k filtering (None to disable)
            top_p: nucleus sampling threshold (None to disable)
            cfg_scale: classifier-free guidance scale (1.0 = no guidance)
            uncond_context: precomputed unconditional context for CFG

        Returns:
            dict with keys:
                gaussian_params: (N, 59) decoded Gaussian parameters
                tokens: (1, T) raw token sequence
                num_gaussians: int
        """
        device = next(self.transformer.parameters()).device

        # encode conditioning
        context = None
        if top_down_image is not None and self.encoder is not None:
            context = self.encoder(top_down_image.to(device))

        if cfg_scale != 1.0 and context is not None:
            tokens = self._generate_cfg(
                context, uncond_context, max_gaussians, temperature, top_k, top_p, cfg_scale
            )
        else:
            tokens = self.transformer.generate(
                context=context,
                max_gaussians=max_gaussians,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # detokenize: token sequence → RVQ indices → Gaussian params
        rvq_indices = self.transformer.tokens_to_rvq_indices(tokens)
        gaussian_params = self.tokenizer.decode(rvq_indices)

        return {
            "gaussian_params": gaussian_params,
            "tokens": tokens,
            "num_gaussians": len(rvq_indices),
        }

    @torch.no_grad()
    def _generate_cfg(
        self,
        cond_context: torch.Tensor,
        uncond_context: torch.Tensor | None,
        max_gaussians: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Generate with classifier-free guidance."""
        device = cond_context.device
        Q = self.transformer.num_codebooks

        if uncond_context is None:
            uncond_context = torch.zeros_like(cond_context)

        tokens = torch.tensor([[BOS_TOKEN]], device=device, dtype=torch.long)
        max_tokens = 1 + max_gaussians * (Q + 1) + 1

        for _ in range(max_tokens):
            if tokens.shape[1] >= self.transformer.max_seq_len:
                break

            # conditional and unconditional forward passes
            logits_cond = self.transformer(tokens, cond_context)[:, -1, :]
            logits_uncond = self.transformer(tokens, uncond_context)[:, -1, :]

            # CFG interpolation
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            logits = logits / temperature

            logits = self.transformer._apply_grammar_mask(tokens, logits)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove] = float("-inf")
                logits.scatter_(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == EOS_TOKEN:
                break

        return tokens

    def params_to_gaussians(self, params: torch.Tensor) -> dict:
        """Split flat parameter tensor into named Gaussian attributes.

        Args:
            params: (N, 59) flat parameter tensor
        Returns:
            dict with positions (N,3), rotations (N,4), scales (N,3),
                 opacities (N,1), sh_coeffs (N,48)
        """
        return {
            "positions": params[:, 0:3],
            "rotations": params[:, 3:7],
            "scales": params[:, 7:10],
            "opacities": params[:, 10:11],
            "sh_coeffs": params[:, 11:59],
        }

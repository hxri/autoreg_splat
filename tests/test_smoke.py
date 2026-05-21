"""Smoke tests: verify all components instantiate and forward-pass correctly."""

import torch
import pytest


def test_tokenizer_roundtrip():
    from autoreg_splat.models.tokenizer import GaussianRVQVAE

    model = GaussianRVQVAE(
        input_dim=59, latent_dim=128, num_codebooks=4,
        codebook_size=64, code_dim=32,
    )

    gaussians = torch.randn(16, 59)
    reconstructed, loss, metrics = model(gaussians)

    assert reconstructed.shape == (16, 59)
    assert loss.dim() == 0
    assert "recon_loss" in metrics

    indices, _ = model.encode(gaussians)
    assert indices.shape == (16, 4)

    decoded = model.decode(indices)
    assert decoded.shape == (16, 59)


def test_transformer_forward():
    from autoreg_splat.models.transformer import AutoregSplatTransformer

    model = AutoregSplatTransformer(
        codebook_size=64, num_codebooks=4,
        num_layers=2, hidden_dim=128, num_heads=4,
        mlp_ratio=2, max_seq_len=512, cross_attn_every_n=1,
    )

    # create a mock token sequence: BOS + 3 Gaussians * (4 RVQ + SEP) + EOS
    rvq_indices = torch.randint(0, 64, (2, 3, 4))
    tokens = model.rvq_indices_to_tokens(rvq_indices)

    context = torch.randn(2, 10, 128)
    logits = model(tokens, context)

    assert logits.shape == (2, tokens.shape[1], model.vocab_size)


def test_transformer_loss():
    from autoreg_splat.models.transformer import AutoregSplatTransformer

    model = AutoregSplatTransformer(
        codebook_size=64, num_codebooks=4,
        num_layers=2, hidden_dim=128, num_heads=4,
        mlp_ratio=2, max_seq_len=512, cross_attn_every_n=1,
    )

    rvq_indices = torch.randint(0, 64, (2, 5, 4))
    tokens = model.rvq_indices_to_tokens(rvq_indices)
    context = torch.randn(2, 10, 128)

    loss, metrics = model.compute_loss(tokens, context)

    assert loss.dim() == 0
    assert loss.item() > 0
    assert "ce_loss" in metrics
    assert "token_accuracy" in metrics


def test_transformer_generate():
    from autoreg_splat.models.transformer import AutoregSplatTransformer

    model = AutoregSplatTransformer(
        codebook_size=64, num_codebooks=4,
        num_layers=2, hidden_dim=128, num_heads=4,
        mlp_ratio=2, max_seq_len=256, cross_attn_every_n=1,
    )
    model.eval()

    context = torch.randn(1, 10, 128)
    tokens = model.generate(context=context, max_gaussians=5, temperature=1.0, top_k=10)

    assert tokens.shape[0] == 1
    assert tokens[0, 0].item() == 0  # BOS


def test_hilbert_sort():
    from autoreg_splat.data.hilbert import hilbert_sort_gaussians_fast

    positions = torch.rand(100, 3)
    sort_idx = hilbert_sort_gaussians_fast(positions, bits=8)

    assert sort_idx.shape == (100,)
    assert set(sort_idx.tolist()) == set(range(100))


def test_gaussian_utils():
    from autoreg_splat.utils.gaussian import (
        pack_gaussians, unpack_gaussians,
        normalize_gaussians, denormalize_gaussians,
    )

    N = 50
    pos = torch.randn(N, 3)
    rot = torch.randn(N, 4)
    rot = rot / rot.norm(dim=-1, keepdim=True)
    scale = torch.rand(N, 3) * 0.1
    opacity = torch.rand(N, 1)
    sh = torch.randn(N, 48)

    packed = pack_gaussians(pos, rot, scale, opacity, sh)
    assert packed.shape == (N, 59)

    unpacked = unpack_gaussians(packed)
    assert torch.allclose(unpacked["positions"], pos)

    normalized, stats = normalize_gaussians(packed)
    assert normalized.shape == packed.shape

    denorm = denormalize_gaussians(normalized, stats)
    assert torch.allclose(denorm[:, :3], pos, atol=1e-5)


def test_token_roundtrip():
    """Verify RVQ indices survive tokenization → detokenization."""
    from autoreg_splat.models.transformer import AutoregSplatTransformer

    model = AutoregSplatTransformer(
        codebook_size=64, num_codebooks=4,
        num_layers=2, hidden_dim=128, num_heads=4,
    )

    original_indices = torch.randint(0, 64, (1, 10, 4))
    tokens = model.rvq_indices_to_tokens(original_indices)
    recovered = model.tokens_to_rvq_indices(tokens)

    assert torch.equal(original_indices.reshape(-1, 4), recovered)


def test_end_to_end_small():
    """Minimal end-to-end: tokenize Gaussians → build sequence → compute loss."""
    from autoreg_splat.models.tokenizer import GaussianRVQVAE
    from autoreg_splat.models.transformer import AutoregSplatTransformer

    tokenizer = GaussianRVQVAE(
        input_dim=59, latent_dim=64, num_codebooks=4,
        codebook_size=32, code_dim=16,
    )
    transformer = AutoregSplatTransformer(
        codebook_size=32, num_codebooks=4,
        num_layers=2, hidden_dim=64, num_heads=4,
        mlp_ratio=2, max_seq_len=256,
    )

    gaussians = torch.randn(2, 8, 59)  # batch of 2, 8 Gaussians each
    B, N, D = gaussians.shape

    indices, _ = tokenizer.encode(gaussians.reshape(B * N, D))
    indices = indices.reshape(B, N, -1)

    tokens = transformer.rvq_indices_to_tokens(indices)

    context = torch.randn(B, 10, 64)
    loss, metrics = transformer.compute_loss(tokens, context)

    assert loss.dim() == 0
    assert not torch.isnan(loss)

"""
Diagnose trained AR transformer quality.

Checks: generation quality, spatial coherence, diversity, conditioning fidelity.

Usage:
    python3 scripts/diagnose_model.py \
        --model_ckpt checkpoints/model/model_best.pt \
        --tokenizer_ckpt checkpoints/tokenizer/tokenizer_best.pt \
        --data_dir data/synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from autoreg_splat.models.tokenizer import GaussianRVQVAE
from autoreg_splat.models.transformer import AutoregSplatTransformer, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN
from autoreg_splat.utils.gaussian import unpack_gaussians


def load_models(cfg, args, device):
    tokenizer = GaussianRVQVAE(
        input_dim=cfg.tokenizer.input_dim,
        latent_dim=cfg.tokenizer.latent_dim,
        num_codebooks=cfg.tokenizer.num_codebooks,
        codebook_size=cfg.tokenizer.codebook_size,
        code_dim=cfg.tokenizer.code_dim,
    ).to(device)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=device))
    tokenizer.eval()

    transformer = AutoregSplatTransformer(
        codebook_size=cfg.tokenizer.codebook_size,
        num_codebooks=cfg.tokenizer.num_codebooks,
        num_layers=cfg.transformer.num_layers,
        hidden_dim=cfg.transformer.hidden_dim,
        num_heads=cfg.transformer.num_heads,
        mlp_ratio=cfg.transformer.mlp_ratio,
        dropout=0.0,
        max_seq_len=cfg.transformer.max_seq_len,
        cross_attn_every_n=cfg.transformer.cross_attn_every_n,
    ).to(device)
    ckpt = torch.load(args.model_ckpt, map_location=device)
    transformer.load_state_dict(ckpt["transformer"])
    transformer.eval()

    return tokenizer, transformer


def check_generation_basics(transformer, device):
    """Does the model generate valid token sequences at all?"""
    print("═══ 1. GENERATION BASICS ═══")
    context = torch.randn(1, 10, transformer.hidden_dim, device=device)

    tokens = transformer.generate(
        context=context, max_gaussians=50, temperature=0.9, top_k=50,
    )

    seq = tokens[0].cpu().tolist()
    print(f"  Sequence length: {len(seq)} tokens")
    print(f"  Starts with BOS: {seq[0] == BOS_TOKEN}")
    print(f"  Ends with EOS:   {seq[-1] == EOS_TOKEN}")

    # count Gaussians (number of SEP tokens + 1 if EOS present)
    n_sep = seq.count(SEP_TOKEN)
    n_gaussians = n_sep + (1 if seq[-1] == EOS_TOKEN else 0)
    # actually count more carefully
    Q = transformer.num_codebooks
    content_tokens = [t for t in seq if t not in (BOS_TOKEN, EOS_TOKEN, SEP_TOKEN)]
    n_gaussians = len(content_tokens) // Q

    print(f"  Gaussians generated: {n_gaussians}")

    # check grammar: every (Q+1)-th token after BOS should be SEP
    valid_grammar = True
    pos = 1  # skip BOS
    while pos < len(seq) - 1:  # -1 to skip EOS
        chunk = seq[pos:pos + Q]
        if len(chunk) < Q:
            break
        for t in chunk:
            if t in (BOS_TOKEN, EOS_TOKEN, SEP_TOKEN):
                valid_grammar = False
                break
        pos += Q
        if pos < len(seq) - 1:
            if seq[pos] not in (SEP_TOKEN, EOS_TOKEN):
                valid_grammar = False
            pos += 1

    print(f"  Valid grammar:   {valid_grammar}")

    if n_gaussians == 0:
        print("  ✗ FAIL — model generates no Gaussians (immediate EOS)")
        return False
    if not valid_grammar:
        print("  ✗ FAIL — token grammar is broken")
        return False
    print("  ✓ PASS")
    return True


def check_spatial_coherence(transformer, tokenizer, device):
    """Do generated Gaussians form spatially coherent structures?"""
    print("\n═══ 2. SPATIAL COHERENCE ═══")

    context = torch.randn(1, 10, transformer.hidden_dim, device=device)
    tokens = transformer.generate(
        context=context, max_gaussians=200, temperature=0.8, top_k=50,
    )

    rvq_indices = transformer.tokens_to_rvq_indices(tokens)
    params = tokenizer.decode(rvq_indices.to(device))
    g = unpack_gaussians(params.cpu())
    positions = g["positions"]

    if len(positions) < 5:
        print("  ✗ Too few Gaussians to analyze")
        return

    # bounding box
    bbox_min = positions.min(dim=0).values
    bbox_max = positions.max(dim=0).values
    bbox_size = bbox_max - bbox_min
    print(f"  Bounding box: [{bbox_min[0]:.2f},{bbox_min[1]:.2f},{bbox_min[2]:.2f}] "
          f"to [{bbox_max[0]:.2f},{bbox_max[1]:.2f},{bbox_max[2]:.2f}]")
    print(f"  Bbox size:    [{bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}]")

    # check: are positions clustered or uniformly random?
    centroid = positions.mean(dim=0)
    distances = (positions - centroid).norm(dim=-1)
    print(f"  Mean dist from centroid: {distances.mean():.3f}")
    print(f"  Std dist from centroid:  {distances.std():.3f}")

    # nearest-neighbor distances (spatial locality)
    from scipy.spatial import cKDTree
    tree = cKDTree(positions.numpy())
    nn_dists, _ = tree.query(positions.numpy(), k=2)
    nn_dists = nn_dists[:, 1]  # skip self
    print(f"  Mean NN distance: {nn_dists.mean():.4f}")
    print(f"  Std NN distance:  {nn_dists.std():.4f}")

    # a random point cloud would have high variance in NN distances
    # a structured scene has more uniform NN distances
    cv = nn_dists.std() / (nn_dists.mean() + 1e-8)
    print(f"  NN distance CV:   {cv:.3f} (lower = more uniform = more structured)")

    if cv < 1.0:
        print("  ✓ Gaussians show spatial structure (not random scatter)")
    elif cv < 2.0:
        print("  ~ Some structure visible, but noisy")
    else:
        print("  ✗ Looks like random scatter — model hasn't learned spatial priors")


def check_diversity(transformer, tokenizer, device, n_samples=5):
    """Do different samples produce different outputs?"""
    print(f"\n═══ 3. DIVERSITY ({n_samples} samples, same conditioning) ═══")

    context = torch.randn(1, 10, transformer.hidden_dim, device=device)
    all_positions = []

    for i in range(n_samples):
        tokens = transformer.generate(
            context=context, max_gaussians=100, temperature=0.9, top_k=50,
        )
        rvq_indices = transformer.tokens_to_rvq_indices(tokens)
        params = tokenizer.decode(rvq_indices.to(device))
        g = unpack_gaussians(params.cpu())
        all_positions.append(g["positions"])
        print(f"  Sample {i+1}: {len(g['positions'])} Gaussians")

    # compare centroids across samples
    centroids = torch.stack([p.mean(dim=0) for p in all_positions])
    centroid_spread = centroids.std(dim=0).mean()
    print(f"  Centroid spread across samples: {centroid_spread:.4f}")

    # compare Gaussian counts
    counts = [len(p) for p in all_positions]
    print(f"  Count range: {min(counts)} — {max(counts)}")

    if centroid_spread < 0.001 and max(counts) - min(counts) < 3:
        print("  ⚠ Very low diversity — model may have collapsed to single mode")
        print("  → Try higher temperature or check for mode collapse")
    elif centroid_spread > 0.001:
        print("  ✓ Model produces diverse outputs from same conditioning")
    else:
        print("  ~ Moderate diversity")


def check_conditioning_sensitivity(transformer, tokenizer, device):
    """Does changing the conditioning change the output?"""
    print("\n═══ 4. CONDITIONING SENSITIVITY ═══")

    # generate with two very different contexts
    context_a = torch.randn(1, 10, transformer.hidden_dim, device=device) * 2
    context_b = -context_a  # maximally different

    tokens_a = transformer.generate(context=context_a, max_gaussians=100, temperature=0.5, top_k=20)
    tokens_b = transformer.generate(context=context_b, max_gaussians=100, temperature=0.5, top_k=20)

    rvq_a = transformer.tokens_to_rvq_indices(tokens_a)
    rvq_b = transformer.tokens_to_rvq_indices(tokens_b)
    params_a = tokenizer.decode(rvq_a.to(device))
    params_b = tokenizer.decode(rvq_b.to(device))
    pos_a = unpack_gaussians(params_a.cpu())["positions"]
    pos_b = unpack_gaussians(params_b.cpu())["positions"]

    centroid_a = pos_a.mean(dim=0)
    centroid_b = pos_b.mean(dim=0)
    centroid_diff = (centroid_a - centroid_b).norm().item()

    print(f"  Context A → {len(pos_a)} Gaussians, centroid: "
          f"[{centroid_a[0]:.2f}, {centroid_a[1]:.2f}, {centroid_a[2]:.2f}]")
    print(f"  Context B → {len(pos_b)} Gaussians, centroid: "
          f"[{centroid_b[0]:.2f}, {centroid_b[1]:.2f}, {centroid_b[2]:.2f}]")
    print(f"  Centroid difference: {centroid_diff:.4f}")

    if centroid_diff > 0.1:
        print("  ✓ Model is sensitive to conditioning (different input → different output)")
    elif centroid_diff > 0.01:
        print("  ~ Weak conditioning effect — model may be ignoring input")
    else:
        print("  ✗ Model ignores conditioning — outputs are independent of input")
        print("  → Check: is the cross-attention working? CFG dropout too high?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--tokenizer_ckpt", required=True)
    parser.add_argument("--data_dir", default="data/synthetic")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, transformer = load_models(cfg, args, device)

    print("╔══════════════════════════════════════════════╗")
    print("║    AutoReg Splat — Model Diagnostics        ║")
    print("╚══════════════════════════════════════════════╝\n")

    ok = check_generation_basics(transformer, device)
    if not ok:
        print("\n⚠ Fix generation basics before proceeding")
        return

    check_spatial_coherence(transformer, tokenizer, device)
    check_diversity(transformer, tokenizer, device)
    check_conditioning_sensitivity(transformer, tokenizer, device)

    print("\n════════════════════════════════════════════════")
    print("See above for per-check verdicts.")
    print("Key question: does check #2 (spatial coherence) show structure?")
    print("If yes — the core idea is working. Scale up data + training.")
    print("If no  — debug tokenizer quality first, then check loss curves.")
    print("════════════════════════════════════════════════")


if __name__ == "__main__":
    main()

"""
Diagnose tokenizer quality after Phase 1 training.

Checks: reconstruction error, codebook utilization, per-parameter errors.

Usage:
    python3 scripts/diagnose_tokenizer.py \
        --tokenizer_ckpt checkpoints/tokenizer/tokenizer_best.pt \
        --data_dir data/synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np

from autoreg_splat.models.tokenizer import GaussianRVQVAE
from autoreg_splat.utils.gaussian import normalize_gaussians, GAUSSIAN_DIM


PARAM_NAMES = (
    ["pos_x", "pos_y", "pos_z"]
    + ["rot_w", "rot_x", "rot_y", "rot_z"]
    + ["scale_x", "scale_y", "scale_z"]
    + ["opacity"]
    + [f"sh_{i}" for i in range(48)]
)

PARAM_GROUPS = {
    "position": slice(0, 3),
    "rotation": slice(3, 7),
    "scale": slice(7, 10),
    "opacity": slice(10, 11),
    "sh_coeffs": slice(11, 59),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_samples", type=int, default=50000)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    tokenizer = GaussianRVQVAE(
        input_dim=cfg.tokenizer.input_dim,
        latent_dim=cfg.tokenizer.latent_dim,
        num_codebooks=cfg.tokenizer.num_codebooks,
        codebook_size=cfg.tokenizer.codebook_size,
        code_dim=cfg.tokenizer.code_dim,
    ).to(device)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=device))
    tokenizer.eval()

    # load a sample of Gaussians
    data_path = Path(args.data_dir)
    all_params = []
    for scene_dir in sorted(data_path.iterdir()):
        npz_path = scene_dir / "gaussians.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        params = np.concatenate([
            data["positions"], data["rotations"], data["scales"],
            data["opacities"], data["sh_coeffs"],
        ], axis=-1)
        all_params.append(params)
        if sum(len(p) for p in all_params) >= args.max_samples:
            break

    all_params = torch.from_numpy(np.concatenate(all_params)[:args.max_samples]).float()
    normalized, stats = normalize_gaussians(all_params)
    normalized = normalized.to(device)

    print(f"Testing on {len(normalized)} Gaussians\n")

    # ── 1. Reconstruction quality ──
    print("═══ RECONSTRUCTION QUALITY ═══")
    with torch.no_grad():
        reconstructed, loss, metrics = tokenizer(normalized)

    errors = (reconstructed - normalized).abs().cpu()

    for name, slc in PARAM_GROUPS.items():
        group_err = errors[:, slc]
        print(f"  {name:12s}  MAE: {group_err.mean():.6f}  max: {group_err.max():.6f}")

    print(f"\n  Overall MAE:  {errors.mean():.6f}")
    print(f"  Overall loss: {metrics['total_loss']:.6f}")

    verdict = "PASS" if errors.mean() < 0.05 else "WARN" if errors.mean() < 0.1 else "FAIL"
    print(f"\n  Verdict: {verdict}")
    if verdict == "FAIL":
        print("  → Reconstruction too lossy. The transformer will be predicting garbage tokens.")
        print("  → Try: increase codebook_size, increase latent_dim, train longer.")

    # ── 2. Codebook utilization ──
    print("\n═══ CODEBOOK UTILIZATION ═══")
    with torch.no_grad():
        indices, _ = tokenizer.encode(normalized)  # (N, num_codebooks)

    for cb_idx in range(tokenizer.get_num_codebooks()):
        codes = indices[:, cb_idx]
        unique_codes = codes.unique().numel()
        total_codes = tokenizer.get_vocab_size()
        utilization = unique_codes / total_codes * 100

        # entropy
        counts = torch.bincount(codes, minlength=total_codes).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum().item()
        max_entropy = np.log(total_codes)

        print(f"  Codebook {cb_idx}: {unique_codes}/{total_codes} codes used "
              f"({utilization:.1f}%) | entropy: {entropy:.2f}/{max_entropy:.2f}")

        if utilization < 30:
            print(f"    ⚠ LOW UTILIZATION — codebook collapse risk")

    # ── 3. Roundtrip test ──
    print("\n═══ ROUNDTRIP TEST (encode → decode) ═══")
    with torch.no_grad():
        indices, _ = tokenizer.encode(normalized[:100])
        decoded = tokenizer.decode(indices)

    roundtrip_error = (decoded - normalized[:100]).abs().mean().item()
    print(f"  Roundtrip MAE: {roundtrip_error:.6f}")

    if roundtrip_error < 0.01:
        print("  ✓ Excellent — tokens faithfully represent Gaussians")
    elif roundtrip_error < 0.05:
        print("  ✓ Good — minor quantization noise, should work fine")
    elif roundtrip_error < 0.1:
        print("  ~ Okay — noticeable quantization. Consider larger codebook")
    else:
        print("  ✗ Poor — tokens lose too much information. Fix before Phase 2")

    print()


if __name__ == "__main__":
    main()

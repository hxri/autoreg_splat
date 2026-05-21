"""
Phase 1: Train the Gaussian RVQ-VAE tokenizer.

Usage:
    python scripts/train_tokenizer.py --data_dir data/processed --epochs 100
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from autoreg_splat.models.tokenizer import GaussianRVQVAE
from autoreg_splat.utils.gaussian import normalize_gaussians, GAUSSIAN_DIM


def load_all_gaussians(data_dir: str) -> torch.Tensor:
    """Load and pool all Gaussians from all scenes."""
    data_path = Path(data_dir)
    all_params = []

    for scene_dir in sorted(data_path.iterdir()):
        npz_path = scene_dir / "gaussians.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        positions = data["positions"]
        rotations = data["rotations"]
        scales = data["scales"]
        opacities = data["opacities"]
        sh_coeffs = data["sh_coeffs"]

        params = np.concatenate(
            [positions, rotations, scales, opacities, sh_coeffs], axis=-1
        )
        all_params.append(params)

    if not all_params:
        raise ValueError(f"No Gaussian data found in {data_dir}")

    combined = torch.from_numpy(np.concatenate(all_params, axis=0)).float()
    print(f"Loaded {len(combined)} Gaussians from {len(all_params)} scenes")
    return combined


def train_tokenizer(cfg, data_dir: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    all_gaussians = load_all_gaussians(data_dir)
    normalized, stats = normalize_gaussians(all_gaussians)

    # save normalization stats
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out_path / "norm_stats.pt")

    dataset = TensorDataset(normalized)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.tokenizer_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # model
    model = GaussianRVQVAE(
        input_dim=cfg.tokenizer.input_dim,
        latent_dim=cfg.tokenizer.latent_dim,
        num_codebooks=cfg.tokenizer.num_codebooks,
        codebook_size=cfg.tokenizer.codebook_size,
        code_dim=cfg.tokenizer.code_dim,
        commitment_weight=cfg.tokenizer.commitment_weight,
        ema_decay=cfg.tokenizer.ema_decay,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.tokenizer_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.tokenizer_epochs
    )

    best_loss = float("inf")

    for epoch in range(cfg.training.tokenizer_epochs):
        model.train()
        epoch_metrics = {"recon_loss": 0, "commitment_loss": 0, "total_loss": 0}
        num_batches = 0

        for (batch,) in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
            batch = batch.to(device)
            reconstructed, loss, metrics = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] += v
            num_batches += 1

        scheduler.step()

        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        print(
            f"Epoch {epoch+1}/{cfg.training.tokenizer_epochs} — "
            f"loss: {avg_metrics['total_loss']:.6f} "
            f"(recon: {avg_metrics['recon_loss']:.6f}, "
            f"commit: {avg_metrics['commitment_loss']:.6f})"
        )

        if avg_metrics["total_loss"] < best_loss:
            best_loss = avg_metrics["total_loss"]
            torch.save(model.state_dict(), out_path / "tokenizer_best.pt")

    torch.save(model.state_dict(), out_path / "tokenizer_final.pt")
    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"Model saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="checkpoints/tokenizer")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train_tokenizer(cfg, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

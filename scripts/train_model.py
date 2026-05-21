"""
Phase 2: Train the autoregressive Gaussian transformer.

Usage:
    python scripts/train_model.py --data_dir data/processed \
        --tokenizer_ckpt checkpoints/tokenizer/tokenizer_best.pt
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from autoreg_splat.models.tokenizer import GaussianRVQVAE
from autoreg_splat.models.encoder import TopDownEncoder
from autoreg_splat.models.transformer import AutoregSplatTransformer
from autoreg_splat.data.dataset import GaussianSplatDataset, collate_fn


def train(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load tokenizer ──
    tokenizer = GaussianRVQVAE(
        input_dim=cfg.tokenizer.input_dim,
        latent_dim=cfg.tokenizer.latent_dim,
        num_codebooks=cfg.tokenizer.num_codebooks,
        codebook_size=cfg.tokenizer.codebook_size,
        code_dim=cfg.tokenizer.code_dim,
    ).to(device)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=device))
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # ── encoder ──
    encoder = TopDownEncoder(
        backbone=cfg.encoder.backbone,
        freeze_backbone=cfg.encoder.freeze_backbone,
        proj_dim=cfg.encoder.proj_dim,
    ).to(device)

    # ── transformer ──
    transformer = AutoregSplatTransformer(
        codebook_size=cfg.tokenizer.codebook_size,
        num_codebooks=cfg.tokenizer.num_codebooks,
        num_layers=cfg.transformer.num_layers,
        hidden_dim=cfg.transformer.hidden_dim,
        num_heads=cfg.transformer.num_heads,
        mlp_ratio=cfg.transformer.mlp_ratio,
        dropout=cfg.transformer.dropout,
        max_seq_len=cfg.transformer.max_seq_len,
        cross_attn_every_n=cfg.transformer.cross_attn_every_n,
    ).to(device)

    # ── dataset ──
    image_transform = TopDownEncoder.get_image_transform()
    dataset = GaussianSplatDataset(
        root_dir=args.data_dir,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_gaussians=cfg.data.max_gaussians,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── optimizer ──
    param_groups = [
        {"params": encoder.proj.parameters(), "lr": cfg.training.lr},
        {"params": transformer.parameters(), "lr": cfg.training.lr},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = len(loader) * cfg.training.max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    global_step = 0

    for epoch in range(cfg.training.max_epochs):
        transformer.train()
        encoder.train()
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
            images = batch["image"].to(device)
            gaussian_tokens = batch["gaussian_tokens"].to(device)  # (B, N, Q)
            num_gaussians = batch["num_gaussians"]

            # encode top-down image → conditioning tokens
            context = encoder(images)

            # classifier-free guidance: randomly drop conditioning
            if cfg.inference.cfg_dropout > 0:
                drop_mask = torch.rand(len(images)) < cfg.inference.cfg_dropout
                if drop_mask.any():
                    context[drop_mask] = 0.0

            # convert RVQ indices → token sequence
            tokens = transformer.rvq_indices_to_tokens(gaussian_tokens)

            # autoregressive loss
            loss, metrics = transformer.compute_loss(tokens, context)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(transformer.parameters()) + list(encoder.proj.parameters()),
                cfg.training.grad_clip,
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += metrics["ce_loss"]
            epoch_acc += metrics["token_accuracy"]
            num_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        print(
            f"Epoch {epoch+1}/{cfg.training.max_epochs} — "
            f"loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "transformer": transformer.state_dict(),
                "encoder_proj": encoder.proj.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }, out_path / "model_best.pt")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "transformer": transformer.state_dict(),
                "encoder_proj": encoder.proj.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }, out_path / f"model_epoch{epoch+1}.pt")

    print(f"Training complete. Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--tokenizer_ckpt", required=True)
    parser.add_argument("--output_dir", default="checkpoints/model")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train(cfg, args)


if __name__ == "__main__":
    main()

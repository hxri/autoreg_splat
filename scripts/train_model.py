"""
Phase 2: Train the autoregressive Gaussian transformer.

Supports multi-GPU via DataParallel and curriculum training
(start with fewer Gaussians, ramp up over training).

Usage:
    python scripts/train_model.py --data_dir data/processed \
        --tokenizer_ckpt checkpoints/tokenizer/tokenizer_best.pt

    # use specific GPUs
    CUDA_VISIBLE_DEVICES=0,1 python scripts/train_model.py ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from autoreg_splat.models.tokenizer import GaussianRVQVAE
from autoreg_splat.models.encoder import TopDownEncoder
from autoreg_splat.models.transformer import AutoregSplatTransformer
from autoreg_splat.data.dataset import GaussianSplatDataset, collate_fn


# batch size per curriculum stage — fewer Gaussians = longer sequences = smaller batch
STAGE_BATCH_SIZE = {
    256: 64,
    512: 32,
    1024: 16,
    2048: 8,
}

GRAD_ACCUM_TARGET_BATCH = 64  # effective batch size via gradient accumulation


def get_curriculum_stage(epoch: int, max_epochs: int, stages: list[int]) -> int:
    """Ramp up max_gaussians over training. Each stage gets equal epochs."""
    epochs_per_stage = max_epochs // len(stages)
    stage_idx = min(epoch // epochs_per_stage, len(stages) - 1)
    return stages[stage_idx]


def train(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s)")

    # ── load tokenizer ──
    tokenizer = GaussianRVQVAE(
        input_dim=cfg.tokenizer.input_dim,
        latent_dim=cfg.tokenizer.latent_dim,
        num_codebooks=cfg.tokenizer.num_codebooks,
        codebook_size=cfg.tokenizer.codebook_size,
        code_dim=cfg.tokenizer.code_dim,
    ).to(device)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=device, weights_only=True))
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

    # ── multi-GPU ──
    if num_gpus > 1:
        transformer = nn.DataParallel(transformer)
        encoder = nn.DataParallel(encoder)
        print(f"  DataParallel across GPUs: {list(range(num_gpus))}")

    # unwrap for saving / method access
    transformer_raw = transformer.module if isinstance(transformer, nn.DataParallel) else transformer

    # ── optimizer ──
    param_groups = [
        {"params": encoder.module.proj.parameters() if isinstance(encoder, nn.DataParallel) else encoder.proj.parameters(), "lr": cfg.training.lr},
        {"params": transformer_raw.parameters(), "lr": cfg.training.lr},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    curriculum_stages = list(cfg.training.curriculum_stages)
    image_transform = TopDownEncoder.get_image_transform()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    global_step = 0

    for epoch in range(cfg.training.max_epochs):
        # ── curriculum: pick max_gaussians and batch size for this epoch ──
        max_g = get_curriculum_stage(epoch, cfg.training.max_epochs, curriculum_stages)
        batch_size = STAGE_BATCH_SIZE.get(max_g, 8)
        if num_gpus > 1:
            batch_size *= num_gpus
        grad_accum_steps = max(1, GRAD_ACCUM_TARGET_BATCH // batch_size)

        # rebuild loader when stage changes
        if epoch == 0 or max_g != get_curriculum_stage(epoch - 1, cfg.training.max_epochs, curriculum_stages):
            print(f"\n── Curriculum stage: max_gaussians={max_g}, "
                  f"batch_size={batch_size}, grad_accum={grad_accum_steps} ──")
            dataset = GaussianSplatDataset(
                root_dir=args.data_dir,
                tokenizer=None,
                image_transform=image_transform,
                max_gaussians=max_g,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=True,
            )

            # recompute scheduler for remaining epochs
            remaining_epochs = cfg.training.max_epochs - epoch
            remaining_steps = len(loader) * remaining_epochs // grad_accum_steps
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(remaining_steps, 1),
            )

        transformer.train()
        encoder.train()
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} (G≤{max_g})", leave=False)):
            images = batch["image"].to(device)
            gaussian_params = batch["gaussian_params"].to(device)
            num_gaussians = batch["num_gaussians"]

            # truncate to current curriculum stage
            B, N, D = gaussian_params.shape
            N = min(N, max_g)
            gaussian_params = gaussian_params[:, :N, :]

            # tokenize on GPU
            with torch.no_grad():
                indices, _ = tokenizer.encode(gaussian_params.reshape(B * N, D))
                gaussian_tokens = indices.reshape(B, N, -1)

            # encode top-down image
            context = encoder(images)

            # classifier-free guidance dropout
            if cfg.inference.cfg_dropout > 0:
                drop_mask = torch.rand(B, device=device) < cfg.inference.cfg_dropout
                if drop_mask.any():
                    context[drop_mask] = 0.0

            # convert RVQ indices → token sequence
            tokens = transformer_raw.rvq_indices_to_tokens(gaussian_tokens)

            # autoregressive loss
            if isinstance(transformer, nn.DataParallel):
                logits = transformer(tokens[:, :-1], context)
                targets = tokens[:, 1:]
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, transformer_raw.vocab_size),
                    targets.reshape(-1),
                )
                with torch.no_grad():
                    acc = (logits.argmax(dim=-1) == targets).float().mean().item()
                metrics = {"ce_loss": loss.item(), "token_accuracy": acc}
            else:
                loss, metrics = transformer.compute_loss(tokens, context)

            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(transformer_raw.parameters()) +
                    list((encoder.module if isinstance(encoder, nn.DataParallel) else encoder).proj.parameters()),
                    cfg.training.grad_clip,
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += metrics["ce_loss"]
            epoch_acc += metrics["token_accuracy"]
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{cfg.training.max_epochs} — "
            f"loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}, "
            f"lr: {lr:.2e}, gaussians: ≤{max_g}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "transformer": transformer_raw.state_dict(),
                "encoder_proj": (encoder.module if isinstance(encoder, nn.DataParallel) else encoder).proj.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }, out_path / "model_best.pt")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "transformer": transformer_raw.state_dict(),
                "encoder_proj": (encoder.module if isinstance(encoder, nn.DataParallel) else encoder).proj.state_dict(),
                "optimizer": optimizer.state_dict(),
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

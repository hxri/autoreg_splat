"""
Evaluate a trained model: generate Gaussians from top-down images and visualize.

Usage:
    python scripts/evaluate.py --model_ckpt checkpoints/model/model_best.pt \
        --tokenizer_ckpt checkpoints/tokenizer/tokenizer_best.pt \
        --input_image test_topdown.png --output_dir outputs/
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from autoreg_splat.models.tokenizer import GaussianRVQVAE
from autoreg_splat.models.encoder import TopDownEncoder
from autoreg_splat.models.transformer import AutoregSplatTransformer
from autoreg_splat.inference.sampling import AutoregSplatSampler
from autoreg_splat.utils.gaussian import unpack_gaussians


def visualize_gaussians_topdown(params: torch.Tensor, image_size: int = 512) -> np.ndarray:
    """Render a simple top-down visualization of generated Gaussians."""
    g = unpack_gaussians(params)
    pos = g["positions"].cpu().numpy()
    sh = g["sh_coeffs"][:, :3].cpu().numpy()

    x, y = pos[:, 0], pos[:, 1]
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    scale = max(x_max - x_min, y_max - y_min, 1e-6)

    px = ((x - x_min) / scale * (image_size - 1)).astype(np.int32)
    py = ((y - y_min) / scale * (image_size - 1)).astype(np.int32)
    px = np.clip(px, 0, image_size - 1)
    py = np.clip(py, 0, image_size - 1)

    img = np.ones((image_size, image_size, 3), dtype=np.float32) * 0.95
    for i in range(len(px)):
        color = np.clip(sh[i] * 0.5 + 0.5, 0, 1)
        img[py[i], px[i]] = color

    return (img * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--tokenizer_ckpt", required=True)
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--max_gaussians", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load models
    tokenizer = GaussianRVQVAE(
        input_dim=cfg.tokenizer.input_dim,
        latent_dim=cfg.tokenizer.latent_dim,
        num_codebooks=cfg.tokenizer.num_codebooks,
        codebook_size=cfg.tokenizer.codebook_size,
        code_dim=cfg.tokenizer.code_dim,
    ).to(device)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=device))
    tokenizer.eval()

    encoder = TopDownEncoder(
        backbone=cfg.encoder.backbone,
        freeze_backbone=True,
        proj_dim=cfg.encoder.proj_dim,
    ).to(device)

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
    encoder.proj.load_state_dict(ckpt["encoder_proj"])
    transformer.eval()
    encoder.eval()

    # load input image
    image_transform = TopDownEncoder.get_image_transform()
    input_img = Image.open(args.input_image).convert("RGB")
    input_tensor = image_transform(input_img).unsqueeze(0).to(device)

    # sample
    sampler = AutoregSplatSampler(transformer, tokenizer, encoder)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_samples} samples...")

    for i in range(args.num_samples):
        result = sampler.sample(
            top_down_image=input_tensor,
            max_gaussians=args.max_gaussians,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        params = result["gaussian_params"]
        n = result["num_gaussians"]
        print(f"  Sample {i+1}: {n} Gaussians generated")

        # save visualization
        viz = visualize_gaussians_topdown(params)
        Image.fromarray(viz).save(out_path / f"sample_{i:03d}_topdown.png")

        # save raw Gaussians
        g = unpack_gaussians(params)
        np.savez(
            out_path / f"sample_{i:03d}_gaussians.npz",
            positions=g["positions"].cpu().numpy(),
            rotations=g["rotations"].cpu().numpy(),
            scales=g["scales"].cpu().numpy(),
            opacities=g["opacities"].cpu().numpy(),
            sh_coeffs=g["sh_coeffs"].cpu().numpy(),
        )

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

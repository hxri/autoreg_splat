"""
Dataset for autoregressive Gaussian splatting.

Each sample: (top_down_image, ordered_gaussian_params, gaussian_tokens)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .hilbert import hilbert_sort_gaussians_fast


class GaussianSplatDataset(Dataset):
    """Dataset of (top-down image, tokenized Gaussian sequence) pairs.

    Expected directory layout per scene:
        scene_dir/
            top_down.png         — bird's-eye-view image
            gaussians.npz        — arrays: positions, rotations, scales, opacities, sh_coeffs
            (optional) cameras.npz — GT camera poses + images for rendering loss
    """

    def __init__(
        self,
        root_dir: str,
        tokenizer=None,
        image_transform=None,
        max_gaussians: int = 2048,
        hilbert_bits: int = 10,
        sh_degree: int = 3,
    ):
        self.root = Path(root_dir)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_gaussians = max_gaussians
        self.hilbert_bits = hilbert_bits
        self.sh_degree = sh_degree
        self.sh_dim = (sh_degree + 1) ** 2 * 3

        self.scenes = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and (d / "gaussians.npz").exists()
        ])

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> dict:
        scene_dir = self.scenes[idx]

        # load top-down image
        img_path = scene_dir / "top_down.png"
        image = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # load Gaussian parameters
        data = np.load(scene_dir / "gaussians.npz")
        positions = torch.from_numpy(data["positions"]).float()     # (N, 3)
        rotations = torch.from_numpy(data["rotations"]).float()     # (N, 4)
        scales = torch.from_numpy(data["scales"]).float()           # (N, 3)
        opacities = torch.from_numpy(data["opacities"]).float()     # (N, 1)
        sh_coeffs = torch.from_numpy(data["sh_coeffs"]).float()     # (N, sh_dim)

        # sort by Hilbert curve
        sort_idx = hilbert_sort_gaussians_fast(positions, bits=self.hilbert_bits)
        positions = positions[sort_idx]
        rotations = rotations[sort_idx]
        scales = scales[sort_idx]
        opacities = opacities[sort_idx]
        sh_coeffs = sh_coeffs[sort_idx]

        # truncate to max_gaussians
        N = min(len(positions), self.max_gaussians)
        positions = positions[:N]
        rotations = rotations[:N]
        scales = scales[:N]
        opacities = opacities[:N]
        sh_coeffs = sh_coeffs[:N]

        # concatenate to single param tensor (N, 59)
        gaussian_params = torch.cat(
            [positions, rotations, scales, opacities, sh_coeffs], dim=-1
        )

        result = {
            "scene_id": scene_dir.name,
            "image": image,
            "gaussian_params": gaussian_params,
            "num_gaussians": N,
        }

        # tokenize if tokenizer is available
        if self.tokenizer is not None:
            with torch.no_grad():
                indices, _ = self.tokenizer.encode(gaussian_params)
            result["gaussian_tokens"] = indices  # (N, num_codebooks)

        return result


def collate_fn(batch: list[dict]) -> dict:
    """Custom collation — pads Gaussian sequences to the same length."""
    max_n = max(b["num_gaussians"] for b in batch)
    param_dim = batch[0]["gaussian_params"].shape[-1]

    images = torch.stack([b["image"] for b in batch])
    num_gaussians = torch.tensor([b["num_gaussians"] for b in batch])

    padded_params = torch.zeros(len(batch), max_n, param_dim)
    for i, b in enumerate(batch):
        n = b["num_gaussians"]
        padded_params[i, :n] = b["gaussian_params"]

    result = {
        "scene_id": [b["scene_id"] for b in batch],
        "image": images,
        "gaussian_params": padded_params,
        "num_gaussians": num_gaussians,
    }

    if "gaussian_tokens" in batch[0]:
        num_cb = batch[0]["gaussian_tokens"].shape[-1]
        padded_tokens = torch.zeros(len(batch), max_n, num_cb, dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["num_gaussians"]
            padded_tokens[i, :n] = b["gaussian_tokens"]
        result["gaussian_tokens"] = padded_tokens

    return result

"""Utilities for packing, unpacking, and normalizing 3D Gaussian parameters."""
from __future__ import annotations

import torch
import numpy as np


SH_DEGREE = 3
SH_DIM = (SH_DEGREE + 1) ** 2 * 3  # 48 for degree 3
GAUSSIAN_DIM = 3 + 4 + 3 + 1 + SH_DIM  # 59


def pack_gaussians(
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh_coeffs: torch.Tensor,
) -> torch.Tensor:
    """Pack individual Gaussian attributes into a flat tensor.

    Returns: (N, 59) tensor
    """
    if opacities.dim() == 1:
        opacities = opacities.unsqueeze(-1)
    return torch.cat([positions, rotations, scales, opacities, sh_coeffs], dim=-1)


def unpack_gaussians(params: torch.Tensor) -> dict[str, torch.Tensor]:
    """Unpack flat (N, 59) tensor into named attributes."""
    return {
        "positions": params[..., 0:3],
        "rotations": params[..., 3:7],
        "scales": params[..., 7:10],
        "opacities": params[..., 10:11],
        "sh_coeffs": params[..., 11:59],
    }


def normalize_gaussians(params: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Normalize Gaussian parameters to a canonical range.

    Positions → [0, 1], scales → log-space, rotations → unit quaternion,
    opacities → logit-space.

    Returns:
        normalized: (N, 59) normalized parameters
        stats: dict of normalization statistics for denormalization
    """
    g = unpack_gaussians(params)

    # position: normalize to [0, 1] cube
    pos_min = g["positions"].min(dim=0).values
    pos_max = g["positions"].max(dim=0).values
    pos_range = pos_max - pos_min
    pos_range[pos_range < 1e-8] = 1.0
    norm_pos = (g["positions"] - pos_min) / pos_range

    # rotation: ensure unit quaternion
    norm_rot = torch.nn.functional.normalize(g["rotations"], dim=-1)

    # scale: log-space (scales are always positive)
    norm_scale = torch.log(g["scales"].clamp(min=1e-7))

    # opacity: logit-space (opacity in [0, 1])
    opacity_clamped = g["opacities"].clamp(1e-4, 1 - 1e-4)
    norm_opacity = torch.log(opacity_clamped / (1 - opacity_clamped))

    # SH coefficients: standardize per-channel
    sh_mean = g["sh_coeffs"].mean(dim=0)
    sh_std = g["sh_coeffs"].std(dim=0).clamp(min=1e-6)
    norm_sh = (g["sh_coeffs"] - sh_mean) / sh_std

    normalized = torch.cat([norm_pos, norm_rot, norm_scale, norm_opacity, norm_sh], dim=-1)

    stats = {
        "pos_min": pos_min,
        "pos_range": pos_range,
        "sh_mean": sh_mean,
        "sh_std": sh_std,
    }

    return normalized, stats


def denormalize_gaussians(
    normalized: torch.Tensor, stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Inverse of normalize_gaussians."""
    g = unpack_gaussians(normalized)

    pos = g["positions"] * stats["pos_range"] + stats["pos_min"]
    rot = torch.nn.functional.normalize(g["rotations"], dim=-1)
    scale = torch.exp(g["scales"])
    opacity = torch.sigmoid(g["opacities"])
    sh = g["sh_coeffs"] * stats["sh_std"] + stats["sh_mean"]

    return torch.cat([pos, rot, scale, opacity, sh], dim=-1)


def load_ply_gaussians(ply_path: str) -> torch.Tensor:
    """Load 3D Gaussians from a .ply file (standard 3DGS format).

    Returns: (N, 59) tensor of Gaussian parameters
    """
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)

    # rotations stored as quaternion (w, x, y, z)
    rotations = np.stack([
        vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]
    ], axis=-1)

    scales = np.stack([
        vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]
    ], axis=-1)

    opacities = vertex["opacity"].reshape(-1, 1)

    # SH coefficients
    sh_names = [f"f_rest_{i}" for i in range(SH_DIM - 3)]
    sh_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1)
    if sh_names[0] in vertex.data.dtype.names:
        sh_rest = np.stack([vertex[n] for n in sh_names], axis=-1)
        sh_coeffs = np.concatenate([sh_dc, sh_rest], axis=-1)
    else:
        sh_coeffs = np.pad(sh_dc, ((0, 0), (0, SH_DIM - 3)), mode="constant")

    params = np.concatenate([positions, rotations, scales, opacities, sh_coeffs], axis=-1)
    return torch.from_numpy(params).float()


def save_gaussians_npz(params: torch.Tensor, path: str):
    """Save Gaussians as .npz for the dataset pipeline."""
    g = unpack_gaussians(params)
    np.savez(
        path,
        positions=g["positions"].numpy(),
        rotations=g["rotations"].numpy(),
        scales=g["scales"].numpy(),
        opacities=g["opacities"].numpy(),
        sh_coeffs=g["sh_coeffs"].numpy(),
    )

"""
Hilbert curve ordering for 3D Gaussians.

Maps 3D positions to a 1D index that preserves spatial locality,
giving the autoregressive model a meaningful sequential ordering.
"""

import torch
import numpy as np


def _xy2d_hilbert(n: int, x: int, y: int) -> int:
    """Convert 2D coordinates to Hilbert curve index."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def _xyz2d_hilbert_3d(bits: int, x: int, y: int, z: int) -> int:
    """Convert 3D coordinates to a 3D Hilbert curve index.

    Uses a simple interleaving approach: project to 2D Hilbert on (x,y),
    then interleave with z for a coarse 3D ordering.
    For a true 3D Hilbert curve we interleave bits from all three axes.
    """
    # bit interleaving (Morton-like with Hilbert on xy planes)
    n = 1 << bits
    xy_index = _xy2d_hilbert(n, x, y)
    # interleave z with xy_index for 3D locality
    result = 0
    for i in range(bits * 2):
        result |= ((xy_index >> i) & 1) << (2 * i + 1)
    for i in range(bits):
        result |= ((z >> i) & 1) << (2 * i)
    return result


def hilbert_sort_gaussians(
    positions: torch.Tensor, bits: int = 10
) -> torch.Tensor:
    """Sort Gaussians by 3D Hilbert curve index.

    Args:
        positions: (N, 3) Gaussian center positions
        bits: resolution — coordinates quantized to [0, 2^bits)
    Returns:
        sort_indices: (N,) permutation that sorts by Hilbert index
    """
    n = 1 << bits
    pos_np = positions.detach().cpu().numpy()

    # normalize to [0, n-1]
    mins = pos_np.min(axis=0)
    maxs = pos_np.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-8] = 1.0
    normalized = ((pos_np - mins) / ranges * (n - 1)).astype(np.int64)
    normalized = np.clip(normalized, 0, n - 1)

    indices = np.array([
        _xyz2d_hilbert_3d(bits, int(p[0]), int(p[1]), int(p[2]))
        for p in normalized
    ])

    sort_order = np.argsort(indices)
    return torch.from_numpy(sort_order).long()


def hilbert_sort_gaussians_fast(
    positions: torch.Tensor, bits: int = 10
) -> torch.Tensor:
    """Fast approximate 3D Hilbert sort using Morton codes (Z-order curve).

    Faster than true Hilbert for large point sets. Morton codes preserve
    most of the locality while being O(N log N) instead of O(N * bits²).
    """
    n = 1 << bits
    pos = positions.detach().cpu()

    mins = pos.min(dim=0).values
    maxs = pos.max(dim=0).values
    ranges = maxs - mins
    ranges[ranges < 1e-8] = 1.0
    normalized = ((pos - mins) / ranges * (n - 1)).long().clamp(0, n - 1)

    x, y, z = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    # Morton code: interleave bits of x, y, z
    morton = torch.zeros(len(x), dtype=torch.long)
    for i in range(bits):
        morton |= ((x >> i) & 1) << (3 * i + 2)
        morton |= ((y >> i) & 1) << (3 * i + 1)
        morton |= ((z >> i) & 1) << (3 * i)

    return morton.argsort()

"""
Generate synthetic Gaussian scenes for proof-of-concept training.

Creates simple geometric shapes (cubes, spheres, rooms) as sets of 3D Gaussians
with corresponding top-down views. No external datasets required.

Usage:
    python scripts/generate_synthetic.py --output_dir data/synthetic --num_scenes 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def random_color_sh(n: int, base_color: np.ndarray | None = None) -> np.ndarray:
    """Generate random SH coefficients. Only DC term (first 3) carry color."""
    sh = np.zeros((n, 48), dtype=np.float32)
    if base_color is not None:
        sh[:, 0:3] = base_color + np.random.randn(n, 3) * 0.05
    else:
        sh[:, 0:3] = np.random.rand(n, 3) * 0.5
    return sh


def make_box(center, size, n_points=200, color=None):
    """Generate Gaussians forming a filled box."""
    positions = center + (np.random.rand(n_points, 3) - 0.5) * size
    rotations = np.tile([1, 0, 0, 0], (n_points, 1)).astype(np.float32)
    scales = np.full((n_points, 3), 0.02, dtype=np.float32) * np.array(size)
    opacities = np.ones((n_points, 1), dtype=np.float32) * 0.9
    sh_coeffs = random_color_sh(n_points, color)
    return positions, rotations, scales, opacities, sh_coeffs


def make_sphere(center, radius, n_points=300, color=None):
    """Generate Gaussians forming a sphere."""
    # uniform points on sphere surface + some interior
    phi = np.random.rand(n_points) * 2 * np.pi
    costheta = np.random.rand(n_points) * 2 - 1
    r = radius * np.cbrt(np.random.rand(n_points))

    theta = np.arccos(costheta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    positions = np.stack([x, y, z], axis=-1).astype(np.float32) + center
    rotations = np.tile([1, 0, 0, 0], (n_points, 1)).astype(np.float32)
    scales = np.full((n_points, 3), 0.015 * radius, dtype=np.float32)
    opacities = np.ones((n_points, 1), dtype=np.float32) * 0.85
    sh_coeffs = random_color_sh(n_points, color)
    return positions, rotations, scales, opacities, sh_coeffs


def make_floor(center, extent, n_points=400, color=None):
    """Generate Gaussians forming a flat floor plane."""
    positions = np.zeros((n_points, 3), dtype=np.float32)
    positions[:, 0] = center[0] + (np.random.rand(n_points) - 0.5) * extent[0]
    positions[:, 1] = center[1] + (np.random.rand(n_points) - 0.5) * extent[1]
    positions[:, 2] = center[2] + np.random.randn(n_points) * 0.005
    rotations = np.tile([1, 0, 0, 0], (n_points, 1)).astype(np.float32)
    scales = np.full((n_points, 3), 0.03, dtype=np.float32)
    scales[:, 2] = 0.005  # flat in Z
    opacities = np.ones((n_points, 1), dtype=np.float32) * 0.95
    sh_coeffs = random_color_sh(n_points, color if color is not None else np.array([0.3, 0.3, 0.3]))
    return positions, rotations, scales, opacities, sh_coeffs


def make_wall(start, end, height, n_points=300, color=None):
    """Generate Gaussians forming a vertical wall between two XY points."""
    t = np.random.rand(n_points, 1)
    wall_pos = (1 - t) * start[:2] + t * end[:2]
    z = np.random.rand(n_points, 1) * height

    positions = np.zeros((n_points, 3), dtype=np.float32)
    positions[:, 0:2] = wall_pos
    positions[:, 2:3] = z
    # add thickness noise perpendicular to wall
    wall_dir = end[:2] - start[:2]
    wall_dir = wall_dir / (np.linalg.norm(wall_dir) + 1e-8)
    perp = np.array([-wall_dir[1], wall_dir[0]])
    positions[:, 0:2] += perp * np.random.randn(n_points, 1) * 0.02

    rotations = np.tile([1, 0, 0, 0], (n_points, 1)).astype(np.float32)
    scales = np.full((n_points, 3), 0.02, dtype=np.float32)
    opacities = np.ones((n_points, 1), dtype=np.float32) * 0.9
    sh_coeffs = random_color_sh(n_points, color if color is not None else np.array([0.5, 0.5, 0.5]))
    return positions, rotations, scales, opacities, sh_coeffs


def generate_simple_room(rng: np.random.Generator) -> dict:
    """Generate a simple room with floor, walls, and random furniture."""
    room_w = rng.uniform(2, 5)
    room_d = rng.uniform(2, 5)
    room_h = rng.uniform(2.5, 4)

    parts = []

    # floor
    floor_color = rng.random(3) * 0.3 + 0.2
    parts.append(make_floor(
        center=np.array([room_w/2, room_d/2, 0]),
        extent=np.array([room_w, room_d]),
        n_points=500,
        color=floor_color,
    ))

    # 4 walls
    wall_color = rng.random(3) * 0.3 + 0.4
    corners = [
        np.array([0, 0, 0]),
        np.array([room_w, 0, 0]),
        np.array([room_w, room_d, 0]),
        np.array([0, room_d, 0]),
    ]
    for i in range(4):
        parts.append(make_wall(corners[i], corners[(i+1) % 4], room_h, 200, wall_color))

    # random furniture: 1-4 boxes
    n_furniture = rng.integers(1, 5)
    for _ in range(n_furniture):
        fw = rng.uniform(0.3, 1.0)
        fd = rng.uniform(0.3, 1.0)
        fh = rng.uniform(0.3, 1.5)
        fx = rng.uniform(fw/2 + 0.2, room_w - fw/2 - 0.2)
        fy = rng.uniform(fd/2 + 0.2, room_d - fd/2 - 0.2)
        fcolor = rng.random(3) * 0.6 + 0.2
        parts.append(make_box(
            center=np.array([fx, fy, fh/2]),
            size=np.array([fw, fd, fh]),
            n_points=rng.integers(100, 300),
            color=fcolor,
        ))

    # concatenate all parts
    all_pos = np.concatenate([p[0] for p in parts], axis=0)
    all_rot = np.concatenate([p[1] for p in parts], axis=0)
    all_scale = np.concatenate([p[2] for p in parts], axis=0)
    all_opacity = np.concatenate([p[3] for p in parts], axis=0)
    all_sh = np.concatenate([p[4] for p in parts], axis=0)

    return {
        "positions": all_pos.astype(np.float32),
        "rotations": all_rot.astype(np.float32),
        "scales": all_scale.astype(np.float32),
        "opacities": all_opacity.astype(np.float32),
        "sh_coeffs": all_sh.astype(np.float32),
    }


def render_topdown_simple(positions: np.ndarray, image_size: int = 256) -> np.ndarray:
    """Simple top-down density render."""
    x, y = positions[:, 0], positions[:, 1]
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    scale = max(x_max - x_min, y_max - y_min)

    px = ((x - x_min) / scale * (image_size - 1)).astype(np.int32)
    py = ((y - y_min) / scale * (image_size - 1)).astype(np.int32)
    px = np.clip(px, 0, image_size - 1)
    py = np.clip(py, 0, image_size - 1)

    z = positions[:, 2]
    z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)

    img = np.zeros((image_size, image_size, 3), dtype=np.float32)
    count = np.zeros((image_size, image_size), dtype=np.float32)

    for i in range(len(px)):
        img[py[i], px[i]] += np.array([z_norm[i], 0.5, 1 - z_norm[i]])
        count[py[i], px[i]] += 1

    mask = count > 0
    for c in range(3):
        img[:, :, c][mask] /= count[mask]

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/synthetic")
    parser.add_argument("--num_scenes", type=int, default=500)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    print(f"Generating {args.num_scenes} synthetic scenes...")

    for i in range(args.num_scenes):
        scene_dir = out_path / f"scene_{i:05d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        scene = generate_simple_room(rng)

        np.savez(scene_dir / "gaussians.npz", **scene)

        topdown = render_topdown_simple(scene["positions"], args.image_size)
        Image.fromarray(topdown).save(scene_dir / "top_down.png")

        if (i + 1) % 50 == 0:
            n = len(scene["positions"])
            print(f"  Generated {i+1}/{args.num_scenes} scenes ({n} Gaussians in latest)")

    print(f"Done. Scenes saved to {out_path}")


if __name__ == "__main__":
    main()

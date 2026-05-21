"""
Preprocess scenes for training: load 3DGS .ply files, render top-down views, save as dataset.

Usage:
    python scripts/preprocess.py --input_dir data/raw_scenes --output_dir data/processed

Expected input: directories each containing a 3DGS point_cloud.ply file.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from autoreg_splat.utils.gaussian import load_ply_gaussians, save_gaussians_npz, unpack_gaussians


def render_topdown(positions: np.ndarray, image_size: int = 256) -> np.ndarray:
    """Render a simple top-down occupancy/density image from Gaussian positions.

    Projects all Gaussians onto the XY plane and creates a density map.
    """
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    scale = max(x_range, y_range)

    # normalize to [0, image_size-1]
    px = ((x - x_min) / scale * (image_size - 1)).astype(np.int32)
    py = ((y - y_min) / scale * (image_size - 1)).astype(np.int32)
    px = np.clip(px, 0, image_size - 1)
    py = np.clip(py, 0, image_size - 1)

    # create density image with height-based coloring
    z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)

    img = np.zeros((image_size, image_size, 3), dtype=np.float32)
    count = np.zeros((image_size, image_size), dtype=np.float32)

    for i in range(len(px)):
        img[py[i], px[i], 0] += z_norm[i]       # height → red
        img[py[i], px[i], 1] += 0.5              # constant green
        img[py[i], px[i], 2] += 1.0 - z_norm[i]  # inverse height → blue
        count[py[i], px[i]] += 1.0

    mask = count > 0
    for c in range(3):
        img[:, :, c][mask] /= count[mask]

    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def preprocess_scene(ply_path: Path, output_dir: Path, image_size: int = 256):
    """Process a single scene: load PLY, render top-down, save."""
    params = load_ply_gaussians(str(ply_path))
    g = unpack_gaussians(params)

    # prune near-zero opacity
    opacity_mask = g["opacities"].squeeze(-1) > 0.01
    params = params[opacity_mask]
    g = unpack_gaussians(params)

    # render top-down view
    positions_np = g["positions"].numpy()
    topdown = render_topdown(positions_np, image_size)
    Image.fromarray(topdown).save(output_dir / "top_down.png")

    # save Gaussians
    save_gaussians_npz(params, str(output_dir / "gaussians.npz"))

    return len(params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ply_files = list(input_path.rglob("point_cloud.ply")) + list(input_path.rglob("*.ply"))
    # deduplicate
    ply_files = list({str(p): p for p in ply_files}.values())

    print(f"Found {len(ply_files)} PLY files")

    for ply_path in tqdm(ply_files):
        scene_name = ply_path.parent.name
        scene_out = output_path / scene_name
        scene_out.mkdir(parents=True, exist_ok=True)

        try:
            n = preprocess_scene(ply_path, scene_out, args.image_size)
            print(f"  {scene_name}: {n} Gaussians")
        except Exception as e:
            print(f"  {scene_name}: FAILED — {e}")


if __name__ == "__main__":
    main()

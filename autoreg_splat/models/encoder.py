"""
Top-down image encoder using DINOv2 backbone.

Extracts spatial conditioning tokens from a bird's-eye-view image.
"""

import torch
import torch.nn as nn


class TopDownEncoder(nn.Module):
    """Encodes a top-down image into spatial conditioning tokens for the AR transformer."""

    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        freeze_backbone: bool = True,
        proj_dim: int = 768,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone)
        backbone_dim = self.backbone.embed_dim

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) top-down images, normalized for DINOv2
        Returns:
            tokens: (B, M, proj_dim) spatial conditioning tokens
                    where M = (H/14) * (W/14) for ViT-B/14
        """
        with torch.set_grad_enabled(self.proj[0].weight.requires_grad):
            features = self.backbone.forward_features(images)
            patch_tokens = features["x_norm_patchtokens"]  # (B, M, backbone_dim)

        return self.proj(patch_tokens)

    @staticmethod
    def get_image_transform():
        """Returns the preprocessing transform for input images."""
        from torchvision import transforms

        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(252),  # divisible by 14
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

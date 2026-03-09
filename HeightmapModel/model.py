"""
U-Net with pretrained EfficientNet-B0 encoder for contour map → heightmap regression.

Architecture:
    - Encoder: EfficientNet-B0 (pretrained on ImageNet) via timm
    - Decoder: 5-stage upsampling with skip connections
    - Input:  (B, 3, 512, 512) RGB contour map image
    - Output: (B, 1, 512, 512) predicted heightmap
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Single decoder stage: upsample + concat skip + conv + conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Channel attention (squeeze-and-excitation) to help the decoder
        # decide which skip features matter
        self.se = SEBlock(out_channels, reduction=16)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            # Handle size mismatch from odd encoder dimensions
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(-2, -1), keepdim=True)
        w = torch.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class HeightmapUNet(nn.Module):
    """
    U-Net with EfficientNet-B0 encoder for heightmap regression.

    EfficientNet-B0 feature stages (for 512×512 input):
        stage 0: (B,  16, 256, 256)  — after stem + first block   (stride 2)
        stage 1: (B,  24, 128, 128)  — MBConv block               (stride 4)
        stage 2: (B,  40, 64,  64)   — MBConv block               (stride 8)
        stage 3: (B, 112, 32,  32)   — MBConv block               (stride 16)
        stage 4: (B, 320, 16,  16)   — MBConv block               (stride 32)

    Decoder mirrors this back up to 512×512.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # --- Encoder ---
        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Get actual channel counts from the encoder
        # (these depend on the specific model variant)
        encoder_channels = self.encoder.feature_info.channels()
        # e.g. [16, 24, 40, 112, 320] for efficientnet_b0

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[4], 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # --- Decoder ---
        # Each block: upsample bottleneck/prev + concat skip → conv
        # Going from deepest (smallest spatial) to shallowest (largest spatial)
        self.decoder4 = DecoderBlock(
            256, encoder_channels[4], 128
        )  # 16→32,  skip=stage4
        self.decoder3 = DecoderBlock(
            128, encoder_channels[3], 64
        )  # 32→64,  skip=stage3
        self.decoder2 = DecoderBlock(
            64, encoder_channels[2], 32
        )  # 64→128, skip=stage2
        self.decoder1 = DecoderBlock(
            32, encoder_channels[1], 16
        )  # 128→256, skip=stage1
        self.decoder0 = DecoderBlock(
            16, encoder_channels[0], 16
        )  # 256→512, skip=stage0

        # --- Head ---
        # Final conv to produce single-channel heightmap
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),  # Output in [0, 1] range, rescale to actual heights outside
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image, expected 512×512

        Returns:
            (B, 1, H, W) predicted heightmap in [0, 1]
        """
        input_size = x.shape[-2:]

        # Encoder forward — collect multi-scale features
        features = self.encoder(x)
        # features[0]: stride 2   (256×256)
        # features[1]: stride 4   (128×128)
        # features[2]: stride 8   (64×64)
        # features[3]: stride 16  (32×32)
        # features[4]: stride 32  (16×16)

        # Bottleneck
        x = self.bottleneck(features[4])

        # Decoder — upsample + skip connections (deepest to shallowest)
        x = self.decoder4(x, features[4])  # → 32×32
        x = self.decoder3(x, features[3])  # → 64×64
        x = self.decoder2(x, features[2])  # → 128×128
        x = self.decoder1(x, features[1])  # → 256×256
        x = self.decoder0(x, features[0])  # → 512×512

        # Head
        x = self.head(x)

        # Ensure output matches input spatial size exactly
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return x


class HeightmapLoss(nn.Module):
    """
    Combined loss for heightmap regression:
        - L1 loss: main pixel-wise reconstruction
        - Gradient loss: penalizes incorrect slopes (critical for contour consistency)
        - SSIM-like structural loss: penalizes structural differences
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        gradient_weight: float = 0.5,
        structure_weight: float = 0.2,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.structure_weight = structure_weight

        # Sobel kernels for gradient computation
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        # Shape: (1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _spatial_gradient(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial gradients using Sobel filters."""
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return gx, gy

    def _structure_loss(
        self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11
    ) -> torch.Tensor:
        """Local structure comparison — correlation between pred and target in sliding windows."""
        pad = window_size // 2
        # Use avg pooling as a fast windowed mean
        mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
        mu_t = F.avg_pool2d(target, window_size, stride=1, padding=pad)

        sigma_p = F.avg_pool2d(pred**2, window_size, stride=1, padding=pad) - mu_p**2
        sigma_t = F.avg_pool2d(target**2, window_size, stride=1, padding=pad) - mu_t**2
        sigma_pt = (
            F.avg_pool2d(pred * target, window_size, stride=1, padding=pad)
            - mu_p * mu_t
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / (
            (mu_p**2 + mu_t**2 + C1) * (sigma_p + sigma_t + C2)
        )

        return 1.0 - ssim.mean()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred:   (B, 1, H, W) predicted heightmap
            target: (B, 1, H, W) ground truth heightmap

        Returns:
            dict with 'loss' (total) and individual components for logging
        """
        # L1 loss
        l1 = F.l1_loss(pred, target)

        # Gradient loss — slopes should match
        pred_gx, pred_gy = self._spatial_gradient(pred)
        target_gx, target_gy = self._spatial_gradient(target)
        grad_loss = F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)

        # Structure loss
        struct_loss = self._structure_loss(pred, target)

        total = (
            self.l1_weight * l1
            + self.gradient_weight * grad_loss
            + self.structure_weight * struct_loss
        )

        return {
            "loss": total,
            "l1": l1.detach(),
            "gradient": grad_loss.detach(),
            "structure": struct_loss.detach(),
        }


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HeightmapUNet(pretrained=False).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Test forward pass
    dummy = torch.randn(2, 3, 512, 512, device=device)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 1, 512, 512), f"Unexpected output shape: {out.shape}"

    # Test loss
    criterion = HeightmapLoss().to(device)
    target = torch.rand(2, 1, 512, 512, device=device)
    losses = criterion(out, target)
    print(f"Loss:      {losses['loss'].item():.4f}")
    print(f"  L1:      {losses['l1'].item():.4f}")
    print(f"  Grad:    {losses['gradient'].item():.4f}")
    print(f"  Struct:  {losses['structure'].item():.4f}")

    print("\n✓ All checks passed")

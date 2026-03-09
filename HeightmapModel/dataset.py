"""
Dataset loader for contour map → heightmap training.

Reads the output of DatasetGenerator:
    - {i}.jpg           : contour line image (512×512, RGB)
    - {i}.data          : raw heightmap (512×512, float32, row-major, y-then-x)
    - {i}.metadata.json : sample metadata

Heavy augmentation pipeline to bridge the domain gap between
clean synthetic renders and noisy phone camera photos.

Augmentation tiers:
    Tier 1 (synthetic variation): color/thickness of contour lines, backgrounds
    Tier 2 (camera simulation):   perspective, lens distortion, lighting, noise,
                                  blur, JPEG artifacts, shadows, paper texture
"""

import json
import os
import struct
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_heightmap_binary(filepath: str, width: int, height: int) -> np.ndarray:
    """
    Load a raw binary heightmap (.data) written by DatasetGenerator.
    Format: row-major float32, y outer loop, x inner loop.
    Returns (height, width) float32 array.
    """
    num_floats = width * height
    expected_bytes = num_floats * 4

    with open(filepath, "rb") as f:
        raw = f.read()

    if len(raw) != expected_bytes:
        raise ValueError(
            f"Expected {expected_bytes} bytes for {width}×{height} heightmap, "
            f"got {len(raw)} bytes in {filepath}"
        )

    values = struct.unpack(f"<{num_floats}f", raw)
    # DatasetGenerator writes: for y in range(height): for x in range(width): write(heightmap[x, y])
    # So the binary is in (y, x) order → reshape directly to (height, width)
    return np.array(values, dtype=np.float32).reshape(height, width)


def load_metadata(filepath: str) -> dict:
    """Load sample metadata JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------


def _build_spatial_transforms(img_size: int = 512) -> A.Compose:
    """
    Spatial transforms applied to BOTH image and heightmap simultaneously.
    These simulate camera angle, paper warping, and framing.
    """
    return A.Compose(
        [
            # Random crop + resize — simulate zooming into part of the map
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.5, 1.0),
                ratio=(0.8, 1.2),
                interpolation=cv2.INTER_LINEAR,
                p=0.5,
            ),
            # Rotation — map won't be axis-aligned in a phone photo
            A.Rotate(
                limit=15,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
            # Perspective warp — phone is never perfectly overhead
            A.Perspective(
                scale=(0.02, 0.08),
                keep_size=True,
                fit_output=False,
                interpolation=cv2.INTER_LINEAR,
                p=0.5,
            ),
            # Affine — slight shear and translation
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                shear=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),
            # Elastic deformation — paper isn't perfectly flat
            A.ElasticTransform(
                alpha=30,
                sigma=5,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.2,
            ),
            # Flip (contour maps are rotation-invariant for height)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
        additional_targets={"heightmap": "mask"},
    )


def _build_photometric_transforms() -> A.Compose:
    """
    Photometric transforms applied ONLY to the image (not heightmap).
    These simulate camera sensor, lighting, and paper properties.
    """
    return A.Compose(
        [
            # --- Color / Lighting ---
            # Brightness + contrast variation (lighting conditions)
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.7,
            ),
            # Color temperature / white balance shift
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=15,
                p=0.5,
            ),
            # Simulate uneven lighting (lamp casting gradient across paper)
            A.RandomToneCurve(scale=0.15, p=0.3),
            # Random gamma — screen vs paper brightness response
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            # --- Shadows ---
            # Simulate finger/object shadow falling on map
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 3),
                shadow_dimension=5,
                p=0.25,
            ),
            # --- Noise (camera sensor) ---
            A.OneOf(
                [
                    A.GaussNoise(std_range=(5.0 / 255.0, 25.0 / 255.0), p=1.0),
                    A.ISONoise(
                        color_shift=(0.01, 0.05),
                        intensity=(0.1, 0.4),
                        p=1.0,
                    ),
                    A.MultiplicativeNoise(
                        multiplier=(0.9, 1.1),
                        per_channel=True,
                        p=1.0,
                    ),
                ],
                p=0.6,
            ),
            # --- Blur (focus / motion) ---
            A.OneOf(
                [
                    # Defocus blur — phone camera out of focus
                    A.Defocus(radius=(2, 4), p=1.0),
                    # Motion blur — hand shake
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    # General Gaussian blur
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    # Zoom blur — moving phone while shooting
                    A.ZoomBlur(max_factor=1.05, p=1.0),
                ],
                p=0.35,
            ),
            # --- Compression artifacts (JPEG from phone) ---
            A.ImageCompression(
                quality_range=(40, 90),
                p=0.4,
            ),
            # --- Simulate paper texture / print quality ---
            # Sharpen — some phone cameras aggressively sharpen
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
            # Slight posterize — simulate limited color depth / print
            A.Posterize(num_bits=(5, 7), p=0.1),
            # Channel shuffle — rare but helps generalization
            A.ChannelShuffle(p=0.02),
            # Normalize last — standard ImageNet normalization
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _build_contour_color_transforms() -> A.Compose:
    """
    Transforms that vary the appearance of the contour lines themselves.
    Simulates maps drawn in different colors, on different paper, etc.
    Applied before photometric transforms.
    """
    return A.Compose(
        [
            # Invert — some maps have white lines on dark background
            A.InvertImg(p=0.05),
            # Convert to grayscale — many real maps are black-and-white
            A.ToGray(p=0.3),
            # Color jitter specifically for contour color variation
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.3,
                hue=0.15,
                p=0.5,
            ),
            # Simulate different paper colors (cream, yellow, aged)
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=15,
                b_shift_limit=20,
                p=0.4,
            ),
        ]
    )


def _build_val_transforms() -> A.Compose:
    """Minimal transforms for validation — only normalization."""
    return A.Compose(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Cutout / CoarseDropout — simulate partial occlusion (finger, pen, ruler)
# ---------------------------------------------------------------------------


def _build_occlusion_transforms() -> A.Compose:
    """Simulate objects occluding part of the map."""
    return A.Compose(
        [
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(20, 80),
                hole_width_range=(20, 80),
                fill="random",
                p=0.2,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class ContourHeightmapDataset(Dataset):
    """
    PyTorch dataset for contour map → heightmap pairs.

    Expects directory structure from DatasetGenerator:
        data_dir/
            0.jpg
            0.data
            0.metadata.json
            1.jpg
            1.data
            ...

    Args:
        data_dir:     Path to generated dataset directory
        split:        'train' or 'val'
        val_fraction: Fraction of data to use for validation (default 0.1)
        img_size:     Target image size (default 512)
        augment:      Whether to apply augmentations (auto: True for train, False for val)
        max_samples:  Limit number of samples (for debugging)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_fraction: float = 0.1,
        img_size: int = 512,
        augment: bool | None = None,
        max_samples: int | None = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment if augment is not None else (split == "train")

        # Discover all samples by looking for .data files
        all_data_files = sorted(self.data_dir.glob("*.data"))
        if not all_data_files:
            raise FileNotFoundError(
                f"No .data files found in {self.data_dir}. Run DatasetGenerator first."
            )

        # Extract integer indices and verify paired files exist
        self.samples: list[dict] = []
        for data_file in all_data_files:
            stem = data_file.stem  # e.g. "42"
            jpg_path = self.data_dir / f"{stem}.jpg"
            meta_path = self.data_dir / f"{stem}.metadata.json"

            if not jpg_path.exists():
                print(f"Warning: missing {jpg_path}, skipping sample {stem}")
                continue

            self.samples.append(
                {
                    "index": stem,
                    "image_path": str(jpg_path),
                    "heightmap_path": str(data_file),
                    "metadata_path": str(meta_path) if meta_path.exists() else None,
                }
            )

        if not self.samples:
            raise FileNotFoundError(
                f"No complete samples (jpg + data pairs) found in {self.data_dir}"
            )

        # Split into train/val deterministically
        n_total = len(self.samples)
        n_val = max(1, int(n_total * val_fraction))
        n_train = n_total - n_val

        if split == "train":
            self.samples = self.samples[:n_train]
        elif split == "val":
            self.samples = self.samples[n_train:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # Apply max_samples limit
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Read image dimensions from first sample's metadata (or use default)
        self._data_width = 512
        self._data_height = 512
        if self.samples[0]["metadata_path"]:
            try:
                meta = load_metadata(self.samples[0]["metadata_path"])
                self._data_width = meta.get("Width", 512)
                self._data_height = meta.get("Height", 512)
            except (json.JSONDecodeError, KeyError):
                pass

        # Build augmentation pipelines
        self.spatial_transforms = _build_spatial_transforms(self.img_size)
        self.contour_color_transforms = _build_contour_color_transforms()
        self.photometric_transforms = _build_photometric_transforms()
        self.occlusion_transforms = _build_occlusion_transforms()
        self.val_transforms = _build_val_transforms()

        print(
            f"Dataset [{split}]: {len(self.samples)} samples from {self.data_dir}, "
            f"augment={self.augment}, img_size={self.img_size}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image (BGR → RGB)
        image = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Failed to load image: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load heightmap
        heightmap = load_heightmap_binary(
            sample["heightmap_path"], self._data_width, self._data_height
        )

        # Resize to target size if needed
        if image.shape[:2] != (self.img_size, self.img_size):
            image = cv2.resize(
                image,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_LINEAR,
            )
        if heightmap.shape != (self.img_size, self.img_size):
            heightmap = cv2.resize(
                heightmap,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_LINEAR,
            )

        # Normalize heightmap to [0, 1]
        h_min, h_max = heightmap.min(), heightmap.max()
        if h_max - h_min > 1e-6:
            heightmap = (heightmap - h_min) / (h_max - h_min)
        else:
            heightmap = np.zeros_like(heightmap)

        if self.augment:
            # 1. Spatial transforms (applied to both image and heightmap)
            spatial_result = self.spatial_transforms(image=image, heightmap=heightmap)
            image = spatial_result["image"]
            heightmap = spatial_result["heightmap"]

            # 2. Contour color variation (image only)
            image = self.contour_color_transforms(image=image)["image"]

            # 3. Occlusion (image only — simulate finger/pen/ruler)
            image = self.occlusion_transforms(image=image)["image"]

            # 4. Photometric transforms (image only — camera simulation + normalize)
            image = self.photometric_transforms(image=image)["image"]
        else:
            # Validation: just normalize
            image = self.val_transforms(image=image)["image"]

        # Convert to tensors
        # Image: albumentations Normalize already converts to float32, we just need CHW
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        # Heightmap: (H, W) → (1, H, W)
        heightmap_tensor = torch.from_numpy(heightmap).unsqueeze(0).float()

        result = {
            "image": image_tensor,
            "heightmap": heightmap_tensor,
            "index": sample["index"],
        }

        return result


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def build_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 512,
    val_fraction: float = 0.1,
    max_samples: int | None = None,
    pin_memory: bool = True,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train and validation DataLoaders.

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = ContourHeightmapDataset(
        data_dir=data_dir,
        split="train",
        val_fraction=val_fraction,
        img_size=img_size,
        max_samples=max_samples,
    )
    val_dataset = ContourHeightmapDataset(
        data_dir=data_dir,
        split="val",
        val_fraction=val_fraction,
        img_size=img_size,
        max_samples=max_samples,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Test dataset loading and augmentation"
    )
    parser.add_argument(
        "data_dir", type=str, help="Path to generated dataset directory"
    )
    parser.add_argument(
        "--num-samples", type=int, default=4, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--no-augment", action="store_true", help="Disable augmentation"
    )
    args = parser.parse_args()

    dataset = ContourHeightmapDataset(
        data_dir=args.data_dir,
        split="train",
        augment=not args.no_augment,
        max_samples=args.num_samples * 2,
    )

    n = min(args.num_samples, len(dataset))
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        sample = dataset[i]
        img = sample["image"]
        hmap = sample["heightmap"]

        # Denormalize image for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = (img * std + mean).clamp(0, 1)

        axes[i, 0].imshow(img_display.permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f"Contour image (sample {sample['index']})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(hmap.squeeze().numpy(), cmap="terrain")
        axes[i, 1].set_title("Heightmap (ground truth)")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("dataset_preview.png", dpi=100)
    plt.show()
    print(f"Saved preview to dataset_preview.png")

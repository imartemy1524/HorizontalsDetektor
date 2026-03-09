"""
Inference script for contour map → heightmap prediction.

Loads a trained model checkpoint and runs predictions on new images.
Supports single images, directories of images, and outputs heightmaps
as PNG visualizations, raw numpy arrays, or both.

Usage:
    # Single image
    python predict.py --checkpoint runs/best_ema.pt --input photo.jpg --output result.png

    # Directory of images
    python predict.py --checkpoint runs/best_ema.pt --input photos/ --output results/

    # Save raw numpy heightmap + visualization
    python predict.py --checkpoint runs/best_ema.pt --input photo.jpg --output result --save-raw --save-vis

    # Use specific device
    python predict.py --checkpoint runs/best_ema.pt --input photo.jpg --output result.png --device cuda:0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from model import HeightmapUNet

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_model(
    checkpoint_path: str,
    device: torch.device,
    img_size: int | None = None,
) -> tuple[HeightmapUNet, int]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Target device
        img_size: Override image size (if None, uses checkpoint's value or 512)

    Returns:
        (model, img_size)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine image size
    if img_size is None:
        img_size = checkpoint.get("img_size", 512)

    # Build model
    model = HeightmapUNet(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Print checkpoint info
    if "best_val_loss" in checkpoint:
        print(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")
    if "epoch" in checkpoint:
        print(f"  Trained for:   {checkpoint['epoch'] + 1} epochs")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:    {total_params:,}")
    print(f"  Image size:    {img_size}")

    return model, img_size


def preprocess_image(
    image_path: str,
    img_size: int = 512,
) -> tuple[torch.Tensor, np.ndarray, tuple[int, int]]:
    """
    Load and preprocess an image for model input.

    Args:
        image_path: Path to input image
        img_size: Target size for model input

    Returns:
        (input_tensor, original_image_rgb, original_size_hw)
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Failed to load image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]  # (H, W)

    # Resize to model input size
    resized = cv2.resize(
        image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR
    )

    # Normalize with ImageNet stats (matching training pipeline)
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std

    # Convert to tensor: (H, W, 3) → (1, 3, H, W)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()

    return tensor, image_rgb, original_size


@torch.no_grad()
def predict_single(
    model: HeightmapUNet,
    input_tensor: torch.Tensor,
    device: torch.device,
    original_size: tuple[int, int] | None = None,
    use_tta: bool = False,
) -> np.ndarray:
    """
    Run prediction on a single preprocessed image tensor.

    Args:
        model: Trained model
        input_tensor: (1, 3, H, W) preprocessed image tensor
        device: Compute device
        original_size: If provided, resize output to (H, W)
        use_tta: Use test-time augmentation (average over flips)

    Returns:
        Heightmap as (H, W) numpy float32 array in [0, 1]
    """
    input_tensor = input_tensor.to(device)

    if use_tta:
        # Test-time augmentation: average predictions over flips
        predictions = []

        # Original
        pred = model(input_tensor)
        predictions.append(pred)

        # Horizontal flip
        flipped_h = torch.flip(input_tensor, dims=[-1])
        pred_h = model(flipped_h)
        pred_h = torch.flip(pred_h, dims=[-1])
        predictions.append(pred_h)

        # Vertical flip
        flipped_v = torch.flip(input_tensor, dims=[-2])
        pred_v = model(flipped_v)
        pred_v = torch.flip(pred_v, dims=[-2])
        predictions.append(pred_v)

        # Both flips
        flipped_hv = torch.flip(input_tensor, dims=[-1, -2])
        pred_hv = model(flipped_hv)
        pred_hv = torch.flip(pred_hv, dims=[-1, -2])
        predictions.append(pred_hv)

        # Average
        prediction = torch.stack(predictions).mean(dim=0)
    else:
        prediction = model(input_tensor)

    # (1, 1, H, W) → (H, W)
    heightmap = prediction.squeeze().cpu().numpy()

    # Resize to original image size if requested
    if original_size is not None and heightmap.shape != original_size:
        heightmap = cv2.resize(
            heightmap,
            (original_size[1], original_size[0]),  # cv2 uses (W, H)
            interpolation=cv2.INTER_LINEAR,
        )

    return heightmap


def save_heightmap_visualization(
    heightmap: np.ndarray,
    output_path: str,
    original_image: np.ndarray | None = None,
    cmap: str = "terrain",
    dpi: int = 150,
):
    """
    Save a heightmap as a colorized PNG visualization.

    If original_image is provided, creates a side-by-side comparison.
    """
    if original_image is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Input (contour map)", fontsize=14)
        axes[0].axis("off")

        # Predicted heightmap
        im = axes[1].imshow(heightmap, cmap=cmap, vmin=0, vmax=1)
        axes[1].set_title("Predicted heightmap", fontsize=14)
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Relative height")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(heightmap, cmap=cmap, vmin=0, vmax=1)
        ax.set_title("Predicted heightmap", fontsize=14)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Relative height")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_heightmap_raw(heightmap: np.ndarray, output_path: str):
    """Save heightmap as raw numpy .npy file."""
    np.save(output_path, heightmap.astype(np.float32))


def save_heightmap_grayscale(heightmap: np.ndarray, output_path: str):
    """Save heightmap as 16-bit grayscale PNG for maximum precision."""
    # Scale [0, 1] → [0, 65535]
    scaled = (heightmap * 65535).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(output_path, scaled)


def find_images(input_path: str) -> list[str]:
    """
    Find all images from the input path.
    If input_path is a file, returns [input_path].
    If input_path is a directory, returns all image files in it.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [str(input_path)]
        else:
            raise ValueError(
                f"Not a supported image format: {input_path.suffix}\n"
                f"Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
            )

    if input_path.is_dir():
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(input_path.glob(f"*{ext}"))
            images.extend(input_path.glob(f"*{ext.upper()}"))
        images = sorted(set(str(p) for p in images))
        if not images:
            raise FileNotFoundError(f"No images found in directory: {input_path}")
        return images

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def determine_output_path(
    input_path: str,
    output_path: str,
    suffix: str = "_heightmap",
    ext: str = ".png",
) -> str:
    """
    Determine the output file path based on input and output arguments.

    If output_path is a directory, generates filename from input filename.
    If output_path is a file, uses it directly.
    """
    output = Path(output_path)
    input_name = Path(input_path).stem

    if output.is_dir() or (not output.suffix and not output.exists()):
        # Treat as directory
        output.mkdir(parents=True, exist_ok=True)
        return str(output / f"{input_name}{suffix}{ext}")

    return str(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run heightmap prediction on contour map images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image path or directory of images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path (file or directory)",
    )

    # Output formats
    parser.add_argument(
        "--save-vis",
        action="store_true",
        default=True,
        help="Save colorized visualization PNG (default: True)",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Don't save visualization",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw heightmap as .npy file",
    )
    parser.add_argument(
        "--save-grayscale",
        action="store_true",
        help="Save heightmap as 16-bit grayscale PNG",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        default=True,
        help="Include input image in visualization (default: True)",
    )
    parser.add_argument(
        "--no-side-by-side",
        action="store_true",
        help="Only show heightmap in visualization",
    )

    # Inference options
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override model input size (default: from checkpoint)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation (flip averaging, slower but potentially better)",
    )
    parser.add_argument(
        "--original-size",
        action="store_true",
        help="Resize output heightmap to match input image dimensions",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="terrain",
        help="Matplotlib colormap for visualization",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for visualization PNG",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/mps/cpu)",
    )

    args = parser.parse_args()

    if args.no_vis:
        args.save_vis = False
    if args.no_side_by_side:
        args.side_by_side = False

    # Must save at least one format
    if not args.save_vis and not args.save_raw and not args.save_grayscale:
        parser.error(
            "No output format selected. Use at least one of: "
            "--save-vis, --save-raw, --save-grayscale"
        )

    return args


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    model, img_size = load_model(args.checkpoint, device, args.img_size)

    # -----------------------------------------------------------------------
    # Find input images
    # -----------------------------------------------------------------------
    image_paths = find_images(args.input)
    print(f"\nFound {len(image_paths)} image(s) to process")

    # -----------------------------------------------------------------------
    # Run predictions
    # -----------------------------------------------------------------------
    total_time = 0.0

    for i, image_path in enumerate(image_paths):
        img_name = Path(image_path).name
        print(f"\n[{i + 1}/{len(image_paths)}] Processing: {img_name}")

        # Preprocess
        input_tensor, original_image, original_size = preprocess_image(
            image_path, img_size
        )

        # Predict
        t0 = time.time()
        heightmap = predict_single(
            model=model,
            input_tensor=input_tensor,
            device=device,
            original_size=original_size if args.original_size else None,
            use_tta=args.tta,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        print(f"  Inference time: {elapsed * 1000:.1f}ms")
        print(f"  Heightmap shape: {heightmap.shape}")
        print(f"  Height range: [{heightmap.min():.4f}, {heightmap.max():.4f}]")

        # Determine output paths
        base_output = determine_output_path(image_path, args.output, suffix="", ext="")
        # Strip extension for building format-specific paths
        base_no_ext = str(Path(base_output).with_suffix(""))

        # If output was specified as a file with extension, use it directly
        # for the primary format, otherwise build paths from stem
        output_has_ext = Path(args.output).suffix != "" and len(image_paths) == 1

        # Save visualization
        if args.save_vis:
            if output_has_ext:
                vis_path = args.output
            else:
                vis_path = f"{base_no_ext}_heightmap.png"

            original_for_vis = original_image if args.side_by_side else None

            # Resize original for display if heightmap was resized back
            if original_for_vis is not None and args.original_size:
                pass  # already at original size
            elif original_for_vis is not None:
                original_for_vis = cv2.resize(
                    original_for_vis,
                    (img_size, img_size),
                    interpolation=cv2.INTER_LINEAR,
                )

            save_heightmap_visualization(
                heightmap,
                vis_path,
                original_image=original_for_vis,
                cmap=args.cmap,
                dpi=args.dpi,
            )
            print(f"  Saved visualization: {vis_path}")

        # Save raw numpy
        if args.save_raw:
            raw_path = f"{base_no_ext}_heightmap.npy"
            save_heightmap_raw(heightmap, raw_path)
            print(f"  Saved raw heightmap: {raw_path}")

        # Save 16-bit grayscale
        if args.save_grayscale:
            gray_path = f"{base_no_ext}_heightmap_gray.png"
            save_heightmap_grayscale(heightmap, gray_path)
            print(f"  Saved grayscale:     {gray_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print(f"Done! Processed {len(image_paths)} image(s)")
    if len(image_paths) > 0:
        avg_time = total_time / len(image_paths)
        print(f"Average inference time: {avg_time * 1000:.1f}ms per image")
        if args.tta:
            print(f"  (with TTA enabled — disable for ~4x speedup)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

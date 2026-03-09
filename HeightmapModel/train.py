"""
Training script for contour map → heightmap U-Net.

Features:
    - Mixed precision training (AMP) for speed
    - Cosine annealing LR schedule with warmup
    - TensorBoard logging (losses, LR, sample predictions)
    - Checkpointing (best + periodic)
    - Gradient clipping
    - EMA (exponential moving average) of model weights
    - Resume from checkpoint
    - Early stopping

Usage:
    python train.py --data-dir /path/to/dataset --epochs 100
    python train.py --data-dir /path/to/dataset --resume runs/run_001/best.pt
"""

import argparse
import copy
import math
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dataset import build_dataloaders
from model import HeightmapLoss, HeightmapUNet
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------


class EMAModel:
    """
    Maintains an exponential moving average of model parameters.
    Use the EMA weights at inference time for better generalization.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module):
        """Replace model params with EMA params (for eval)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model params after eval."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict: dict):
        self.shadow = {name: tensor.clone() for name, tensor in state_dict.items()}


# ---------------------------------------------------------------------------
# LR Scheduler with Warmup
# ---------------------------------------------------------------------------


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
):
    """
    Cosine annealing with linear warmup.
    LR goes: 0 → max_lr (warmup) → min_lr (cosine decay)
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def visualize_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_samples: int = 4,
) -> plt.Figure:
    """
    Create a figure showing input / prediction / ground truth side by side.
    Returns matplotlib Figure for TensorBoard logging.
    """
    n = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    # Denormalize images (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    imgs = (images * std + mean).clamp(0, 1).cpu()

    preds = predictions.cpu().detach()
    tgts = targets.cpu().detach()

    for i in range(n):
        # Input image
        axes[i, 0].imshow(imgs[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Input (contour map)")
        axes[i, 0].axis("off")

        # Prediction
        im1 = axes[i, 1].imshow(preds[i, 0].numpy(), cmap="terrain", vmin=0, vmax=1)
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis("off")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Ground truth
        im2 = axes[i, 2].imshow(tgts[i, 0].numpy(), cmap="terrain", vmin=0, vmax=1)
        axes[i, 2].set_title("Ground truth")
        axes[i, 2].axis("off")
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: HeightmapLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler | None,
    ema: EMAModel,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> dict[str, float]:
    """Train for one epoch. Returns dict of average losses."""
    model.train()

    running_losses = {"loss": 0.0, "l1": 0.0, "gradient": 0.0, "structure": 0.0}
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["heightmap"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            predictions = model(images)
            losses = criterion(predictions, targets)
            loss = losses["loss"]

        scaler.scale(loss).backward()

        # Gradient clipping (unscale first for correct norm computation)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        ema.update(model)

        # Accumulate losses
        for key in running_losses:
            running_losses[key] += losses[key].item() if key != "loss" else loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    # Average losses
    avg_losses = {key: val / max(num_batches, 1) for key, val in running_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: HeightmapLoss,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
) -> tuple[
    dict[str, float], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]:
    """
    Validate the model. Returns average losses and sample tensors for visualization.
    """
    model.eval()

    running_losses = {"loss": 0.0, "l1": 0.0, "gradient": 0.0, "structure": 0.0}
    num_batches = 0
    sample_images = None
    sample_preds = None
    sample_targets = None

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["heightmap"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            predictions = model(images)
            losses = criterion(predictions, targets)

        for key in running_losses:
            running_losses[key] += (
                losses[key].item() if key != "loss" else losses["loss"].item()
            )
        num_batches += 1

        # Save first batch for visualization
        if sample_images is None:
            sample_images = images
            sample_preds = predictions
            sample_targets = targets

    avg_losses = {key: val / max(num_batches, 1) for key, val in running_losses.items()}
    return avg_losses, sample_images, sample_preds, sample_targets


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler | None,
    ema: EMAModel,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    scaler: GradScaler | None = None,
    ema: EMAModel | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load training checkpoint. Returns checkpoint dict."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if ema and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    return checkpoint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train contour map → heightmap U-Net",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to generated dataset directory",
    )
    parser.add_argument("--img-size", type=int, default=512, help="Input image size")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total samples (for debugging)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.01,
        help="Min LR as fraction of peak LR",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--warmup-epochs", type=float, default=3.0, help="Warmup epochs"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )

    # Loss weights
    parser.add_argument("--l1-weight", type=float, default=1.0, help="L1 loss weight")
    parser.add_argument(
        "--gradient-weight",
        type=float,
        default=0.5,
        help="Gradient (slope) loss weight",
    )
    parser.add_argument(
        "--structure-weight",
        type=float,
        default=0.2,
        help="Structural (SSIM) loss weight",
    )

    # Model
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained encoder (default: True)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained encoder",
    )
    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=2,
        help="Freeze encoder for this many initial epochs (0 to disable)",
    )

    # Checkpointing
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs",
        help="Root directory for training runs",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--vis-every",
        type=int,
        default=5,
        help="Log prediction visualizations every N epochs",
    )

    # System
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (0 to disable)",
    )

    args = parser.parse_args()

    if args.no_pretrained:
        args.pretrained = False

    return args


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Device setup
    # -----------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slow)")

    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    # -----------------------------------------------------------------------
    # Run directory + TensorBoard
    # -----------------------------------------------------------------------
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path(args.run_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir / "tb_logs"))
    print(f"Run directory: {run_dir}")
    print(f"TensorBoard:   tensorboard --logdir {run_dir / 'tb_logs'}")

    # Save args
    with open(run_dir / "args.txt", "w") as f:
        for key, val in sorted(vars(args).items()):
            f.write(f"{key}: {val}\n")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print("\n--- Loading data ---")
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        val_fraction=args.val_fraction,
        max_samples=args.max_samples,
        pin_memory=(device.type == "cuda"),
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps:     {total_steps}")
    print(f"Warmup steps:    {warmup_steps}")

    # -----------------------------------------------------------------------
    # Model, optimizer, scheduler, loss
    # -----------------------------------------------------------------------
    print("\n--- Building model ---")
    model = HeightmapUNet(pretrained=args.pretrained).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Separate encoder and decoder param groups (lower LR for pretrained encoder)
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder"):
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_params,
                "lr": args.lr * 0.1,
            },  # lower LR for pretrained encoder
            {"params": decoder_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    scaler = GradScaler(enabled=use_amp)

    criterion = HeightmapLoss(
        l1_weight=args.l1_weight,
        gradient_weight=args.gradient_weight,
        structure_weight=args.structure_weight,
    ).to(device)

    ema = EMAModel(model, decay=args.ema_decay)

    # -----------------------------------------------------------------------
    # Resume from checkpoint
    # -----------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, ema, device
        )
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.6f}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Starting training: epochs {start_epoch}..{args.epochs - 1}")
    print(f"{'=' * 60}\n")

    patience_counter = 0
    train_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # --- Encoder freezing for initial epochs ---
        if args.freeze_encoder_epochs > 0:
            if epoch < args.freeze_encoder_epochs:
                if epoch == start_epoch:  # only print once
                    print(
                        f"Freezing encoder for first {args.freeze_encoder_epochs} epochs"
                    )
                for param in model.encoder.parameters():
                    param.requires_grad = False
            elif epoch == args.freeze_encoder_epochs:
                print("Unfreezing encoder")
                for param in model.encoder.parameters():
                    param.requires_grad = True
                # Rebuild EMA with newly unfrozen params
                ema = EMAModel(model, decay=args.ema_decay)

        # --- Train ---
        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            device=device,
            epoch=epoch,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
        )

        # --- Validate (using EMA weights) ---
        ema.apply(model)
        val_losses, sample_imgs, sample_preds, sample_tgts = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
        )
        ema.restore(model)

        epoch_time = time.time() - epoch_start

        # --- Logging ---
        current_lr = optimizer.param_groups[-1]["lr"]  # decoder LR (the higher one)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss: {train_losses['loss']:.5f} | "
            f"val_loss: {val_losses['loss']:.5f} | "
            f"val_l1: {val_losses['l1']:.5f} | "
            f"lr: {current_lr:.2e} | "
            f"time: {epoch_time:.1f}s"
        )

        # TensorBoard scalars
        writer.add_scalars(
            "loss/total",
            {"train": train_losses["loss"], "val": val_losses["loss"]},
            epoch,
        )
        writer.add_scalars(
            "loss/l1",
            {"train": train_losses["l1"], "val": val_losses["l1"]},
            epoch,
        )
        writer.add_scalars(
            "loss/gradient",
            {"train": train_losses["gradient"], "val": val_losses["gradient"]},
            epoch,
        )
        writer.add_scalars(
            "loss/structure",
            {"train": train_losses["structure"], "val": val_losses["structure"]},
            epoch,
        )
        writer.add_scalar("lr/decoder", current_lr, epoch)
        writer.add_scalar("lr/encoder", optimizer.param_groups[0]["lr"], epoch)

        # TensorBoard visualizations
        if sample_imgs is not None and (
            epoch % args.vis_every == 0 or epoch == args.epochs - 1
        ):
            fig = visualize_predictions(
                sample_imgs, sample_preds, sample_tgts, num_samples=4
            )
            writer.add_figure("predictions", fig, epoch)
            plt.close(fig)

        # --- Checkpointing ---
        is_best = val_losses["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_losses["loss"]
            patience_counter = 0
            save_checkpoint(
                str(run_dir / "best.pt"),
                model,
                optimizer,
                scheduler,
                scaler,
                ema,
                epoch,
                best_val_loss,
                args,
            )
            print(f"  ↳ New best model saved (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1

        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                str(run_dir / f"epoch_{epoch:04d}.pt"),
                model,
                optimizer,
                scheduler,
                scaler,
                ema,
                epoch,
                best_val_loss,
                args,
            )

        # Always save latest for easy resume
        save_checkpoint(
            str(run_dir / "latest.pt"),
            model,
            optimizer,
            scheduler,
            scaler,
            ema,
            epoch,
            best_val_loss,
            args,
        )

        # --- Early stopping ---
        if args.patience > 0 and patience_counter >= args.patience:
            print(
                f"\nEarly stopping triggered after {args.patience} epochs without improvement."
            )
            break

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    total_time = time.time() - train_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Total time:    {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best model:    {run_dir / 'best.pt'}")
    print(f"TensorBoard:   tensorboard --logdir {run_dir / 'tb_logs'}")
    print(f"{'=' * 60}")

    # Export best model with EMA weights for inference
    print("\nExporting best model with EMA weights...")
    best_ckpt = load_checkpoint(str(run_dir / "best.pt"), model, device=device)
    ema_export = EMAModel(model, decay=args.ema_decay)
    if "ema_state_dict" in best_ckpt:
        ema_export.load_state_dict(best_ckpt["ema_state_dict"])
    ema_export.apply(model)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "img_size": args.img_size,
            "best_val_loss": best_val_loss,
        },
        str(run_dir / "best_ema.pt"),
    )
    print(f"Exported: {run_dir / 'best_ema.pt'}")

    writer.close()


if __name__ == "__main__":
    main()

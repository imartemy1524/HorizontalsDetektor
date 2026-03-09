"""
Python port of the C# DatasetGenerator + ReliefGenerator.

Generates training data for the heightmap model with identical output format:
    - {i}.jpg           : contour line image (RGB, white bg + brown contour lines)
    - {i}.data          : raw heightmap (float32, row-major, y-then-x)
    - {i}.metadata.json : sample metadata (Seed, CountOfHills, Elongation, Width, Height)

This is a faithful port of the Relief.compute shader logic as implemented in
ReliefGenerator.cs, so the generated terrain should look the same.

Usage:
    # Generate 10000 samples to ./dataset
    python generate_dataset.py --output ./dataset --count 10000

    # Quick test with 10 samples + preview
    python generate_dataset.py --output ./dataset --count 10 --preview

    # Custom parameters
    python generate_dataset.py --output ./dataset --count 5000 \
        --width 512 --height 512 \
        --min-hills 2 --max-hills 6 \
        --min-elongation 0.3 --max-elongation 3.0 \
        --horizontal 5 --workers 8
"""

import argparse
import json
import math
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# ReliefGenerator — port of ReliefGenerator.cs
# ---------------------------------------------------------------------------


class ReliefGenerator:
    """
    Generates relief heightmaps and contour line images.
    Faithful port of the C# ReliefGenerator which itself was ported from
    the Unity compute shader (Relief.compute).
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        map_height: int = 1000,
        seed: int | None = None,
    ):
        self.width = width
        self.height = height
        self.map_height = map_height
        self.rng = np.random.RandomState(seed)

        # Main heightmap buffer and scratch buffer (matching C# _result / _result2)
        self.result = np.zeros(width * height, dtype=np.float32)
        self.result2 = np.zeros(width * height, dtype=np.float32)

        # Gradient table for Perlin noise (256 unit vectors)
        self._init_gradients()

    def _init_gradients(self):
        """Initialize random gradient vectors for Perlin noise (matches C# InitializeGradients)."""
        self.gradients = np.zeros((256, 2), dtype=np.float32)
        for i in range(256):
            x = self.rng.uniform(-1, 1)
            y = self.rng.uniform(-1, 1)
            length = math.sqrt(x * x + y * y)
            if length > 0:
                self.gradients[i] = [x / length, y / length]
            else:
                self.gradients[i] = [1.0, 0.0]

    # --- Buffer access (matches C# macros / methods) ---

    def _get(self, x: int, y: int) -> float:
        return self.result[y * self.width + x]

    def _set(self, x: int, y: int, value: float):
        self.result[y * self.width + x] = value

    def _get2(self, x: int, y: int) -> float:
        return self.result2[y * self.width + x]

    def _set2(self, x: int, y: int, value: float):
        self.result2[y * self.width + x] = value

    def _height_val(self, x: float) -> float:
        """HEIGHT macro: x / map_height"""
        return x / self.map_height

    def _result_f(self, x: int, y: int, horizontal: float) -> int:
        """RESULTF: quantized height level index (for contour line detection)."""
        return int(self._get(x, y) * self.map_height / horizontal)

    # --- Perlin noise (from Relief.compute) ---

    def _drop_off(self, x: float) -> float:
        """Smooth interpolation curve (quintic, gives nice normals)."""
        v = 1.0 - abs(x)
        if v <= 0:
            return 0.0
        return 6 * v**5 - 15 * v**4 + 10 * v**3

    def _grad(self, ux: int, uy: int) -> np.ndarray:
        """Look up gradient from table (matches C# Grad)."""
        return self.gradients[(ux + uy * 16) % 256]

    def _noise_se(self, g: np.ndarray, vx: float, vy: float) -> float:
        """Single noise contribution from one grid corner."""
        return (g[0] * vx + g[1] * vy) * self._drop_off(vx) * self._drop_off(vy)

    def _noises(self, vx: float, vy: float, t: float) -> float:
        """Evaluate Perlin noise at a point (matches C# Noises)."""
        vx += t

        gix = int(math.floor(vx))
        giy = int(math.floor(vy))
        gizx = gix + 1
        giwy = giy + 1

        frac_x = vx - math.floor(vx)
        frac_y = vy - math.floor(vy)

        return (
            self._noise_se(self._grad(gix, giy), frac_x, frac_y)
            + self._noise_se(self._grad(gizx, giy), frac_x - 1, frac_y)
            + self._noise_se(self._grad(gix, giwy), frac_x, frac_y - 1)
            + self._noise_se(self._grad(gizx, giwy), frac_x - 1, frac_y - 1)
        )

    def _noise_step(
        self, x: int, y: int, i: int, res: float, elongation: float, t: float
    ) -> float:
        """Single octave of noise (matches C# NoiseStep)."""
        scale = (2.0**i) / res
        xy_x = x * elongation * scale
        xy_y = y * elongation * scale
        return (1.0 + self._noises(xy_x, xy_y, t)) * (2.0 ** (-(i + 2)))

    def generate_noise(
        self, count_of_hills: int, elongation: float = 1.0, t: float | None = None
    ):
        """
        Generate Perlin noise terrain (matches C# GenerateNoise / compute kernel 'noise').
        Fills self.result with heightmap values in [0, 1].
        """
        if t is None:
            t = self.rng.uniform(0, 1)
        res = float(max(self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                h = 0.0
                # Unrolled octave loop (matching compute shader)
                h += self._noise_step(x, y, 0, res, elongation, t)
                if count_of_hills > 1:
                    h += self._noise_step(x, y, 1, res, elongation, t)
                if count_of_hills > 2:
                    h += self._noise_step(x, y, 2, res, elongation, t)
                if count_of_hills > 3:
                    h += self._noise_step(x, y, 3, res, elongation, t)
                if count_of_hills > 4:
                    h += self._noise_step(x, y, 4, res, elongation, t)
                if count_of_hills > 5:
                    h += self._noise_step(x, y, 5, res, elongation, t)
                if count_of_hills > 6:
                    h += self._noise_step(x, y, 6, res, elongation, t)
                if count_of_hills > 7:
                    h += self._noise_step(x, y, 7, res, elongation, t)
                if count_of_hills > 8:
                    h += self._noise_step(x, y, 8, res, elongation, t)

                value = ((h - 0.5) / 1.5 + 0.5) % 1.0
                if value < 0:
                    value += 1.0
                self._set(x, y, value)

    def generate_noise_vectorized(
        self, count_of_hills: int, elongation: float = 1.0, t: float | None = None
    ):
        """
        Vectorized (numpy) version of generate_noise for speed.
        Produces identical results to the scalar version.
        """
        if t is None:
            t = self.rng.uniform(0, 1)
        res = float(max(self.width, self.height))

        # Build coordinate grids
        xs = np.arange(self.width, dtype=np.float32)
        ys = np.arange(self.height, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)  # shape (height, width)

        h = np.zeros((self.height, self.width), dtype=np.float64)

        for i in range(min(count_of_hills, 9)):
            scale = (2.0**i) / res
            vx_base = xx * elongation * scale + t
            vy_base = yy * elongation * scale

            gix = np.floor(vx_base).astype(np.int32)
            giy = np.floor(vy_base).astype(np.int32)

            frac_x = vx_base - np.floor(vx_base)
            frac_y = vy_base - np.floor(vy_base)

            def grad_field(ux, uy):
                idx = (ux + uy * 16) % 256
                return self.gradients[idx, 0], self.gradients[idx, 1]

            def drop_off_vec(x):
                v = np.clip(1.0 - np.abs(x), 0, None)
                return 6 * v**5 - 15 * v**4 + 10 * v**3

            def noise_se_vec(gx, gy, vvx, vvy):
                return (gx * vvx + gy * vvy) * drop_off_vec(vvx) * drop_off_vec(vvy)

            g00x, g00y = grad_field(gix, giy)
            g10x, g10y = grad_field(gix + 1, giy)
            g01x, g01y = grad_field(gix, giy + 1)
            g11x, g11y = grad_field(gix + 1, giy + 1)

            n = (
                noise_se_vec(g00x, g00y, frac_x, frac_y)
                + noise_se_vec(g10x, g10y, frac_x - 1, frac_y)
                + noise_se_vec(g01x, g01y, frac_x, frac_y - 1)
                + noise_se_vec(g11x, g11y, frac_x - 1, frac_y - 1)
            )

            h += (1.0 + n) * (2.0 ** (-(i + 2)))

        result_2d = ((h - 0.5) / 1.5 + 0.5) % 1.0
        result_2d = np.where(result_2d < 0, result_2d + 1.0, result_2d)

        # Flatten to 1D buffer (row-major, y outer)
        self.result = result_2d.astype(np.float32).ravel()

    # --- Smoothing (from Relief.compute - medium kernel) ---

    def apply_smoothing(self, iterations: int = 4):
        """
        Horizontal smoothing filter (matches C# ApplySmoothing / compute kernel 'medium').
        Weights: [1.2, 1.0, 3.0, 1.0, 1.2] / 7.4
        """
        w = self.width
        h = self.height
        s = 7.4

        for iteration in range(iterations):
            if iteration % 2 == 0:
                src = self.result
                dst = self.result2
            else:
                src = self.result2
                dst = self.result

            # Reshape for easier indexing
            src_2d = src.reshape(h, w)
            dst_2d = dst.reshape(h, w)

            # Boundary columns: copy directly
            dst_2d[:, 0] = src_2d[:, 0]
            dst_2d[:, 1] = src_2d[:, 1]
            if w > 2:
                dst_2d[:, w - 2] = src_2d[:, w - 2]
            if w > 1:
                dst_2d[:, w - 1] = src_2d[:, w - 1]

            # Interior: apply 1D horizontal kernel
            if w > 4:
                dst_2d[:, 2 : w - 2] = (
                    1.0 * src_2d[:, 1 : w - 3]
                    + 3.0 * src_2d[:, 2 : w - 2]
                    + 1.0 * src_2d[:, 3 : w - 1]
                    + 1.2 * src_2d[:, 4:w]
                    + 1.2 * src_2d[:, 0 : w - 4]
                ) / s

        # Ensure final result is in self.result
        if iterations % 2 != 0:
            np.copyto(self.result, self.result2)

    # --- Contour line detection (from Relief.compute - isHorizontal) ---

    def _is_horizontal(self, x: int, y: int, horizontal: float) -> float:
        """
        Determine contour line intensity at a point.
        Returns 0 (not on contour), 0.5 (thin line), or up to 1.0 (thick/major line).
        Matches C# IsHorizontal exactly.
        """
        coef = 0.5
        c = 0.0
        num = self._result_f(x, y, horizontal)

        if x > 0 and self._result_f(x - 1, y, horizontal) != num:
            c += coef
        if y > 0 and self._result_f(x, y - 1, horizontal) != num:
            c += coef
        if x < self.width - 1 and self._result_f(x + 1, y, horizontal) != num:
            c += coef
        if y < self.height - 1 and self._result_f(x, y + 1, horizontal) != num:
            c += coef

        if c > 0.5:
            return min(c, 1.0)

        if num % 5 == 0:
            if c == 0.5:
                return 1.0
            if x > 1 and self._result_f(x - 2, y, horizontal) > num:
                c += coef
            if y > 1 and self._result_f(x, y - 2, horizontal) > num:
                c += coef
            if x < self.width - 2 and self._result_f(x + 2, y, horizontal) > num:
                c += coef
            if y < self.height - 2 and self._result_f(x, y + 2, horizontal) > num:
                c += coef
        elif num % 5 == 1:
            if c == 0.5:
                return 1.0
            if x > 1 and self._result_f(x - 2, y, horizontal) < num:
                c += coef
            if y > 1 and self._result_f(x, y - 2, horizontal) < num:
                c += coef
            if x < self.width - 2 and self._result_f(x + 2, y, horizontal) < num:
                c += coef
            if y < self.height - 2 and self._result_f(x, y + 2, horizontal) < num:
                c += coef

        return min(c, 0.5)

    # --- Gradient / slope direction ---

    def _get_gradient(self, x: int, y: int) -> tuple[float, float]:
        """
        Compute gradient (slope direction) at a point. Returns normalized (gx, gy)
        pointing downhill. Matches C# GetGradient.
        """
        h_l = self._get(x - 1, y) if x > 0 else self._get(x, y)
        h_r = self._get(x + 1, y) if x < self.width - 1 else self._get(x, y)
        h_d = self._get(x, y - 1) if y > 0 else self._get(x, y)
        h_u = self._get(x, y + 1) if y < self.height - 1 else self._get(x, y)

        gx = h_l - h_r
        gy = h_d - h_u

        length = math.sqrt(gx * gx + gy * gy)
        if length > 0.0001:
            return gx / length, gy / length
        return 0.0, 0.0

    # --- Depression finding (for slope ticks) ---

    def _find_all_depressions(
        self, grid_step: int = 10, check_radius: int = 12
    ) -> list[tuple[int, int]]:
        """
        Find all local minima (depressions) in the heightmap, including edge depressions.
        Matches C# FindAllDepressions.
        """
        depressions = []
        w, h = self.width, self.height

        # Interior depressions
        for gy in range(check_radius, h - check_radius, grid_step):
            for gx in range(check_radius, w - check_radius, grid_step):
                min_x, min_y = gx, gy
                min_h = self._get(gx, gy)

                half = grid_step // 2
                for dy in range(-half, half + 1):
                    for dx in range(-half, half + 1):
                        px, py = gx + dx, gy + dy
                        if 0 <= px < w and 0 <= py < h:
                            val = self._get(px, py)
                            if val < min_h:
                                min_h = val
                                min_x, min_y = px, py

                is_minimum = True
                for dy in range(-check_radius, check_radius + 1, 3):
                    if not is_minimum:
                        break
                    for dx in range(-check_radius, check_radius + 1, 3):
                        if dx == 0 and dy == 0:
                            continue
                        px, py = min_x + dx, min_y + dy
                        if 0 <= px < w and 0 <= py < h:
                            if self._get(px, py) < min_h:
                                is_minimum = False
                                break

                if is_minimum:
                    depressions.append((min_x, min_y))

        # Edge depressions
        edge_depth = 15
        edge_step = grid_step

        # Top edge
        for x in range(edge_step, w - edge_step, edge_step):
            if self._get(x, 0) < self._get(x, edge_depth) - 0.01:
                depressions.append((x, 0))

        # Bottom edge
        for x in range(edge_step, w - edge_step, edge_step):
            if self._get(x, h - 1) < self._get(x, h - 1 - edge_depth) - 0.01:
                depressions.append((x, h - 1))

        # Left edge
        for y in range(edge_step, h - edge_step, edge_step):
            if self._get(0, y) < self._get(edge_depth, y) - 0.01:
                depressions.append((0, y))

        # Right edge
        for y in range(edge_step, h - edge_step, edge_step):
            if self._get(w - 1, y) < self._get(w - 1 - edge_depth, y) - 0.01:
                depressions.append((w - 1, y))

        return depressions

    # --- Slope tick drawing ---

    def _draw_slope_ticks(
        self,
        pixels: np.ndarray,
        horizontal: float,
        color: tuple[int, int, int],
        tick_length: int = 6,
        tick_spacing: int = 25,
    ):
        """
        Draw slope tick marks (бергштрихи) on depression contours, pointing inward.
        Draws directly into the pixels array (H, W, 4) RGBA.
        Matches C# DrawSlopeTicks.
        """
        depressions = self._find_all_depressions()

        for dep_x, dep_y in depressions:
            search_radius = 30
            ticks_drawn: set[tuple[int, int]] = set()

            for angle in range(0, 360, 45):
                rad = angle * math.pi / 180.0
                dir_x = math.cos(rad)
                dir_y = math.sin(rad)

                for dist in range(3, search_radius):
                    px = dep_x + int(dir_x * dist)
                    py = dep_y + int(dir_y * dist)

                    if px < 0 or px >= self.width or py < 0 or py >= self.height:
                        break

                    contour_strength = self._is_horizontal(px, py, horizontal)
                    if contour_strength >= 0.5:
                        cell_key = (px // tick_spacing, py // tick_spacing)

                        if cell_key not in ticks_drawn:
                            ticks_drawn.add(cell_key)

                            gx, gy = self._get_gradient(px, py)
                            for i in range(1, tick_length + 1):
                                tx = px + int(gx * i)
                                ty = py + int(gy * i)
                                if 0 <= tx < self.width and 0 <= ty < self.height:
                                    pixels[ty, tx] = [color[0], color[1], color[2], 255]
                        break

    # --- Public API ---

    def generate(
        self,
        count_of_hills: int = 4,
        horizontal: float = 5.0,
        elongation: float = 1.0,
        contour_color: tuple[int, int, int] = (139, 90, 43),
        draw_slope_ticks: bool = True,
        tick_spacing: int = 25,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate relief data: heightmap and contour line image.
        Matches C# ReliefGenerator.Generate.

        Args:
            count_of_hills: Number of noise octaves (1-9)
            horizontal: Height interval between contour lines
            elongation: Terrain stretch factor
            contour_color: RGB tuple for contour lines (default: brown)
            draw_slope_ticks: Whether to draw slope tick marks
            tick_spacing: Spacing between slope tick marks in pixels

        Returns:
            (heightmap, contour_image)
            - heightmap: (height, width) float32 array
            - contour_image: (height, width, 4) uint8 RGBA array
        """
        # Generate noise terrain (use vectorized version for speed)
        self.generate_noise_vectorized(count_of_hills, elongation)

        # Apply smoothing
        self.apply_smoothing(4)

        # Extract heightmap as 2D array
        heightmap = self.result.reshape(self.height, self.width).copy()

        # Create contour image (RGBA, transparent background)
        image = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                g = self._is_horizontal(x, y, horizontal)
                if g != 0:
                    alpha = int(g * 255)
                    image[y, x] = [
                        contour_color[0],
                        contour_color[1],
                        contour_color[2],
                        alpha,
                    ]

        # Draw slope tick marks
        if draw_slope_ticks:
            self._draw_slope_ticks(
                image,
                horizontal,
                contour_color,
                tick_length=6,
                tick_spacing=tick_spacing,
            )

        return heightmap, image


# ---------------------------------------------------------------------------
# Dataset generation (port of DatasetGenerator.cs)
# ---------------------------------------------------------------------------


@dataclass
class SampleMetadata:
    Seed: int
    CountOfHills: int
    Elongation: float
    Width: int
    Height: int


def generate_single_sample(args: tuple) -> str:
    """
    Generate a single dataset sample. Designed to be called from a process pool.

    Args:
        args: tuple of (index, output_dir, width, height, map_height, seed,
               count_of_hills, elongation, horizontal)

    Returns:
        Status string for progress reporting.
    """
    (
        index,
        output_dir,
        width,
        height,
        map_height,
        seed,
        count_of_hills,
        elongation,
        horizontal,
    ) = args

    generator = ReliefGenerator(width, height, map_height, seed)
    heightmap, contour_image = generator.generate(
        count_of_hills=count_of_hills,
        horizontal=horizontal,
        elongation=elongation,
        draw_slope_ticks=True,
    )

    base_path = os.path.join(output_dir, str(index))

    # Save contour image as JPG (composite onto white background, matching C#)
    # C# creates a white RGBA image then composites the contour lines on top
    rgb_image = np.full((height, width, 3), 255, dtype=np.uint8)
    alpha = contour_image[:, :, 3:4].astype(np.float32) / 255.0
    contour_rgb = contour_image[:, :, :3].astype(np.float32)
    bg = rgb_image.astype(np.float32)
    composited = (contour_rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)

    img = Image.fromarray(composited, "RGB")
    img.save(f"{base_path}.jpg", "JPEG", quality=90)

    # Save metadata as JSON (matching C# property names exactly)
    metadata = {
        "Seed": seed,
        "CountOfHills": count_of_hills,
        "Elongation": elongation,
        "Width": width,
        "Height": height,
    }
    with open(f"{base_path}.metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save heightmap as raw binary floats (matching C# SaveHeightmapBinary)
    # C# writes: for y in height: for x in width: write(heightmap[x, y])
    # Since the C# heightmap is [width, height] with [x, y] indexing,
    # and it iterates y-outer x-inner, the binary layout is row-major (y, x)
    # which is exactly how our numpy array (height, width) is laid out.
    with open(f"{base_path}.data", "wb") as f:
        f.write(heightmap.tobytes())

    return f"Sample {index} done"


def generate_dataset(
    output_dir: str,
    count: int,
    width: int = 512,
    height: int = 512,
    map_height: int = 1000,
    min_count_of_hills: int = 2,
    max_count_of_hills: int = 6,
    min_elongation: float = 0.3,
    max_elongation: float = 3.0,
    horizontal: float = 5.0,
    workers: int | None = None,
    master_seed: int | None = None,
):
    """
    Generate a full dataset of contour map + heightmap pairs.

    Args:
        output_dir: Directory to save samples to
        count: Number of samples to generate
        width: Image/heightmap width
        height: Image/heightmap height
        map_height: Height scale factor (used in contour quantization)
        min_count_of_hills: Minimum noise octaves
        max_count_of_hills: Maximum noise octaves
        min_elongation: Minimum terrain stretch
        max_elongation: Maximum terrain stretch
        horizontal: Contour line height interval
        workers: Number of parallel workers (default: CPU count)
        master_seed: Seed for reproducible dataset generation
    """
    os.makedirs(output_dir, exist_ok=True)

    if workers is None:
        workers = os.cpu_count() or 4

    rng = np.random.RandomState(master_seed)

    print("=== Relief Map Dataset Generator (Python) ===")
    print(f"Image size:    {width}×{height}")
    print(f"Dataset size:  {count} samples")
    print(f"Output dir:    {output_dir}")
    print(f"Workers:       {workers}")
    print(f"Hills range:   [{min_count_of_hills}, {max_count_of_hills}]")
    print(f"Elongation:    [{min_elongation:.2f}, {max_elongation:.2f}]")
    print(f"Horizontal:    {horizontal}")
    print()

    # Pre-generate all random parameters
    tasks = []
    for i in range(count):
        seed = int(rng.randint(0, 2**31))
        count_of_hills = int(rng.randint(min_count_of_hills, max_count_of_hills + 1))
        elongation = float(
            min_elongation + rng.uniform() * (max_elongation - min_elongation)
        )
        tasks.append(
            (
                i,
                output_dir,
                width,
                height,
                map_height,
                seed,
                count_of_hills,
                elongation,
                horizontal,
            )
        )

    start_time = time.time()
    completed = 0

    if workers <= 1:
        # Single-process mode (easier to debug)
        for task in tasks:
            generate_single_sample(task)
            completed += 1
            elapsed = time.time() - start_time
            if completed % max(1, count // 100) == 0 or completed == count:
                if completed > 0:
                    eta = elapsed * (count - completed) / completed
                    print(
                        f"\rProgress: {completed}/{count} "
                        f"({completed * 100 // count}%) - "
                        f"ETA: {int(eta // 60):02d}:{int(eta % 60):02d}    ",
                        end="",
                        flush=True,
                    )
    else:
        # Multi-process mode
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(generate_single_sample, t): t[0] for t in tasks}
            progress_step = max(1, count // 100)

            for future in as_completed(futures):
                future.result()  # raise any exceptions
                completed += 1
                if completed % progress_step == 0 or completed == count:
                    elapsed = time.time() - start_time
                    if completed > 0:
                        eta = elapsed * (count - completed) / completed
                        print(
                            f"\rProgress: {completed}/{count} "
                            f"({completed * 100 // count}%) - "
                            f"ETA: {int(eta // 60):02d}:{int(eta % 60):02d}    ",
                            end="",
                            flush=True,
                        )

    total_time = time.time() - start_time
    print()
    print(
        f"Dataset generation complete! "
        f"Total time: {int(total_time // 60):02d}:{int(total_time % 60):02d}"
    )
    print()
    print("Output files per sample:")
    print("  - {number}.jpg           : Contour line image")
    print("  - {number}.metadata.json : Parameters (Seed, CountOfHills, Elongation)")
    print("  - {number}.data          : Raw heightmap (binary floats, row-major)")
    print()
    print(f"Dataset saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Preview utility
# ---------------------------------------------------------------------------


def preview_samples(output_dir: str, num_samples: int = 4):
    """Generate a visual preview grid of samples from the dataset."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not installed, skipping preview. Install with: pip3 install matplotlib"
        )
        return

    # Find samples
    data_files = sorted(Path(output_dir).glob("*.data"))
    if not data_files:
        print("No samples found for preview.")
        return

    n = min(num_samples, len(data_files))

    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        stem = data_files[i].stem
        jpg_path = Path(output_dir) / f"{stem}.jpg"
        data_path = data_files[i]

        # Load and display contour image
        if jpg_path.exists():
            img = Image.open(jpg_path)
            axes[i, 0].imshow(np.array(img))
            axes[i, 0].set_title(f"Contour map (sample {stem})")
            axes[i, 0].axis("off")

        # Load and display heightmap
        meta_path = Path(output_dir) / f"{stem}.metadata.json"
        w, h = 512, 512
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                w = meta.get("Width", 512)
                h = meta.get("Height", 512)

        raw = np.fromfile(str(data_path), dtype=np.float32)
        heightmap = raw.reshape(h, w)

        im = axes[i, 1].imshow(heightmap, cmap="terrain")
        axes[i, 1].set_title(f"Heightmap (sample {stem})")
        axes[i, 1].axis("off")
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    preview_path = os.path.join(output_dir, "preview.png")
    plt.savefig(preview_path, dpi=100)
    plt.close(fig)
    print(f"Preview saved to: {preview_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate contour map + heightmap dataset for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./dataset",
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=10000,
        help="Number of samples to generate",
    )
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument(
        "--map-height",
        type=int,
        default=1000,
        help="Height scale factor for contour quantization",
    )
    parser.add_argument(
        "--min-hills", type=int, default=2, help="Min noise octaves (count of hills)"
    )
    parser.add_argument(
        "--max-hills", type=int, default=6, help="Max noise octaves (count of hills)"
    )
    parser.add_argument(
        "--min-elongation", type=float, default=0.3, help="Min terrain stretch"
    )
    parser.add_argument(
        "--max-elongation", type=float, default=3.0, help="Max terrain stretch"
    )
    parser.add_argument(
        "--horizontal", type=float, default=5.0, help="Contour line height interval"
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed for reproducible generation",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate a visual preview after dataset creation",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=4,
        help="Number of samples to show in preview",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    generate_dataset(
        output_dir=args.output,
        count=args.count,
        width=args.width,
        height=args.height,
        map_height=args.map_height,
        min_count_of_hills=args.min_hills,
        max_count_of_hills=args.max_hills,
        min_elongation=args.min_elongation,
        max_elongation=args.max_elongation,
        horizontal=args.horizontal,
        workers=args.workers,
        master_seed=args.seed,
    )

    if args.preview:
        print()
        preview_samples(args.output, num_samples=args.preview_count)


if __name__ == "__main__":
    main()

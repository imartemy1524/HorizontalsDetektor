//! Dataset generator for contour map → heightmap training data.
//!
//! Faithful port of the C# ReliefGenerator + DatasetGenerator.
//! Produces identical output format:
//!   - {i}.jpg           : contour line image (RGB, white bg + brown contour lines)
//!   - {i}.data          : raw heightmap (float32, row-major, y-then-x)
//!   - {i}.metadata.json : sample metadata
//!
//! Usage:
//!   cargo run --release -- --output ./dataset --count 10000
//!   cargo run --release -- --output ./dataset --count 10 --preview

use clap::Parser;
use image::{ImageBuffer, Rgb, Rgba, RgbaImage};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashSet;
use std::f32::consts::PI;
use std::fs;
use std::time::Instant;

// ---------------------------------------------------------------------------
// ReliefGenerator — port of ReliefGenerator.cs
// ---------------------------------------------------------------------------

struct ReliefGenerator {
    width: usize,
    height: usize,
    map_height: i32,
    result: Vec<f32>,
    result2: Vec<f32>,
    gradients: Vec<[f32; 2]>,
    rng: StdRng,
}

impl ReliefGenerator {
    fn new(width: usize, height: usize, map_height: i32, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let gradients = Self::init_gradients(&mut rng);

        Self {
            width,
            height,
            map_height,
            result: vec![0.0f32; width * height],
            result2: vec![0.0f32; width * height],
            gradients,
            rng,
        }
    }

    fn init_gradients(rng: &mut StdRng) -> Vec<[f32; 2]> {
        let mut gradients = vec![[0.0f32; 2]; 256];
        for g in gradients.iter_mut() {
            let x: f32 = rng.random::<f32>() * 2.0 - 1.0;
            let y: f32 = rng.random::<f32>() * 2.0 - 1.0;
            let length = (x * x + y * y).sqrt();
            if length > 0.0 {
                *g = [x / length, y / length];
            } else {
                *g = [1.0, 0.0];
            }
        }
        gradients
    }

    // --- Buffer access ---

    #[inline(always)]
    fn get(&self, x: usize, y: usize) -> f32 {
        self.result[y * self.width + x]
    }

    #[inline(always)]
    fn set(&mut self, x: usize, y: usize, value: f32) {
        self.result[y * self.width + x] = value;
    }

    #[inline(always)]
    fn result_f(&self, x: usize, y: usize, horizontal: f32) -> i32 {
        (self.get(x, y) * self.map_height as f32 / horizontal) as i32
    }

    // --- Perlin noise ---

    #[inline(always)]
    fn drop_off(x: f32) -> f32 {
        let v = (1.0 - x.abs()).max(0.0);
        6.0 * v.powi(5) - 15.0 * v.powi(4) + 10.0 * v.powi(3)
    }

    #[inline(always)]
    fn grad(&self, ux: i32, uy: i32) -> [f32; 2] {
        let idx = ((ux + uy * 16).rem_euclid(256)) as usize;
        self.gradients[idx]
    }

    #[inline(always)]
    fn noise_se(g: [f32; 2], vx: f32, vy: f32) -> f32 {
        (g[0] * vx + g[1] * vy) * Self::drop_off(vx) * Self::drop_off(vy)
    }

    fn noises(&self, vx: f32, vy: f32, t: f32) -> f32 {
        let vx = vx + t;

        let gix = vx.floor() as i32;
        let giy = vy.floor() as i32;
        let gizx = gix + 1;
        let giwy = giy + 1;

        let frac_x = vx - vx.floor();
        let frac_y = vy - vy.floor();

        Self::noise_se(self.grad(gix, giy), frac_x, frac_y)
            + Self::noise_se(self.grad(gizx, giy), frac_x - 1.0, frac_y)
            + Self::noise_se(self.grad(gix, giwy), frac_x, frac_y - 1.0)
            + Self::noise_se(self.grad(gizx, giwy), frac_x - 1.0, frac_y - 1.0)
    }

    #[inline(always)]
    fn noise_step(&self, x: usize, y: usize, i: i32, res: f32, elongation: f32, t: f32) -> f32 {
        let scale = 2.0f32.powi(i) / res;
        let xy_x = x as f32 * elongation * scale;
        let xy_y = y as f32 * elongation * scale;
        (1.0 + self.noises(xy_x, xy_y, t)) * 2.0f32.powi(-(i + 2))
    }

    fn generate_noise(&mut self, count_of_hills: i32, elongation: f32) {
        let t: f32 = self.rng.random();
        let res = self.width.max(self.height) as f32;

        for y in 0..self.height {
            for x in 0..self.width {
                let mut h = 0.0f32;

                h += self.noise_step(x, y, 0, res, elongation, t);
                if count_of_hills > 1 {
                    h += self.noise_step(x, y, 1, res, elongation, t);
                }
                if count_of_hills > 2 {
                    h += self.noise_step(x, y, 2, res, elongation, t);
                }
                if count_of_hills > 3 {
                    h += self.noise_step(x, y, 3, res, elongation, t);
                }
                if count_of_hills > 4 {
                    h += self.noise_step(x, y, 4, res, elongation, t);
                }
                if count_of_hills > 5 {
                    h += self.noise_step(x, y, 5, res, elongation, t);
                }
                if count_of_hills > 6 {
                    h += self.noise_step(x, y, 6, res, elongation, t);
                }
                if count_of_hills > 7 {
                    h += self.noise_step(x, y, 7, res, elongation, t);
                }
                if count_of_hills > 8 {
                    h += self.noise_step(x, y, 8, res, elongation, t);
                }

                let mut value = ((h - 0.5) / 1.5 + 0.5).rem_euclid(1.0);
                if value < 0.0 {
                    value += 1.0;
                }
                self.set(x, y, value);
            }
        }
    }

    // --- Smoothing ---

    fn apply_smoothing(&mut self, iterations: usize) {
        let w = self.width;
        let h = self.height;
        let s: f32 = 7.4;

        for iter in 0..iterations {
            let (src, dst) = if iter % 2 == 0 {
                // Clone src so we can write to dst without borrow issues
                let src = self.result.clone();
                (src, &mut self.result2)
            } else {
                let src = self.result2.clone();
                (src, &mut self.result)
            };

            for y in 0..h {
                for x in 0..w {
                    let idx = y * w + x;
                    if x <= 1 || x >= w - 2 {
                        dst[idx] = src[idx];
                    } else {
                        let ans = (1.0 * src[y * w + (x - 1)]
                            + 3.0 * src[y * w + x]
                            + 1.0 * src[y * w + (x + 1)]
                            + 1.2 * src[y * w + (x + 2)]
                            + 1.2 * src[y * w + (x - 2)])
                            / s;
                        dst[idx] = ans;
                    }
                }
            }
        }

        // Ensure final result is in self.result
        if iterations % 2 != 0 {
            self.result.copy_from_slice(&self.result2);
        }
    }

    // --- Contour line detection ---

    fn is_horizontal(&self, x: usize, y: usize, horizontal: f32) -> f32 {
        let coef: f32 = 0.5;
        let mut c: f32 = 0.0;
        let num = self.result_f(x, y, horizontal);
        let w = self.width;
        let h = self.height;

        if x > 0 && self.result_f(x - 1, y, horizontal) != num {
            c += coef;
        }
        if y > 0 && self.result_f(x, y - 1, horizontal) != num {
            c += coef;
        }
        if x < w - 1 && self.result_f(x + 1, y, horizontal) != num {
            c += coef;
        }
        if y < h - 1 && self.result_f(x, y + 1, horizontal) != num {
            c += coef;
        }

        if c > 0.5 {
            return c.min(1.0);
        }

        if num % 5 == 0 {
            if c == 0.5 {
                return 1.0;
            }
            if x > 1 && self.result_f(x - 2, y, horizontal) > num {
                c += coef;
            }
            if y > 1 && self.result_f(x, y - 2, horizontal) > num {
                c += coef;
            }
            if x < w - 2 && self.result_f(x + 2, y, horizontal) > num {
                c += coef;
            }
            if y < h - 2 && self.result_f(x, y + 2, horizontal) > num {
                c += coef;
            }
        } else if num % 5 == 1 {
            if c == 0.5 {
                return 1.0;
            }
            if x > 1 && self.result_f(x - 2, y, horizontal) < num {
                c += coef;
            }
            if y > 1 && self.result_f(x, y - 2, horizontal) < num {
                c += coef;
            }
            if x < w - 2 && self.result_f(x + 2, y, horizontal) < num {
                c += coef;
            }
            if y < h - 2 && self.result_f(x, y + 2, horizontal) < num {
                c += coef;
            }
        }

        c.min(0.5)
    }

    // --- Gradient ---

    fn get_gradient(&self, x: usize, y: usize) -> (f32, f32) {
        let w = self.width;
        let h = self.height;

        let h_l = if x > 0 { self.get(x - 1, y) } else { self.get(x, y) };
        let h_r = if x < w - 1 { self.get(x + 1, y) } else { self.get(x, y) };
        let h_d = if y > 0 { self.get(x, y - 1) } else { self.get(x, y) };
        let h_u = if y < h - 1 { self.get(x, y + 1) } else { self.get(x, y) };

        let gx = h_l - h_r;
        let gy = h_d - h_u;

        let length = (gx * gx + gy * gy).sqrt();
        if length > 0.0001 {
            (gx / length, gy / length)
        } else {
            (0.0, 0.0)
        }
    }

    // --- Depression finding ---

    fn find_all_depressions(&self, grid_step: usize, check_radius: usize) -> Vec<(usize, usize)> {
        let w = self.width;
        let h = self.height;
        let mut depressions = Vec::new();

        // Interior depressions
        let mut gy = check_radius;
        while gy < h.saturating_sub(check_radius) {
            let mut gx = check_radius;
            while gx < w.saturating_sub(check_radius) {
                let mut min_x = gx;
                let mut min_y = gy;
                let mut min_h = self.get(gx, gy);

                let half = grid_step / 2;
                for dy_off in 0..=(half * 2) {
                    let dy = dy_off as isize - half as isize;
                    for dx_off in 0..=(half * 2) {
                        let dx = dx_off as isize - half as isize;
                        let px = (gx as isize + dx) as usize;
                        let py = (gy as isize + dy) as usize;
                        if px < w && py < h {
                            let val = self.get(px, py);
                            if val < min_h {
                                min_h = val;
                                min_x = px;
                                min_y = py;
                            }
                        }
                    }
                }

                let mut is_minimum = true;
                'outer: for dy_off in (0..=(check_radius * 2)).step_by(3) {
                    let dy = dy_off as isize - check_radius as isize;
                    for dx_off in (0..=(check_radius * 2)).step_by(3) {
                        let dx = dx_off as isize - check_radius as isize;
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let px = min_x as isize + dx;
                        let py = min_y as isize + dy;
                        if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                            if self.get(px as usize, py as usize) < min_h {
                                is_minimum = false;
                                break 'outer;
                            }
                        }
                    }
                }

                if is_minimum {
                    depressions.push((min_x, min_y));
                }
                gx += grid_step;
            }
            gy += grid_step;
        }

        // Edge depressions
        let edge_depth: usize = 15;
        let edge_step = grid_step;

        // Top edge
        let mut x = edge_step;
        while x < w.saturating_sub(edge_step) {
            if self.get(x, 0) < self.get(x, edge_depth) - 0.01 {
                depressions.push((x, 0));
            }
            x += edge_step;
        }

        // Bottom edge
        let mut x = edge_step;
        while x < w.saturating_sub(edge_step) {
            if self.get(x, h - 1) < self.get(x, h - 1 - edge_depth) - 0.01 {
                depressions.push((x, h - 1));
            }
            x += edge_step;
        }

        // Left edge
        let mut y = edge_step;
        while y < h.saturating_sub(edge_step) {
            if self.get(0, y) < self.get(edge_depth, y) - 0.01 {
                depressions.push((0, y));
            }
            y += edge_step;
        }

        // Right edge
        let mut y = edge_step;
        while y < h.saturating_sub(edge_step) {
            if self.get(w - 1, y) < self.get(w - 1 - edge_depth, y) - 0.01 {
                depressions.push((w - 1, y));
            }
            y += edge_step;
        }

        depressions
    }

    // --- Slope tick drawing ---

    fn draw_slope_ticks(
        &self,
        image: &mut RgbaImage,
        horizontal: f32,
        color: [u8; 3],
        tick_length: i32,
        tick_spacing: i32,
    ) {
        let depressions = self.find_all_depressions(10, 12);
        let w = self.width;
        let h = self.height;

        for (dep_x, dep_y) in depressions {
            let search_radius = 30;
            let mut ticks_drawn: HashSet<(i32, i32)> = HashSet::new();

            for angle_step in 0..8 {
                let angle = angle_step as f32 * 45.0;
                let rad = angle * PI / 180.0;
                let dir_x = rad.cos();
                let dir_y = rad.sin();

                for dist in 3..search_radius {
                    let px = dep_x as i32 + (dir_x * dist as f32) as i32;
                    let py = dep_y as i32 + (dir_y * dist as f32) as i32;

                    if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 {
                        break;
                    }

                    let pxu = px as usize;
                    let pyu = py as usize;

                    let contour_strength = self.is_horizontal(pxu, pyu, horizontal);
                    if contour_strength >= 0.5 {
                        let cell_key = (px / tick_spacing, py / tick_spacing);

                        if !ticks_drawn.contains(&cell_key) {
                            ticks_drawn.insert(cell_key);

                            let (gx, gy) = self.get_gradient(pxu, pyu);
                            for i in 1..=tick_length {
                                let tx = px + (gx * i as f32) as i32;
                                let ty = py + (gy * i as f32) as i32;
                                if tx >= 0 && tx < w as i32 && ty >= 0 && ty < h as i32 {
                                    image.put_pixel(
                                        tx as u32,
                                        ty as u32,
                                        Rgba([color[0], color[1], color[2], 255]),
                                    );
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    // --- Public API ---

    fn generate(
        &mut self,
        count_of_hills: i32,
        horizontal: f32,
        elongation: f32,
        contour_color: [u8; 3],
        draw_slope_ticks: bool,
        tick_spacing: i32,
    ) -> (Vec<f32>, RgbaImage) {
        // Generate noise terrain
        self.generate_noise(count_of_hills, elongation);

        // Apply smoothing
        self.apply_smoothing(4);

        // Copy heightmap
        let heightmap = self.result.clone();

        // Create contour image (RGBA, transparent bg)
        let w = self.width as u32;
        let h = self.height as u32;
        let mut image = RgbaImage::new(w, h);

        for y in 0..self.height {
            for x in 0..self.width {
                let g = self.is_horizontal(x, y, horizontal);
                if g != 0.0 {
                    let alpha = (g * 255.0) as u8;
                    image.put_pixel(
                        x as u32,
                        y as u32,
                        Rgba([contour_color[0], contour_color[1], contour_color[2], alpha]),
                    );
                }
            }
        }

        // Draw slope tick marks
        if draw_slope_ticks {
            self.draw_slope_ticks(&mut image, horizontal, contour_color, 6, tick_spacing);
        }

        (heightmap, image)
    }
}

// ---------------------------------------------------------------------------
// Dataset generation
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[allow(non_snake_case)]
struct SampleMetadata {
    Seed: u64,
    CountOfHills: i32,
    Elongation: f32,
    Width: usize,
    Height: usize,
}

fn composite_onto_white(contour: &RgbaImage, width: u32, height: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut rgb = ImageBuffer::from_pixel(width, height, Rgb([255u8, 255, 255]));

    for y in 0..height {
        for x in 0..width {
            let pixel = contour.get_pixel(x, y);
            let alpha = pixel[3] as f32 / 255.0;
            if alpha > 0.0 {
                let bg = rgb.get_pixel(x, y);
                let r = (pixel[0] as f32 * alpha + bg[0] as f32 * (1.0 - alpha)) as u8;
                let g = (pixel[1] as f32 * alpha + bg[1] as f32 * (1.0 - alpha)) as u8;
                let b = (pixel[2] as f32 * alpha + bg[2] as f32 * (1.0 - alpha)) as u8;
                rgb.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
    }

    rgb
}

fn generate_single_sample(
    index: usize,
    output_dir: &str,
    width: usize,
    height: usize,
    map_height: i32,
    seed: u64,
    count_of_hills: i32,
    elongation: f32,
    horizontal: f32,
) {
    let mut generator = ReliefGenerator::new(width, height, map_height, seed);
    let (heightmap, contour_image) = generator.generate(
        count_of_hills,
        horizontal,
        elongation,
        [139, 90, 43], // brown
        true,
        25,
    );

    let base_path = format!("{}/{}", output_dir, index);

    // Save contour image as JPG (composite onto white background)
    let rgb_image = composite_onto_white(&contour_image, width as u32, height as u32);
    rgb_image
        .save(format!("{}.jpg", base_path))
        .expect("Failed to save JPG");

    // Save metadata as JSON
    let metadata = SampleMetadata {
        Seed: seed,
        CountOfHills: count_of_hills,
        Elongation: elongation,
        Width: width,
        Height: height,
    };
    let json = serde_json::to_string_pretty(&metadata).unwrap();
    fs::write(format!("{}.metadata.json", base_path), json).expect("Failed to save metadata");

    // Save heightmap as raw binary floats (row-major, y outer, x inner)
    // This matches the C# SaveHeightmapBinary output exactly.
    let bytes: Vec<u8> = heightmap.iter().flat_map(|f| f.to_le_bytes()).collect();
    fs::write(format!("{}.data", base_path), bytes).expect("Failed to save heightmap");
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "dataset_generator",
    about = "Generate contour map + heightmap dataset for ML training"
)]
struct Args {
    /// Output directory for generated samples
    #[arg(short, long, default_value = "./dataset")]
    output: String,

    /// Number of samples to generate
    #[arg(short = 'n', long, default_value_t = 10000)]
    count: usize,

    /// Image width
    #[arg(long, default_value_t = 512)]
    width: usize,

    /// Image height
    #[arg(long, default_value_t = 512)]
    height: usize,

    /// Height scale factor for contour quantization
    #[arg(long, default_value_t = 1000)]
    map_height: i32,

    /// Min noise octaves (count of hills)
    #[arg(long, default_value_t = 2)]
    min_hills: i32,

    /// Max noise octaves (count of hills)
    #[arg(long, default_value_t = 6)]
    max_hills: i32,

    /// Min terrain stretch
    #[arg(long, default_value_t = 0.3)]
    min_elongation: f32,

    /// Max terrain stretch
    #[arg(long, default_value_t = 3.0)]
    max_elongation: f32,

    /// Contour line height interval
    #[arg(long, default_value_t = 5.0)]
    horizontal: f32,

    /// Master seed for reproducible generation
    #[arg(long)]
    seed: Option<u64>,

    /// Number of threads (0 = auto)
    #[arg(short = 'j', long, default_value_t = 0)]
    threads: usize,
}

fn main() {
    let args = Args::parse();

    // Set thread count
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap();
    }

    let thread_count = rayon::current_num_threads();

    // Create output directory
    fs::create_dir_all(&args.output).expect("Failed to create output directory");

    println!("=== Relief Map Dataset Generator (Rust) ===");
    println!("Image size:    {}×{}", args.width, args.height);
    println!("Dataset size:  {} samples", args.count);
    println!("Output dir:    {}", args.output);
    println!("Threads:       {}", thread_count);
    println!(
        "Hills range:   [{}, {}]",
        args.min_hills, args.max_hills
    );
    println!(
        "Elongation:    [{:.2}, {:.2}]",
        args.min_elongation, args.max_elongation
    );
    println!("Horizontal:    {}", args.horizontal);
    println!();

    // Pre-generate random parameters for all samples
    let master_seed = args.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });
    let mut master_rng = StdRng::seed_from_u64(master_seed);

    struct SampleParams {
        index: usize,
        seed: u64,
        count_of_hills: i32,
        elongation: f32,
    }

    let tasks: Vec<SampleParams> = (0..args.count)
        .map(|i| {
            let seed: u64 = master_rng.random::<u32>() as u64;
            let count_of_hills =
                master_rng.random_range(args.min_hills..=args.max_hills);
            let elongation = args.min_elongation
                + master_rng.random::<f32>() * (args.max_elongation - args.min_elongation);
            SampleParams {
                index: i,
                seed,
                count_of_hills,
                elongation,
            }
        })
        .collect();

    let start = Instant::now();

    // Progress bar
    let pb = ProgressBar::new(args.count as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} [{elapsed_precise}] [{bar:50.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );
    pb.set_message("Generating");

    // Parallel generation
    tasks.par_iter().progress_with(pb.clone()).for_each(|task| {
        generate_single_sample(
            task.index,
            &args.output,
            args.width,
            args.height,
            args.map_height,
            task.seed,
            task.count_of_hills,
            task.elongation,
            args.horizontal,
        );
    });

    pb.finish_with_message("Done");

    let elapsed = start.elapsed();
    let secs = elapsed.as_secs();
    let mins = secs / 60;
    let secs = secs % 60;

    println!();
    println!(
        "Dataset generation complete! Total time: {:02}:{:02}",
        mins, secs
    );
    println!();
    println!("Output files per sample:");
    println!("  - {{number}}.jpg           : Contour line image");
    println!("  - {{number}}.metadata.json : Parameters (Seed, CountOfHills, Elongation)");
    println!("  - {{number}}.data          : Raw heightmap (binary floats, row-major)");
    println!();
    println!("Dataset saved to: {}", args.output);
    println!();
    println!(
        "Samples/sec: {:.1}",
        args.count as f64 / elapsed.as_secs_f64()
    );
}

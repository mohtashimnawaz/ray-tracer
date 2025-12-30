mod vec3;
mod ray;
mod hittable;
mod sphere;
mod material;
mod camera;

use vec3::{Vec3, Color, Point3};
use ray::Ray;
use sphere::Sphere;
use hittable::{Hittable, HittableList};
use camera::Camera;
use std::sync::Arc;
use image::{RgbImage, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use material::{Lambertian, Metal, Dielectric};
use clap::Parser;
use image::imageops::FilterType;

/// Simple CLI for the ray tracer
#[derive(Parser, Debug)]
#[command(author, version, about = "A tiny ray tracer", long_about = None)]
struct Cli {
    /// Image width in pixels
    #[arg(short, long, default_value_t = 400)]
    width: u32,

    /// Image height in pixels. If omitted, height is computed from aspect ratio
    #[arg(short, long)]
    height: Option<u32>,

    /// Samples per pixel
    #[arg(short = 's', long, default_value_t = 50)]
    samples: u32,

    /// Max recursion depth
    #[arg(short = 'd', long, default_value_t = 10)]
    max_depth: u32,

    /// Output filename
    #[arg(short, long, default_value = "render.png")]
    output: String,

    /// Number of threads to use (optional)
    #[arg(long)]
    threads: Option<usize>,

    /// Optional input image to overlay or use as background (PNG/JPG/etc)
    #[arg(long)]
    input: Option<String>,

    /// Blend the input image with the render (50/50) instead of replacing pixels
    #[arg(long, default_value_t = false)]
    blend: bool,

    /// Manually set pixel(s) as `x,y,r,g,b`. Can be provided multiple times.
    /// Example: --set-pixel 10,20,255,0,0
    #[arg(long = "set-pixel")]
    set_pixel: Vec<String>,
}

fn ray_color(r: &Ray, world: &HittableList, depth: u32) -> Color {
    if depth == 0 {
        return Color::zero();
    }

    if let Some(rec) = world.hit(r, 0.001, f64::INFINITY) {
        if let Some((atten, scattered)) = rec.mat.scatter(r, &rec) {
            return atten * ray_color(&scattered, world, depth - 1);
        }
        return Color::zero();
    }

    let unit_direction = r.direction.unit_vector();
    let t = 0.5 * (unit_direction.y + 1.0);
    Color::new(1.0, 1.0, 1.0) * (1.0 - t) + Color::new(0.5, 0.7, 1.0) * t
}

fn main() {
    // Parse CLI
    let cli = Cli::parse();

    // Image
    let aspect_ratio = 16.0 / 9.0;
    let image_width: u32 = cli.width;
    let image_height: u32 = match cli.height {
        Some(h) => h,
        None => (image_width as f64 / aspect_ratio) as u32,
    };
    let samples_per_pixel = cli.samples;
    let max_depth = cli.max_depth;
    let output_file = cli.output;

    // Optional thread control
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .expect("Failed to build rayon thread pool");
    }

    println!("Rendering {w}x{h}, {s} spp, max depth {d} -> {out}", w = image_width, h = image_height, s = samples_per_pixel, d = max_depth, out = output_file);

    // World
    let mut world = HittableList::new();

    let mat_ground = Arc::new(Lambertian::new(Color::new(0.8, 0.8, 0.0)));
    let mat_center = Arc::new(Lambertian::new(Color::new(0.1, 0.2, 0.5)));
    let mat_left = Arc::new(Dielectric::new(1.5));
    let mat_right = Arc::new(Metal::new(Color::new(0.8, 0.6, 0.2), 0.0));

    world.add(Arc::new(Sphere::new(Point3::new(0.0, -100.5, -1.0), 100.0, mat_ground)));
    world.add(Arc::new(Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5, mat_center)));
    world.add(Arc::new(Sphere::new(Point3::new(-1.0, 0.0, -1.0), 0.5, mat_left.clone())));
    world.add(Arc::new(Sphere::new(Point3::new(-1.0, 0.0, -1.0), -0.45, mat_left)));
    world.add(Arc::new(Sphere::new(Point3::new(1.0, 0.0, -1.0), 0.5, mat_right)));

    // Camera
    let lookfrom = Point3::new(3.0, 3.0, 2.0);
    let lookat = Point3::new(0.0, 0.0, -1.0);
    let vup = Vec3::new(0.0, 1.0, 0.0);
    let dist_to_focus = (lookfrom - lookat).length();
    let aperture = 2.0;
    let cam = Camera::new(lookfrom, lookat, vup, 20.0, aspect_ratio, aperture, dist_to_focus);

    // Progress bar
    let bar = ProgressBar::new(image_height as u64);
    bar.set_style(ProgressStyle::default_bar().template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} rows").expect("progress template"));

    // Render (rows in parallel)
    let rows: Vec<Vec<[u8; 3]>> = (0..image_height).into_par_iter().map(|j| {
        // Build row pixels for scanline `j` (bottom->top ordering)
        let mut row_pixels: Vec<[u8; 3]> = Vec::with_capacity(image_width as usize);
        for i in 0..image_width {
            let mut pixel_color = Color::zero();
            for _s in 0..samples_per_pixel {
                let u = (i as f64 + rand::random::<f64>()) / (image_width as f64 - 1.0);
                let v = (j as f64 + rand::random::<f64>()) / (image_height as f64 - 1.0);
                let r = cam.get_ray(u, v);
                pixel_color += ray_color(&r, &world, max_depth);
            }
            row_pixels.push(pixel_color.to_rgb8(samples_per_pixel));
        }
        bar.inc(1);
        row_pixels
    }).collect();

    // Assemble image
    let mut imgbuf: RgbImage = RgbImage::new(image_width, image_height);

    for (row_idx, row) in rows.into_iter().enumerate() {
        let y = image_height - 1 - row_idx as u32; // map back to image coords
        for (x, px) in row.into_iter().enumerate() {
            imgbuf.put_pixel(x as u32, y, Rgb(px));
        }
    }

    bar.finish_with_message("done");

    // If an input image was provided, overlay or blend it into the final image
    if let Some(input_path) = &cli.input {
        match image::open(input_path) {
            Ok(img) => {
                let img = img.to_rgb8();
                let resized = image::imageops::resize(&img, image_width, image_height, FilterType::Lanczos3);
                if cli.blend {
                    for y in 0..image_height {
                        for x in 0..image_width {
                            let base = imgbuf.get_pixel(x, y).0;
                            let src = resized.get_pixel(x, y).0;
                            let blended = [
                                ((base[0] as u16 + src[0] as u16) / 2) as u8,
                                ((base[1] as u16 + src[1] as u16) / 2) as u8,
                                ((base[2] as u16 + src[2] as u16) / 2) as u8,
                            ];
                            imgbuf.put_pixel(x, y, Rgb(blended));
                        }
                    }
                } else {
                    // Replace pixels with input image
                    for y in 0..image_height {
                        for x in 0..image_width {
                            let src = resized.get_pixel(x, y);
                            imgbuf.put_pixel(x, y, *src);
                        }
                    }
                }
            }
            Err(e) => eprintln!("Failed to open input image {}: {}", input_path, e),
        }
    }

    // Apply any --set-pixel edits (format: x,y,r,g,b) - origin at top-left (0,0)
    for spec in &cli.set_pixel {
        let parts: Vec<&str> = spec.split(',').map(|s| s.trim()).collect();
        if parts.len() != 5 {
            eprintln!("Ignored --set-pixel '{}': expected 5 comma-separated values", spec);
            continue;
        }
        let parse_u32 = |s: &str| s.parse::<u32>().ok();
        let parse_u8 = |s: &str| s.parse::<u8>().ok();
        if let (Some(x), Some(y), Some(r), Some(g), Some(b)) = (parse_u32(parts[0]), parse_u32(parts[1]), parse_u8(parts[2]), parse_u8(parts[3]), parse_u8(parts[4])) {
            if x < image_width && y < image_height {
                imgbuf.put_pixel(x, y, Rgb([r, g, b]));
            } else {
                eprintln!("Ignored --set-pixel '{}': coordinates out of bounds", spec);
            }
        } else {
            eprintln!("Ignored --set-pixel '{}': could not parse numbers", spec);
        }
    }

    // Save
    imgbuf.save(&output_file).expect("Failed to save image");
    println!("Wrote {out} ({width}x{height})", out = output_file, width = image_width, height = image_height);
}

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
    // Image
    let aspect_ratio = 16.0 / 9.0;
    let image_width: u32 = 400;
    let image_height: u32 = (image_width as f64 / aspect_ratio) as u32;
    let samples_per_pixel = 5000;
    let max_depth = 10;

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

    // Save
    imgbuf.save("render.png").expect("Failed to save image");
    println!("Wrote render.png ({width}x{height})", width = image_width, height = image_height);
}

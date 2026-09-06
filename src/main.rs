use clap::Parser;
use lanczos_ultra::Resizer;
use std::{path::PathBuf, time::Instant};

#[derive(Parser, Debug)]
#[command(author, version, about = "Antialiased Lanczos RGBA resizer")]
struct Args {
    input: PathBuf,
    output: PathBuf,
    #[arg(short = 'w', long, value_parser = clap::value_parser!(u32).range(1..), conflicts_with = "scale")]
    width: Option<u32>,
    #[arg(short = 'H', long, value_parser = clap::value_parser!(u32).range(1..), conflicts_with = "scale")]
    height: Option<u32>,
    #[arg(short = 's', long, value_parser = clap::value_parser!(u32).range(1..))]
    scale: Option<u32>,
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=8), conflicts_with = "radius_percent")]
    lobes: Option<u32>,
    /// Radius as a percentage of the smaller source dimension (1..100).
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=100))]
    radius_percent: Option<u32>,
}
fn dimension(n: u64) -> Result<u32, Box<dyn std::error::Error>> {
    Ok(u32::try_from(n.max(1)).map_err(|_| "output dimension exceeds u32")?)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.width.is_none() && args.height.is_none() && args.scale.is_none() {
        return Err("specify --width, --height, or --scale (percent)".into());
    }
    let rgba = image::open(&args.input)?.to_rgba8();
    let (sw, sh) = rgba.dimensions();
    let (dw, dh) = match (args.width, args.height, args.scale) {
        (Some(w), Some(h), _) => (w, h),
        (Some(w), None, _) => (
            w,
            dimension((sh as u64 * w as u64 + sw as u64 / 2) / sw as u64)?,
        ),
        (None, Some(h), _) => (
            dimension((sw as u64 * h as u64 + sh as u64 / 2) / sh as u64)?,
            h,
        ),
        (_, _, Some(s)) => (
            dimension((sw as u64 * s as u64 + 50) / 100)?,
            dimension((sh as u64 * s as u64 + 50) / 100)?,
        ),
        _ => unreachable!(),
    };
    let start = Instant::now();
    let (plan, filter) = if let Some(percent) = args.radius_percent {
        (
            Resizer::with_radius_percent(sw, sh, dw, dh, percent)?,
            format!(
                "Lanczos radius {}% ({:.3} source pixels)",
                percent,
                (sw.min(sh) as f64 * percent as f64 / 100.0).max(1.0)
            ),
        )
    } else {
        let lobes = args.lobes.unwrap_or(5);
        (
            Resizer::with_lobes(sw, sh, dw, dh, lobes)?,
            format!("Lanczos{lobes}"),
        )
    };
    let out = plan.resize(rgba.as_raw())?;
    let elapsed = start.elapsed();
    image::save_buffer(&args.output, &out, dw, dh, image::ColorType::Rgba8)?;
    eprintln!(
        "{}x{} -> {}x{}, {}, {:.3} ms, {} equivalent spatial sample contributions",
        sw,
        sh,
        dw,
        dh,
        filter,
        elapsed.as_secs_f64() * 1000.0,
        plan.sample_count()
    );
    Ok(())
}

//! Research helper: raw RGBA8 input/output, configurable percentage kernel.
use lanczos_ultra::{RadiusOptions, Resizer};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 11 {
        return Err("input output sw sh dw dh radius filter_scale margin floor".into());
    }
    let src = std::fs::read(&a[1])?;
    let plan = Resizer::with_radius_options(
        a[3].parse()?,
        a[4].parse()?,
        a[5].parse()?,
        a[6].parse()?,
        a[7].parse()?,
        RadiusOptions {
            filter_scale: a[8].parse()?,
            ringing_margin: a[9].parse()?,
            ringing_floor: a[10].parse()?,
        },
    )?;
    std::fs::write(&a[2], plan.resize(&src)?)?;
    Ok(())
}

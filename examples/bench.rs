use lanczos_ultra::Resizer;
use std::{hint::black_box, time::Instant};
fn main() {
    for (sw, sh, dw, dh) in [
        (1920, 1080, 960, 540),
        (3840, 2160, 1920, 1080),
        (1920, 1080, 2880, 1620),
        (3840, 2160, 320, 180),
    ] {
        let src: Vec<u8> = (0..sw * sh)
            .flat_map(|i| [(i % 251) as u8, (i % 239) as u8, (i % 233) as u8, 255])
            .collect();
        let start = Instant::now();
        let plan = Resizer::new(sw, sh, dw, dh).unwrap();
        let setup = start.elapsed().as_secs_f64() * 1000.0;
        let mut out = vec![0; dw as usize * dh as usize * 4];
        let mut times = Vec::new();
        for i in 0..8 {
            let start = Instant::now();
            plan.resize_into(black_box(&src), black_box(&mut out))
                .unwrap();
            if i > 0 {
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{sw}x{sh} -> {dw}x{dh}: plan={setup:.3} ms, resize={:.3} ms, samples={}, intermediate={:.2} MiB",
            times[3],
            plan.sample_count(),
            dw as f64 * sh as f64 * 16.0 / 1048576.0
        );
    }
}

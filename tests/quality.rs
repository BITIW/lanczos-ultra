use lanczos_ultra::{ResizeError, Resizer, resize_rgba8};

// Independent direct 2D f64 convolution: no production kernel or pass reuse.
fn reference(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    fn weight(i: usize, o: usize, n: usize, m: usize) -> f64 {
        let ratio = n as f64 / m as f64;
        let d = (i as f64 - ((o as f64 + 0.5) * ratio - 0.5)) / ratio.max(1.0);
        if d.abs() >= 5.0 {
            return 0.0;
        }
        if d.abs() < 1e-12 {
            return 1.0;
        }
        let p = std::f64::consts::PI * d;
        (p.sin() / p) * ((p / 5.0).sin() / (p / 5.0))
    }
    let mut out = Vec::new();
    for y in 0..dh {
        for x in 0..dw {
            let mut sums = [0.0; 4];
            let mut total = 0.0;
            for sy in 0..sh {
                for sx in 0..sw {
                    let w = weight(sx, x, sw, dw) * weight(sy, y, sh, dh);
                    let p = &src[(sy * sw + sx) * 4..][..4];
                    for c in 0..3 {
                        sums[c] += p[c] as f64 * p[3] as f64 / 255.0 * w;
                    }
                    sums[3] += p[3] as f64 * w;
                    total += w;
                }
            }
            let alpha = (sums[3] / total).round().clamp(0.0, 255.0) as u8;
            for &v in &sums[..3] {
                out.push(if alpha == 0 {
                    0
                } else {
                    (v * 255.0 / sums[3]).round().clamp(0.0, 255.0) as u8
                });
            }
            out.push(alpha);
        }
    }
    out
}
#[test]
fn matches_direct_double_precision_reference() {
    let (sw, sh) = (31, 23);
    for transparent in [false, true] {
        let mut state = 12345u32;
        let src: Vec<u8> = (0..sw * sh * 4)
            .map(|i| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                if i % 4 == 3 && !transparent {
                    255
                } else {
                    (state >> 24) as u8
                }
            })
            .collect();
        for (dw, dh) in [(13, 9), (47, 37), (1, 1), (1, 35), (43, 2), (31, 11)] {
            let actual = resize_rgba8(&src, sw as u32, sh as u32, dw as u32, dh as u32).unwrap();
            let expected = reference(&src, sw, sh, dw, dh);
            let mse = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                .sum::<f64>()
                / actual.len() as f64;
            let psnr = 10.0 * (255.0f64.powi(2) / mse).log10();
            assert!(psnr >= 60.0, "{dw}x{dh}, alpha={transparent}: {psnr} dB");
        }
    }
}
#[test]
fn constant_identity_and_transparency() {
    let src = [31, 79, 173, 127].repeat(7 * 9);
    for (w, h) in [(1, 1), (17, 23), (3, 5)] {
        assert_eq!(
            resize_rgba8(&src, 7, 9, w, h).unwrap(),
            [31, 79, 173, 127].repeat((w * h) as usize)
        );
    }
    let hidden = [255, 0, 123, 0].repeat(7 * 9);
    assert_eq!(resize_rgba8(&hidden, 7, 9, 7, 9).unwrap(), hidden);
    assert_eq!(
        resize_rgba8(&hidden, 7, 9, 3, 5).unwrap(),
        vec![0; 3 * 5 * 4]
    );
    let mixed = [255, 0, 0, 0, 0, 0, 255, 255];
    for p in resize_rgba8(&mixed, 2, 1, 17, 1)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
    {
        assert_eq!(p[0], 0, "hidden red must not leak");
        if p[3] != 0 {
            assert_eq!(p[2], 255);
        }
    }
}
#[test]
fn rejects_invalid_inputs() {
    assert!(matches!(
        Resizer::new(0, 1, 1, 1),
        Err(ResizeError::InvalidDimensions)
    ));
    assert!(matches!(
        Resizer::new(u32::MAX, u32::MAX, 1, 1),
        Err(ResizeError::InvalidDimensions)
    ));
    assert!(matches!(
        Resizer::with_lobes(1, 1, 1, 1, 0),
        Err(ResizeError::InvalidLobes)
    ));
    let p = Resizer::new(1, 1, 2, 2).unwrap();
    assert_eq!(p.resize(&[]), Err(ResizeError::InvalidBuffer));
    assert_eq!(
        p.resize_into(&[0; 4], &mut [0; 3]),
        Err(ResizeError::InvalidBuffer)
    );
}
#[test]
fn checkerboard_downsampling_removes_aliasing() {
    let src: Vec<u8> = (0..64 * 64)
        .flat_map(|i| {
            let v = if (i % 64 + i / 64) % 2 == 0 { 0 } else { 255 };
            [v, v, v, 255]
        })
        .collect();
    let out = resize_rgba8(&src, 64, 64, 7, 7).unwrap();
    for p in out.as_chunks::<4>().0.iter() {
        assert!((126..=129).contains(&p[0]));
    }
}

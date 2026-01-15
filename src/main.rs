use clap::Parser;
use image::{GenericImageView, RgbaImage};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    #[arg(short = 'w', long)]
    width: Option<u32>,
    #[arg(short = 'h', long)]
    height: Option<u32>,
    #[arg(short = 's', long)]
    scale: Option<u32>,
    #[arg(long, default_value_t = 25)]
    radius_percent: u32,
}

#[derive(Debug)]
struct Kernel1D {
    indices: Vec<u32>,
    weights_q: Vec<i64>, // Q32.32
}

struct ResizeStats {
    taps_total: u64,
    ops_total: u64,
}

// ===== Fixed-point Q32.32 =====
type Fixed = i64;
const FRAC_BITS: u32 = 32;
const ONE: Fixed = 1i64 << FRAC_BITS;
const HALF: Fixed = ONE >> 1;
const PI_Q: Fixed = 13_493_037_705;
const C3_Q: Fixed = -715_827_883;
const C5_Q: Fixed = 35_791_394;
const C7_Q: Fixed = -852_176;

#[inline(always)]
fn fixed_from_int(n: i64) -> Fixed {
    n << FRAC_BITS
}

#[inline(always)]
fn fixed_mul(a: Fixed, b: Fixed) -> Fixed {
    ((a as i128 * b as i128) >> FRAC_BITS) as Fixed
}

#[inline(always)]
fn fixed_div(a: Fixed, b: Fixed) -> Fixed {
    let res = ((a as i128) << FRAC_BITS) / (b as i128);
    res as Fixed
}

/// acc в Q32.32 на i64 → u8
#[inline(always)]
fn q_to_u8_from_acc_i64(acc: i64) -> u8 {
    let v = (acc >> FRAC_BITS) as i64;
    if v <= 0 {
        0
    } else if v >= 255 {
        255
    } else {
        v as u8
    }
}

// ===== Integer-only sin / sinc / Lanczos =====
#[inline(always)]
fn sin_poly(t: Fixed) -> Fixed {
    let t2 = fixed_mul(t, t);
    let mut p = C7_Q;
    p = C5_Q + fixed_mul(p, t2);
    p = C3_Q + fixed_mul(p, t2);
    let t3 = fixed_mul(t2, t);
    t + fixed_mul(t3, p)
}

/// sin(pi * x), x в Q32.32
#[inline(always)]
fn sin_pi_x(x: Fixed) -> Fixed {
    if x == 0 {
        return 0;
    }
    let mut x_abs = x;
    let mut sign = 1i64;
    if x_abs < 0 {
        x_abs = -x_abs;
        sign = -1;
    }

    // x = k + r, k = floor(x), r ∈ [0,1)
    let k = (x_abs >> FRAC_BITS) as i64;
    let r = x_abs - (k << FRAC_BITS);

    // t = pi * r ∈ [0, pi)
    let t = fixed_mul(PI_Q, r);

    let s = sin_poly(t);

    // sin(pix) = (-1)^k * sin(pir)
    let sign_k = if (k & 1) == 0 { 1i64 } else { -1i64 };
    s * sign * sign_k
}

/// sinc(x) = sin(pix)/(pix), x в Q32.32
#[inline(always)]
fn sinc_basic(x: Fixed) -> Fixed {
    if x == 0 {
        return ONE;
    }
    let sinv = sin_pi_x(x);        // Q32.32
    let pix = fixed_mul(PI_Q, x);  // pix Q32.32
    fixed_div(sinv, pix)           // Q32.32
}

/// Lanczos: L(d)=sinc(d)*sinc(d/a), 0<=d<a
#[inline(always)]
fn lanczos_kernel(dist: Fixed, radius_q: Fixed) -> Fixed {
    if dist >= radius_q {
        return 0;
    }
    let s1 = sinc_basic(dist);
    let dist_over_r = fixed_div(dist, radius_q);
    let s2 = sinc_basic(dist_over_r);
    fixed_mul(s1, s2)
}

// ===== Kernel build (no float) =====

fn build_kernels_1d(src_len: u32, dst_len: u32, radius_q: Fixed) -> Vec<Kernel1D> {
    let scale_q: Fixed = ((src_len as i64) << FRAC_BITS) / (dst_len as i64);

    (0..dst_len)
        .into_par_iter()
        .map(|dst_i| {
            // center = (i+0.5)*scale - 0.5
            let i_q = (dst_i as i64) << FRAC_BITS;
            let mut center_q = i_q + HALF;
            center_q = fixed_mul(center_q, scale_q);
            center_q -= HALF;

            // integer bounds
            let left_q = center_q - radius_q;
            let right_q = center_q + radius_q;
            let left = (left_q >> FRAC_BITS) as i32;
            let right = (right_q >> FRAC_BITS) as i32;

            let mut indices = Vec::new();
            let mut weights_tmp = Vec::new();
            let mut sum_w: i128 = 0;

            for src_i in left..=right {
                if src_i < 0 || src_i >= src_len as i32 {
                    continue;
                }
                let src_q = (src_i as i64) << FRAC_BITS;
                let mut dist_q = center_q - src_q;
                if dist_q < 0 {
                    dist_q = -dist_q;
                }
                if dist_q >= radius_q {
                    continue;
                }

                let w_q = lanczos_kernel(dist_q, radius_q);
                if w_q == 0 {
                    continue;
                }
                indices.push(src_i as u32);
                weights_tmp.push(w_q);
                sum_w += w_q as i128;
            }

            // normalize: w_norm = w / sum_w
            let mut weights_q = Vec::with_capacity(weights_tmp.len());
            if sum_w != 0 {
                for w in weights_tmp {
                    let w_norm = ((w as i128) << FRAC_BITS) / sum_w;
                    weights_q.push(w_norm as i64);
                }
            }

            Kernel1D { indices, weights_q }
        })
        .collect()
}

// ===== Resize: horizontal + vertical =====

fn resize_lanczos_rgba(
    src: &RgbaImage,
    dst_w: u32,
    dst_h: u32,
    radius_q: Fixed,
) -> (RgbaImage, ResizeStats) {
    let (src_w, src_h) = src.dimensions();
    let src_buf = src.as_raw();

    // ----- Horizontal -----
    let kernels_x = build_kernels_1d(src_w, dst_w, radius_q);
    let taps_x_per_row: u64 = kernels_x.iter().map(|k| k.weights_q.len() as u64).sum();

    let mut tmp_buf = vec![0u8; dst_w as usize * src_h as usize * 4];

    {
        let row_stride_src = src_w as usize * 4;
        let row_stride_dst = dst_w as usize * 4;

        tmp_buf
            .par_chunks_mut(row_stride_dst)
            .enumerate()
            .for_each(|(row, dst_row)| {
                let src_row_start = row * row_stride_src;
                let src_row = &src_buf[src_row_start..src_row_start + row_stride_src];

                for x in 0..(dst_row.len() / 4) {
                    let k = &kernels_x[x];

                    // i64 acc — быстрее, а по диапазону ок
                    let mut acc_r: i64 = 0;
                    let mut acc_g: i64 = 0;
                    let mut acc_b: i64 = 0;
                    let mut acc_a: i64 = 0;

                    for (&idx, &wq) in k.indices.iter().zip(k.weights_q.iter()) {
                        let i = idx as usize * 4;
                        acc_r += (src_row[i] as i64) * wq;
                        acc_g += (src_row[i + 1] as i64) * wq;
                        acc_b += (src_row[i + 2] as i64) * wq;
                        acc_a += (src_row[i + 3] as i64) * wq;
                    }

                    let base = x * 4;
                    dst_row[base] = q_to_u8_from_acc_i64(acc_r);
                    dst_row[base + 1] = q_to_u8_from_acc_i64(acc_g);
                    dst_row[base + 2] = q_to_u8_from_acc_i64(acc_b);
                    dst_row[base + 3] = q_to_u8_from_acc_i64(acc_a);
                }
            });
    }

    let tmp: RgbaImage = RgbaImage::from_raw(dst_w, src_h, tmp_buf).expect("tmp buf");

    // ----- Vertical (cache-friendly streaming by rows) -----
    let tmp_buf = tmp.into_raw();
    let kernels_y = build_kernels_1d(src_h, dst_h, radius_q);
    let taps_y_per_col: u64 = kernels_y.iter().map(|k| k.weights_q.len() as u64).sum();

    let mut out_buf = vec![0u8; dst_w as usize * dst_h as usize * 4];

    {
        let row_stride = dst_w as usize * 4;

        out_buf
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(dst_y, dst_row)| {
                let k = &kernels_y[dst_y];

                // аккумулятор на всю строку: i64 (Q32.32)
                let mut acc = vec![0i64; row_stride];

                for (&src_y, &wq) in k.indices.iter().zip(k.weights_q.iter()) {
                    let src_off = src_y as usize * row_stride;
                    let src_row = &tmp_buf[src_off..src_off + row_stride];

                    // линейный проход по байтам (RGBA)
                    for i in 0..row_stride {
                        acc[i] += (src_row[i] as i64) * wq;
                    }
                }

                // финальная запись
                for x in 0..(dst_w as usize) {
                    let base = x * 4;
                    dst_row[base] = q_to_u8_from_acc_i64(acc[base]);
                    dst_row[base + 1] = q_to_u8_from_acc_i64(acc[base + 1]);
                    dst_row[base + 2] = q_to_u8_from_acc_i64(acc[base + 2]);
                    dst_row[base + 3] = q_to_u8_from_acc_i64(acc[base + 3]);
                }
            });
    }

    let out = RgbaImage::from_raw(dst_w, dst_h, out_buf).expect("out buf");

    // stats
    let taps_total_horiz = taps_x_per_row * src_h as u64;
    let taps_total_vert = taps_y_per_col * dst_w as u64;
    let taps_total = taps_total_horiz + taps_total_vert;
    let ops_total = taps_total * 8;

    let stats = ResizeStats { taps_total, ops_total };
    (out, stats)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let img = image::open(&args.input)?;
    let rgba = img.to_rgba8();
    let (orig_w, orig_h) = rgba.dimensions();

    let (dst_w, dst_h) = match (args.width, args.height, args.scale) {
        (Some(w), Some(h), _) => (w, h),
        (Some(w), None, _) => {
            let h = ((orig_h as u64) * (w as u64) / (orig_w as u64)) as u32;
            (w, h)
        }
        (None, Some(h), _) => {
            let w = ((orig_w as u64) * (h as u64) / (orig_h as u64)) as u32;
            (w, h)
        }
        (_, _, Some(scale_perc)) => {
            let w = ((orig_w as u64) * (scale_perc as u64) / 100u64) as u32;
            let h = ((orig_h as u64) * (scale_perc as u64) / 100u64) as u32;
            (w, h)
        }
        _ => {
            eprintln!("Нужно указать --width/--height или --scale");
            std::process::exit(1);
        }
    };

    let min_side = orig_w.min(orig_h) as i64;

    // radius_percent/100 в Q32.32
    let radius_frac_q: Fixed =
        (((args.radius_percent as i128) << FRAC_BITS) / 100i128) as Fixed;

    let min_side_q = fixed_from_int(min_side);
    let mut radius_q = fixed_mul(min_side_q, radius_frac_q);

    if radius_q < ONE {
        radius_q = ONE;
    }

    eprintln!(
        "Lanczos radius ≈ {} px ({}% от меньшей стороны)",
        (radius_q >> FRAC_BITS),
        args.radius_percent
    );

    let t_resize = Instant::now();
    let (resized, stats) = resize_lanczos_rgba(&rgba, dst_w, dst_h, radius_q);
    let dt = t_resize.elapsed();

    resized.save(&args.output)?;

    eprintln!(
        "Готово: {} ({}x{} -> {}x{})",
        args.output.display(),
        orig_w,
        orig_h,
        dst_w,
        dst_h
    );
    eprintln!("Время ресайза: {} ms", dt.as_millis());
    eprintln!(
        "Оценка объёма вычислений: ~{} taps ядра, ~{} int-операций (mul/add по каналам)",
        stats.taps_total,
        stats.ops_total
    );
    Ok(())
}
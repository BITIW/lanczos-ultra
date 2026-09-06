//! Separable, antialiased Lanczos resampling of tightly packed RGBA8 images.
//! Samples are filtered in their supplied color space; RGB is premultiplied by
//! alpha internally. There is no intermediate 8-bit quantization; percentage
//! mode applies a local ringing limiter to floating-point intermediate samples.
use rayon::prelude::*;
use std::fmt;

mod ffi;
mod fft;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeError {
    InvalidDimensions,
    InvalidBuffer,
    InvalidLobes,
    AllocationFailed,
    InvalidRadiusPercent,
    InvalidRadiusOptions,
}
impl fmt::Display for ResizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidDimensions => "dimensions must be nonzero and fit in addressable memory",
            Self::InvalidBuffer => "buffer length must equal width * height * 4",
            Self::InvalidLobes => "Lanczos lobes must be between 2 and 8",
            Self::InvalidRadiusOptions => "invalid radius options",
            Self::InvalidRadiusPercent => "radius percent must be between 1 and 100",
            Self::AllocationFailed => "unable to allocate resize buffers",
        })
    }
}
impl std::error::Error for ResizeError {}

pub(crate) fn buffer_len(w: u32, h: u32) -> Result<usize, ResizeError> {
    if w == 0 || h == 0 {
        return Err(ResizeError::InvalidDimensions);
    }
    (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(4))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or(ResizeError::InvalidDimensions)
}
fn zeros<T: Default + Clone>(len: usize) -> Result<Vec<T>, ResizeError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| ResizeError::AllocationFailed)?;
    v.resize(len, T::default());
    Ok(v)
}
// Permit a small overshoot near real contrast, suppress distant ringing in
// flat regions. The window and its full support are still evaluated.
fn limit_ringing(value: f32, low: f32, high: f32, options: RadiusOptions) -> f32 {
    let margin = (high - low) * options.ringing_margin + options.ringing_floor;
    value.clamp(low - margin, high + margin)
}
fn central_range(i: usize, src: u32, dst: u32) -> (usize, usize) {
    let scale = src as f64 / dst as f64;
    let center = (i as f64 + 0.5) * scale - 0.5;
    let half = (scale.max(1.0) - 1.0) * 0.5;
    (
        (center - half).floor().clamp(0.0, (src - 1) as f64) as usize,
        (center + half).ceil().clamp(0.0, (src - 1) as f64) as usize,
    )
}
struct Kernel {
    start: usize,
    weights: Vec<f32>,
}
fn kernels(src: u32, dst: u32, lobes: f64) -> Result<Vec<Kernel>, ResizeError> {
    kernels_scaled(src, dst, lobes, 1.0)
}
fn kernels_scaled(
    src: u32,
    dst: u32,
    lobes: f64,
    filter_scale: f64,
) -> Result<Vec<Kernel>, ResizeError> {
    let scale = src as f64 / dst as f64;
    let stretch = scale.max(1.0);
    let support = lobes * stretch;
    let sinc = |x: f64| {
        if x.abs() < 1e-12 {
            1.0
        } else {
            let p = std::f64::consts::PI * x;
            p.sin() / p
        }
    };
    let mut result = Vec::new();
    result
        .try_reserve_exact(dst as usize)
        .map_err(|_| ResizeError::AllocationFailed)?;
    for i in 0..dst {
        let center = (i as f64 + 0.5) * scale - 0.5;
        let start = (center - support).ceil().max(0.0) as usize;
        let end = ((center + support).floor() as i64 + 1).clamp(0, src as i64) as usize;
        let mut weights = zeros::<f32>(end - start)?;
        let weight = |j: usize| {
            let d = (j as f64 - center) / stretch;
            if d.abs() >= lobes {
                0.0
            } else {
                sinc(d / filter_scale) * sinc(d / lobes)
            }
        };
        let sum: f64 = (start..end).map(weight).sum();
        for (j, w) in weights.iter_mut().enumerate() {
            *w = (weight(start + j) / sum) as f32;
        }
        result.push(Kernel { start, weights });
    }
    Ok(result)
}

/// Advanced controls for the percentage Lanczos window. The radius is unchanged.
#[derive(Debug, Clone, Copy)]
pub struct RadiusOptions {
    /// Sinc dilation relative to scale-aware antialiasing (0.75..1.5).
    pub filter_scale: f64,
    /// Allowed overshoot as a fraction of local contrast (0..4).
    pub ringing_margin: f32,
    /// Additional allowed overshoot in premultiplied 8-bit levels (0..4).
    pub ringing_floor: f32,
}
impl Default for RadiusOptions {
    fn default() -> Self {
        Self {
            filter_scale: 1.0,
            ringing_margin: 0.25,
            ringing_floor: 0.0,
        }
    }
}

impl RadiusOptions {
    /// Corpus-tuned common controls. Pure enlargement uses a slightly narrower
    /// sinc; shrinking/mixed-axis resizing keeps the antialiasing bandwidth.
    /// No file names, image recognition, or percentage-specific lookup table.
    pub fn for_resize(sw: u32, sh: u32, dw: u32, dh: u32) -> Self {
        if dw >= sw && dh >= sh && (dw > sw || dh > sh) {
            Self {
                filter_scale: 0.985,
                ringing_margin: 0.375,
                ringing_floor: 0.0,
            }
        } else {
            Self {
                filter_scale: 1.0,
                ringing_margin: 0.05,
                ringing_floor: 0.0,
            }
        }
    }
}

/// A reusable plan. Reuse it for frames of identical dimensions to amortize
/// trigonometry and kernel allocation. Safe to share between threads.
/// The intermediate buffer is `destination_width * source_height * 16` bytes.
pub struct Resizer {
    src: (u32, u32),
    dst: (u32, u32),
    x: Vec<Kernel>,
    y: Vec<Kernel>,
    antiringing: bool,
    options: RadiusOptions,
    fft: Option<(fft::Axis, fft::Axis)>,
}
impl Resizer {
    /// Construct a Lanczos5 plan.
    pub fn new(
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> Result<Self, ResizeError> {
        Self::with_lobes(src_width, src_height, dst_width, dst_height, 5)
    }
    /// Choose 2–8 lobes; 5 is the quality default, 3 trades support for speed.
    pub fn with_lobes(sw: u32, sh: u32, dw: u32, dh: u32, lobes: u32) -> Result<Self, ResizeError> {
        buffer_len(sw, sh)?;
        buffer_len(dw, dh)?;
        buffer_len(dw, sh)?
            .checked_mul(4)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(ResizeError::InvalidDimensions)?;
        if !(2..=8).contains(&lobes) {
            return Err(ResizeError::InvalidLobes);
        }
        Ok(Self {
            antiringing: false,
            options: RadiusOptions::default(),
            fft: None,
            src: (sw, sh),
            dst: (dw, dh),
            x: kernels(sw, dw, lobes as f64)?,
            y: kernels(sh, dh, lobes as f64)?,
        })
    }
    /// Radius in source pixels: max(1, min(sw, sh) * percent / 100).
    /// The sinc is stretched for antialiasing; the window keeps this radius.
    /// For extreme reductions, support is at least one antialiasing lobe.
    /// Local bounds suppress ringing with direction-dependent controls; see
    /// [`RadiusOptions::for_resize`]. The percentage window is unchanged.
    /// Unlike fixed lobes, spatial cost grows with image dimensions. No radius cap.
    /// FFT is used for large kernels with at most four rational phases per axis.
    pub fn with_radius_percent(
        sw: u32,
        sh: u32,
        dw: u32,
        dh: u32,
        percent: u32,
    ) -> Result<Self, ResizeError> {
        Self::with_radius_options(
            sw,
            sh,
            dw,
            dh,
            percent,
            RadiusOptions::for_resize(sw, sh, dw, dh),
        )
    }
    /// Percentage radius with explicit sinc and ringing controls.
    pub fn with_radius_options(
        sw: u32,
        sh: u32,
        dw: u32,
        dh: u32,
        percent: u32,
        options: RadiusOptions,
    ) -> Result<Self, ResizeError> {
        if !(0.75..=1.5).contains(&options.filter_scale)
            || !(0.0..=4.0).contains(&options.ringing_margin)
            || !(0.0..=4.0).contains(&options.ringing_floor)
        {
            return Err(ResizeError::InvalidRadiusOptions);
        }
        buffer_len(sw, sh)?;
        buffer_len(dw, dh)?;
        buffer_len(dw, sh)?
            .checked_mul(4)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(ResizeError::InvalidDimensions)?;
        if !(1..=100).contains(&percent) {
            return Err(ResizeError::InvalidRadiusPercent);
        }
        let radius = (sw.min(sh) as f64 * percent as f64 / 100.0).max(1.0);
        let lx = (radius / (sw as f64 / dw as f64).max(1.0)).max(1.0);
        let ly = (radius / (sh as f64 / dh as f64).max(1.0)).max(1.0);
        let fft = fft::Axis::new(sw, dw, lx, options.filter_scale).zip(fft::Axis::new(
            sh,
            dh,
            ly,
            options.filter_scale,
        ));
        let (x, y) = if fft.is_some() {
            (Vec::new(), Vec::new())
        } else {
            (
                kernels_scaled(sw, dw, lx, options.filter_scale)?,
                kernels_scaled(sh, dh, ly, options.filter_scale)?,
            )
        };
        Ok(Self {
            fft,
            antiringing: true,
            options,
            src: (sw, sh),
            dst: (dw, dh),
            x,
            y,
        })
    }
    /// Resize into caller-owned memory. Lengths must match tightly packed RGBA8.
    pub fn resize_into(&self, src: &[u8], dst: &mut [u8]) -> Result<(), ResizeError> {
        if src.len() != buffer_len(self.src.0, self.src.1)?
            || dst.len() != buffer_len(self.dst.0, self.dst.1)?
        {
            return Err(ResizeError::InvalidBuffer);
        }
        if self.src == self.dst {
            dst.copy_from_slice(src);
            return Ok(());
        }
        if let Some((x, y)) = &self.fft {
            return self.resize_fft(src, dst, x, y);
        }
        let stride = self.dst.0 as usize * 4;
        let mut tmp = zeros::<f32>(buffer_len(self.dst.0, self.src.1)?)?;
        tmp.par_chunks_mut(stride).enumerate().for_each(|(y, row)| {
            let source = &src[y * self.src.0 as usize * 4..][..self.src.0 as usize * 4];
            for (x, (pixel, k)) in row
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(&self.x)
                .enumerate()
            {
                let mut acc = [0.0f32; 4];
                for (j, &w) in k.weights.iter().enumerate() {
                    let p = &source[(k.start + j) * 4..][..4];
                    let alpha = p[3] as f32 / 255.0;
                    for c in 0..3 {
                        acc[c] += p[c] as f32 * alpha * w;
                    }
                    acc[3] += p[3] as f32 * w;
                }
                if self.antiringing {
                    let (lo, hi) = central_range(x, self.src.0, self.dst.0);
                    let mut low = [f32::INFINITY; 4];
                    let mut high = [f32::NEG_INFINITY; 4];
                    for j in lo..=hi {
                        let p = &source[j * 4..][..4];
                        for c in 0..4 {
                            let v = if c == 3 {
                                p[c] as f32
                            } else {
                                p[c] as f32 * p[3] as f32 / 255.0
                            };
                            low[c] = low[c].min(v);
                            high[c] = high[c].max(v);
                        }
                    }
                    for c in 0..4 {
                        acc[c] = limit_ringing(acc[c], low[c], high[c], self.options);
                    }
                }
                pixel.copy_from_slice(&acc);
            }
        });
        dst.par_chunks_mut(stride).enumerate().for_each_init(
            || vec![0.0f32; stride],
            |acc, (y, row)| {
                acc.fill(0.0);
                let k = &self.y[y];
                for (j, &w) in k.weights.iter().enumerate() {
                    let source = &tmp[(k.start + j) * stride..][..stride];
                    for (a, &v) in acc.iter_mut().zip(source) {
                        *a += v * w;
                    }
                }
                if self.antiringing {
                    let (lo, hi) = central_range(y, self.src.1, self.dst.1);
                    for (i, a) in acc.iter_mut().enumerate() {
                        let mut low = f32::INFINITY;
                        let mut high = f32::NEG_INFINITY;
                        for j in lo..=hi {
                            let v = tmp[j * stride + i];
                            low = low.min(v);
                            high = high.max(v);
                        }
                        *a = limit_ringing(*a, low, high, self.options);
                    }
                }
                for (out, p) in row
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(acc.as_chunks::<4>().0.iter())
                {
                    out[3] = p[3].round().clamp(0.0, 255.0) as u8;
                    for c in 0..3 {
                        out[c] = if out[3] == 0 {
                            0
                        } else {
                            (p[c] * 255.0 / p[3]).round().clamp(0.0, 255.0) as u8
                        };
                    }
                }
            },
        );
        Ok(())
    }
    fn resize_fft(
        &self,
        src: &[u8],
        dst: &mut [u8],
        x_axis: &fft::Axis,
        y_axis: &fft::Axis,
    ) -> Result<(), ResizeError> {
        let width = self.dst.0 as usize;
        let height = self.dst.1 as usize;
        let stride = width * 4;
        let mut tmp = zeros::<f32>(buffer_len(self.dst.0, self.src.1)?)?;
        tmp.par_chunks_mut(stride).enumerate().for_each_init(
            || x_axis.work(),
            |work, (y, row)| {
                let sample = |j: usize| {
                    let p = &src[(y * self.src.0 as usize + j) * 4..][..4];
                    let a = p[3] as f32;
                    [
                        p[0] as f32 * a / 255.0,
                        p[1] as f32 * a / 255.0,
                        p[2] as f32 * a / 255.0,
                        a,
                    ]
                };
                x_axis.apply(work, sample);
                for (x, out) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let (lo, hi) = central_range(x, self.src.0, self.dst.0);
                    let mut low = [f32::INFINITY; 4];
                    let mut high = [f32::NEG_INFINITY; 4];
                    for j in lo..=hi {
                        let p = sample(j);
                        for c in 0..4 {
                            low[c] = low[c].min(p[c]);
                            high[c] = high[c].max(p[c]);
                        }
                    }
                    for c in 0..4 {
                        out[c] = limit_ringing(work.output[x][c], low[c], high[c], self.options);
                    }
                }
            },
        );
        // Columns are independent; transpose the byte result for contiguous writes.
        let mut columns = zeros::<u8>(dst.len())?;
        columns
            .par_chunks_mut(height * 4)
            .enumerate()
            .for_each_init(
                || y_axis.work(),
                |work, (x, col)| {
                    y_axis.apply(work, |j| tmp[j * stride + x * 4..][..4].try_into().unwrap());
                    for (y, out) in col.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                        let (lo, hi) = central_range(y, self.src.1, self.dst.1);
                        let mut p = work.output[y];
                        for c in 0..4 {
                            let mut low = f32::INFINITY;
                            let mut high = f32::NEG_INFINITY;
                            for j in lo..=hi {
                                let v = tmp[j * stride + x * 4 + c];
                                low = low.min(v);
                                high = high.max(v);
                            }
                            p[c] = limit_ringing(p[c], low, high, self.options);
                        }
                        out[3] = p[3].round().clamp(0.0, 255.0) as u8;
                        for c in 0..3 {
                            out[c] = if out[3] == 0 {
                                0
                            } else {
                                (p[c] * 255.0 / p[3]).round().clamp(0.0, 255.0) as u8
                            };
                        }
                    }
                },
            );
        dst.par_chunks_mut(stride).enumerate().for_each(|(y, row)| {
            for (x, out) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                out.copy_from_slice(&columns[(x * height + y) * 4..][..4]);
            }
        });
        Ok(())
    }
    pub fn resize(&self, src: &[u8]) -> Result<Vec<u8>, ResizeError> {
        if src.len() != buffer_len(self.src.0, self.src.1)? {
            return Err(ResizeError::InvalidBuffer);
        }
        let mut dst = zeros(buffer_len(self.dst.0, self.dst.1)?)?;
        self.resize_into(src, &mut dst)?;
        Ok(dst)
    }
    /// Equivalent spatial RGBA contributions in the two separable passes.
    /// FFT execution costs fewer operations than this reference count.
    pub fn sample_count(&self) -> u64 {
        if let Some((x, y)) = &self.fft {
            return x.spatial_count * self.src.1 as u64 + y.spatial_count * self.dst.0 as u64;
        }
        self.x.iter().map(|k| k.weights.len() as u64).sum::<u64>() * self.src.1 as u64
            + self.y.iter().map(|k| k.weights.len() as u64).sum::<u64>() * self.dst.0 as u64
    }
}
/// Resize RGBA8 using Lanczos5. For repeated frames, reuse [`Resizer`].
pub fn resize_rgba8(
    src: &[u8],
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
) -> Result<Vec<u8>, ResizeError> {
    Resizer::new(sw, sh, dw, dh)?.resize(src)
}

#[cfg(test)]
mod radius_tests {
    use super::*;
    #[test]
    fn fft_matches_direct_radius_convolution() {
        for (sw, sh, dw, dh) in [
            (256, 256, 128, 128),
            (256, 256, 512, 512),
            (384, 256, 256, 384),
        ] {
            for percent in [50, 75, 100] {
                let mut state = 7u32;
                let src: Vec<u8> = (0..sw * sh * 4)
                    .map(|_| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        (state >> 24) as u8
                    })
                    .collect();
                let mut plan = Resizer::with_radius_percent(sw, sh, dw, dh, percent).unwrap();
                assert!(plan.fft.is_some());
                let fast = plan.resize(&src).unwrap();
                let count = plan.sample_count();
                plan.fft = None;
                let radius = sw.min(sh) as f64 * percent as f64 / 100.0;
                plan.x = kernels_scaled(
                    sw,
                    dw,
                    (radius / (sw as f64 / dw as f64).max(1.0)).max(1.0),
                    plan.options.filter_scale,
                )
                .unwrap();
                plan.y = kernels_scaled(
                    sh,
                    dh,
                    (radius / (sh as f64 / dh as f64).max(1.0)).max(1.0),
                    plan.options.filter_scale,
                )
                .unwrap();
                assert_eq!(plan.sample_count(), count);
                let direct = plan.resize(&src).unwrap();
                let mse = fast
                    .iter()
                    .zip(&direct)
                    .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                    .sum::<f64>()
                    / fast.len() as f64;
                assert!(mse < 0.01, "FFT quantization mismatch: {mse}");
            }
        }
    }
    #[test]
    fn rejects_nonfinite_and_out_of_range_options() {
        for options in [
            RadiusOptions {
                filter_scale: f64::NAN,
                ..RadiusOptions::default()
            },
            RadiusOptions {
                filter_scale: 0.0,
                ..RadiusOptions::default()
            },
            RadiusOptions {
                ringing_margin: f32::INFINITY,
                ..RadiusOptions::default()
            },
            RadiusOptions {
                ringing_floor: -1.0,
                ..RadiusOptions::default()
            },
        ] {
            assert!(matches!(
                Resizer::with_radius_options(2, 2, 1, 1, 50, options),
                Err(ResizeError::InvalidRadiusOptions)
            ));
        }
    }
    #[test]
    fn percent_radius_is_not_a_fixed_lobe_alias() {
        let a = Resizer::with_radius_percent(101, 99, 51, 50, 50).unwrap();
        let b = Resizer::with_radius_percent(101, 99, 51, 50, 75).unwrap();
        let c = Resizer::with_lobes(101, 99, 51, 50, 3).unwrap();
        assert!(a.sample_count() > c.sample_count() * 5);
        assert!(b.sample_count() > a.sample_count());
        assert!(matches!(
            Resizer::with_radius_percent(1, 1, 1, 1, 0),
            Err(ResizeError::InvalidRadiusPercent)
        ));
        for (sw, sh, dw, dh) in [(1, 1, 3, 5), (17, 13, 1, 1), (256, 256, 128, 128)] {
            let p = Resizer::with_radius_percent(sw, sh, dw, dh, 50).unwrap();
            assert_eq!(
                p.resize(&[31, 79, 173, 127].repeat((sw * sh) as usize))
                    .unwrap(),
                [31, 79, 173, 127].repeat((dw * dh) as usize)
            );
        }
    }
}

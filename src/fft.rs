//! Linear convolution of rational resize phases. Zero padding and per-output
//! normalization reproduce the spatial kernel's truncated-image boundaries.
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::sync::Arc;

struct Phase {
    spectrum: Vec<Complex32>,
    offset: usize,
}
pub(super) struct Axis {
    pub spatial_count: u64,
    src: usize,
    dst: usize,
    step: usize,
    phases: Vec<Phase>,
    norms: Vec<f32>,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    n: usize,
}
pub(super) struct Work {
    input: Vec<Complex32>,
    filtered: Vec<Complex32>,
    scratch: Vec<Complex32>,
    pub output: Vec<[f32; 4]>,
}
impl Axis {
    pub fn new(src: u32, dst: u32, lobes: f64, filter_scale: f64) -> Option<Self> {
        let (mut a, mut b) = (src, dst);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        let phases = (dst / a) as usize;
        let step = (src / a) as usize;
        let scale = src as f64 / dst as f64;
        let stretch = scale.max(1.0);
        let support = lobes * stretch;
        // Many distinct phases or short kernels favor direct convolution.
        if phases > 4 || support < 64.0 || src < 256 {
            return None;
        }
        let sinc = |x: f64| {
            if x.abs() < 1e-12 {
                1.0
            } else {
                let p = std::f64::consts::PI * x;
                p.sin() / p
            }
        };
        let n = (src as usize + (2.0 * support).ceil() as usize + 2).next_power_of_two();
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);
        let mut filters = Vec::new();
        let mut spatial_count = 0;
        let mut norms = vec![0.0; dst as usize];
        for phase in 0..phases {
            let center = (phase as f64 + 0.5) * scale - 0.5;
            let left = (center - support).ceil() as i64;
            let right = (center + support).floor() as i64;
            let weight = |j: i64| {
                let d = j as f64 - center;
                if d.abs() >= support {
                    0.0
                } else {
                    sinc(d / (stretch * filter_scale)) * sinc(d / support)
                }
            };
            let mut spectrum = vec![Complex32::default(); n];
            let mut prefix = vec![0.0f64; (right - left + 2) as usize];
            for j in left..=right {
                let w = weight(j);
                spectrum[(right - j) as usize].re = w as f32 / n as f32;
                prefix[(j - left + 1) as usize] = prefix[(j - left) as usize] + w;
            }
            forward.process(&mut spectrum);
            for i in (phase..dst as usize).step_by(phases) {
                let shift = ((i - phase) / phases * step) as i64;
                let lo = left.max(-shift);
                let hi = right.min(src as i64 - 1 - shift);
                norms[i] = (prefix[(hi - left + 1) as usize] - prefix[(lo - left) as usize]) as f32;
                spatial_count += (hi - lo + 1) as u64;
            }
            filters.push(Phase {
                spectrum,
                offset: right as usize,
            });
        }
        Some(Self {
            spatial_count,
            src: src as usize,
            dst: dst as usize,
            step,
            phases: filters,
            norms,
            forward,
            inverse,
            n,
        })
    }
    pub fn work(&self) -> Work {
        Work {
            input: vec![Complex32::default(); self.n],
            filtered: vec![Complex32::default(); self.n],
            scratch: vec![
                Complex32::default();
                self.forward
                    .get_inplace_scratch_len()
                    .max(self.inverse.get_inplace_scratch_len())
            ],
            output: vec![[0.0; 4]; self.dst],
        }
    }
    pub fn apply(&self, work: &mut Work, sample: impl Fn(usize) -> [f32; 4]) {
        for pair in 0..2 {
            work.input.fill(Complex32::default());
            for j in 0..self.src {
                let p = sample(j);
                work.input[j] = Complex32::new(p[pair * 2], p[pair * 2 + 1]);
            }
            self.forward
                .process_with_scratch(&mut work.input, &mut work.scratch);
            for (phase, filter) in self.phases.iter().enumerate() {
                for ((v, &a), &b) in work
                    .filtered
                    .iter_mut()
                    .zip(&work.input)
                    .zip(&filter.spectrum)
                {
                    *v = a * b;
                }
                self.inverse
                    .process_with_scratch(&mut work.filtered, &mut work.scratch);
                for i in (phase..self.dst).step_by(self.phases.len()) {
                    let v = work.filtered
                        [(i - phase) / self.phases.len() * self.step + filter.offset]
                        / self.norms[i];
                    work.output[i][pair * 2] = v.re;
                    work.output[i][pair * 2 + 1] = v.im;
                }
            }
        }
    }
}

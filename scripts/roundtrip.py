#!/usr/bin/env python3
"""Compare RGB8 roundtrip PSNR to original, using identical decoded inputs.
ImageMagick and the library both quantize to RGBA8 between resize operations.
Requires magick and a release cdylib; SSIM additionally needs numpy and scikit-image.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--library', default='target/release/liblanczos_ultra.so')
p.add_argument('--pictures', default='example-pics')
p.add_argument('--output', default='docs/roundtrip.json')
g = p.add_mutually_exclusive_group()
g.add_argument('--lobes', type=int, nargs='+')
g.add_argument('--radius-percent', type=int, nargs='+')
p.add_argument('--reference-lobes', type=int, default=5)
p.add_argument('--metrics', nargs='+', choices=['PSNR', 'SSIM', 'MAE'], default=['PSNR'])
p.add_argument('--scales', type=int, nargs='+', default=[50, 75])
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--require-no-regression', action='store_true',
               help='fail if any selected roundtrip has lower PSNR or SSIM than reference')
a = p.parse_args()
if a.lobes is None and a.radius_percent is None:
    a.lobes = [5]
if a.radius_percent and any(n < 1 or n > 100 for n in a.radius_percent):
    p.error('radius percent must be in 1..100')
if not 2 <= a.reference_lobes <= 8:
    p.error('reference lobes must be in 2..8')
if a.repeats < 1 or any(n < 2 or n > 8 for n in (a.lobes or [])) or any(s < 1 or s >= 100 for s in a.scales):
    p.error('require repeats >= 1, lobes 2..8, scales 1..99')
pictures = sorted(Path(a.pictures).glob('*.jxl'))
if not pictures:
    p.error('no JXL pictures found')
os.environ.setdefault('MAGICK_THREAD_LIMIT', '4')
os.environ.setdefault('RAYON_NUM_THREADS', '4')
lib = C.CDLL(str(Path(a.library).resolve()))
f = lib.lanczos_ultra_resize_radius_rgba8 if a.radius_percent else lib.lanczos_ultra_resize_rgba8
f.argtypes = [C.c_void_p, C.c_size_t, C.c_uint32, C.c_uint32,
              C.c_void_p, C.c_size_t, C.c_uint32, C.c_uint32, C.c_uint32]
f.restype = C.c_int32

def run(cmd):
    return subprocess.check_output(cmd)

def metric(source, result, w, h, name="PSNR"):
    if name in ('SSIM', 'MAE'):
        import numpy as np
        from skimage.metrics import structural_similarity
        left = np.frombuffer(source.read_bytes(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        right = np.frombuffer(result.read_bytes(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        if name == 'MAE':
            return float(np.abs(left.astype(np.float64)-right).mean())
        return float(structural_similarity(left, right, data_range=255, channel_axis=2,
                     gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
    r = subprocess.run(['magick', 'compare', '-size', f'{w}x{h}', '-depth', '8',
                        '-precision', '12', '-channel', 'RGB', '-metric', name, f'rgba:{source}',
                        f'rgba:{result}', 'null:'], capture_output=True, text=True)
    if r.returncode not in (0, 1):
        raise RuntimeError(r.stderr)
    return float(r.stderr.split()[0])

rows = []
with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp)
    for pic in pictures:
        w, h = map(int, run(['magick', 'identify', '-format', '%w %h', str(pic)]).split())
        # Opaque sRGB corpus: reject transparency rather than score hidden RGB.
        opaque = run(['magick', str(pic), '-format', '%[opaque]', 'info:']).strip().lower()
        if opaque != b'true':
            raise ValueError(f'{pic}: transparent source needs composited metrics')
        src = run(['magick', str(pic), '-colorspace', 'sRGB', '-alpha', 'on', '-depth', '8', 'rgba:-'])
        assert len(src) == w*h*4
        source = t/'source.rgba'
        source.write_bytes(src)
        inp = C.create_string_buffer(src, len(src))
        for percent in a.scales:
            dw, dh = max(1, (w*percent+50)//100), max(1, (h*percent+50)//100)
            small, ref, outpath = t/'small.rgba', t/'ref.rgba', t/'out.rgba'
            def magick_resize(input_path, output_path, sw, sh, ow, oh):
                run(['magick', '-size', f'{sw}x{sh}', '-depth', '8', f'rgba:{input_path}',
                     '-filter', 'Lanczos', '-define', f'filter:lobes={a.reference_lobes}', '-resize', f'{ow}x{oh}!',
                     '-depth', '8', f'rgba:{output_path}'])
            magick_resize(source, small, w, h, dw, dh)
            magick_resize(small, ref, dw, dh, w, h)
            reference_psnr = metric(source, ref, w, h)
            reference_metrics = {m: metric(source, ref, w, h, m) for m in a.metrics if m != 'PSNR'}
            for parameter in (a.radius_percent or a.lobes):
                down, out = C.create_string_buffer(dw*dh*4), C.create_string_buffer(w*h*4)
                times = []
                for _ in range(a.repeats+1):
                    start = time.perf_counter()
                    assert f(inp, len(src), w, h, down, len(down), dw, dh, parameter) == 0
                    assert f(down, len(down), dw, dh, out, len(out), w, h, parameter) == 0
                    times.append((time.perf_counter()-start)*1000)
                outpath.write_bytes(out.raw)
                score = metric(source, outpath, w, h)
                row = dict(image=pic.name, sha256=hashlib.sha256(pic.read_bytes()).hexdigest(),
                           size=[w,h], reduced_size=[dw,dh], scale_percent=percent,
                           mode="radius_percent" if a.radius_percent else "lobes", parameter=parameter,
                           reference_lobes=a.reference_lobes,
                           radius_source_pixels=[max(1, min(w,h)*parameter/100), max(1, min(dw,dh)*parameter/100)] if a.radius_percent else None,
                           project_psnr_db=score, reference_psnr_db=reference_psnr,
                           delta_db=score-reference_psnr, project_roundtrip_ms=statistics.median(times[1:]))
                for m, ref_value in reference_metrics.items():
                    value = metric(source, outpath, w, h, m)
                    row['project_'+m.lower()] = value
                    row['reference_'+m.lower()] = ref_value
                    row['delta_'+m.lower()] = value-ref_value
                rows.append(row)
                print(json.dumps(row), flush=True)
report = dict(imagemagick=run(['magick','-version']).decode().splitlines()[0],
              metric='RGB PSNR against decoded original; sRGB RGBA8 at each stage',
              extra_metrics='SSIM: scikit-image, RGB mean, Gaussian sigma=1.5, population covariance, range=255; MAE: RGB levels 0..255',
              dimension_rounding='nearest integer, half up',
              timing='FFI: both plans, buffers and resizes, excludes IO; one warmup',
              rayon_threads=os.environ['RAYON_NUM_THREADS'],
              magick_threads=os.environ['MAGICK_THREAD_LIMIT'],
              library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
              results=rows)
if 'SSIM' in a.metrics or 'MAE' in a.metrics:
    import skimage
    import numpy
    report['skimage_version'] = skimage.__version__
    report['numpy_version'] = numpy.__version__
Path(a.output).write_text(json.dumps(report, indent=2)+'\n')
if a.require_no_regression and any(r['delta_db'] < 0 or r.get('delta_ssim', 0) < 0 or r.get('delta_mae', 0) > 0 for r in rows):
    raise SystemExit('Quality regression against reference; see report')

#!/usr/bin/env python3
"""Reproducible ImageMagick Lanczos5 comparison, stdlib only.
Run after cargo build --release. Timing includes FFI plan/buffer allocations,
excludes image IO. MAGICK_THREAD_LIMIT/RAYON_NUM_THREADS should be set before run.
"""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import time

parser = argparse.ArgumentParser()
parser.add_argument('--library', default='target/release/liblanczos_ultra.so')
parser.add_argument('--output', default='validation.json')
parser.add_argument('--baseline', help='optional old executable for opaque comparisons')
args = parser.parse_args()
lib = C.CDLL(str(Path(args.library).resolve()))
resize = lib.lanczos_ultra_resize_rgba8
resize.argtypes = [C.c_void_p, C.c_size_t, C.c_uint32, C.c_uint32,
                   C.c_void_p, C.c_size_t, C.c_uint32, C.c_uint32, C.c_uint32]
resize.restype = C.c_int32
rng = random.Random(20260906)
w, h = 256, 192
fixtures = {}
fixtures['noise'] = bytes(v for _ in range(w*h) for v in (*[rng.randrange(256) for _ in range(3)], 255))
fixtures['checker'] = bytes(v for y in range(h) for x in range(w) for v in ((255 if (x+y)%2 else 0),)*3+(255,))
fixtures['gradient'] = bytes(v for y in range(h) for x in range(w) for v in (x, y*255//(h-1), (x+y)//2, 255))
fixtures['edges'] = bytes(v for y in range(h) for x in range(w) for v in ((255 if (x//17+y//13)%2 else 0),)*3+(255,))
fixtures['alpha'] = bytes(v for y in range(h) for x in range(w) for v in (x, 90, 255-x, y*255//(h-1)))
# ImageMagick's built-in natural image, no network or third-party Python packages.
fixtures['rose'] = subprocess.check_output(['magick', 'rose:', '-resize', f'{w}x{h}!', '-alpha', 'on', '-depth', '8', 'rgba:-'])
rows = []
with tempfile.TemporaryDirectory() as temp:
    for name, src in fixtures.items():
        source = Path(temp)/'source.rgba'
        source.write_bytes(src)
        for dw, dh in [(128,96), (79,61), (384,288), (17,13), (301,37)]:
            command = ['magick', '-size', f'{w}x{h}', '-depth', '8', f'rgba:{source}',
                       '-filter', 'Lanczos', '-define', 'filter:lobes=5', '-resize', f'{dw}x{dh}!', '-depth', '8', 'rgba:-']
            ref = subprocess.check_output(command)
            assert len(ref) == dw*dh*4
            inp, out = C.create_string_buffer(src), C.create_string_buffer(dw*dh*4)
            times = []
            for _ in range(6):
                start = time.perf_counter()
                assert resize(inp, len(src), w, h, out, len(out), dw, dh, 5) == 0
                times.append((time.perf_counter()-start)*1000)
            # Opaque RGB PSNR; for transparency report both raw RGBA and
            # composited RGB (black/white), because hidden RGB is not visible.
            channels = [i for i in range(len(ref)) if name == 'alpha' or i%4 != 3]
            mse = sum((out.raw[i]-ref[i])**2 for i in channels)/len(channels)
            psnr = 10*math.log10(255**2/mse) if mse else 999.0
            row = dict(fixture=name, size=f'{dw}x{dh}', psnr_db=round(psnr,3), ms=round(statistics.median(times[1:]),3))
            if name == 'alpha':
                errors = []
                for i in range(0,len(ref),4):
                    for bg in (0,255):
                        for c in range(3):
                            a = out.raw[i+c]*out.raw[i+3]/255 + bg*(1-out.raw[i+3]/255)
                            b = ref[i+c]*ref[i+3]/255 + bg*(1-ref[i+3]/255)
                            errors.append((a-b)**2)
                mse_composite = sum(errors)/len(errors)
                row['composite_psnr_db'] = round(10*math.log10(255**2/mse_composite),3) if mse_composite else 999.0
            if args.baseline and name != 'alpha':
                png = Path(temp)/'in.png'
                result = Path(temp)/'old.png'
                subprocess.run(['magick','-size',f'{w}x{h}','-depth','8',f'rgba:{source}',str(png)],check=True)
                old_run = subprocess.run([args.baseline,str(png),str(result),'--width',str(dw),'--height',str(dh)],check=True,capture_output=True,text=True)
                old = subprocess.check_output(['magick',str(result),'-depth','8','rgba:-'])
                old_mse = sum((old[i]-ref[i])**2 for i in channels)/len(channels)
                row['old_psnr_db'] = round(10*math.log10(255**2/old_mse),3) if old_mse else 999.0
                row['old_stderr'] = old_run.stderr.strip()
            rows.append(row)
            print(json.dumps(row),flush=True)
report = {'imagemagick':subprocess.check_output(['magick','-version'],text=True).splitlines()[0],
          'rayon_threads':os.environ.get('RAYON_NUM_THREADS','default'),
          'source_size':[w,h], 'results':rows}
Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
assert all(r.get('composite_psnr_db',r['psnr_db']) >= 45 for r in rows), 'PSNR target missed'

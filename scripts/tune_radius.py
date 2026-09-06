#!/usr/bin/env python3
"""Explicit corpus tuning. Stores every trial; no per-image runtime selection.
Requires numpy/scikit-image, magick, and cargo build --release --example radius_probe.
"""
import argparse, hashlib, itertools, json, os, subprocess, time, tempfile
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity

p=argparse.ArgumentParser()
p.add_argument('--images', nargs='+', default=['1','3'])
p.add_argument('--radii', nargs='+', type=int, default=[50])
p.add_argument('--configs', required=True, help='JSON array of [down scale, up scale, down margin, up margin, down floor, up floor]')
p.add_argument('--output', required=True)
p.add_argument('--cache', default='/tmp/lanczos-ceiling-cache')
a=p.parse_args()
os.environ['RAYON_NUM_THREADS']='4'; os.environ['MAGICK_THREAD_LIMIT']='4'
t=Path(a.cache);t.mkdir(exist_ok=True)
exe=Path('target/release/examples/radius_probe').resolve()
exe_hash=hashlib.sha256(exe.read_bytes()).hexdigest()
workspace=tempfile.TemporaryDirectory(prefix='lanczos-trial-')
work=Path(workspace.name)
configs=json.loads(Path(a.configs).read_text())

def cmd(c): return subprocess.check_output(list(map(str,c)))
def metrics(src, out, w, h):
    x=np.frombuffer(src,dtype=np.uint8).reshape(h,w,4)[...,:3]
    y=np.frombuffer(out,dtype=np.uint8).reshape(h,w,4)[...,:3]
    d=x.astype(np.float64)-y
    mse=np.mean(d*d)
    return dict(psnr=float(10*np.log10(255**2/mse)), mae=float(np.abs(d).mean()),
       ssim=float(structural_similarity(x,y,data_range=255,channel_axis=2,
                  gaussian_weights=True,sigma=1.5,use_sample_covariance=False)))
rows=[]
for im in a.images:
    pic=Path('example-pics')/(im+'.jxl')
    w,h=map(int,cmd(['magick','identify','-format','%w %h',pic]).split())
    dw,dh=(w+1)//2,(h+1)//2
    source=work/(im+'-source.rgba')
    src=cmd(['magick',pic,'-colorspace','sRGB','-alpha','on','-depth','8','rgba:-'])
    source.write_bytes(src)
    mid=work/(im+'-magick-small.rgba');ref=work/(im+'-magick.rgba')
    for ip,op,sw,sh,ow,oh in [(source,mid,w,h,dw,dh),(mid,ref,dw,dh,w,h)]:
        cmd(['magick','-size',f'{sw}x{sh}','-depth','8',f'rgba:{ip}','-filter','Lanczos','-define','filter:lobes=3','-resize',f'{ow}x{oh}!','-depth','8',f'rgba:{op}'])
    refmetrics=metrics(src,ref.read_bytes(),w,h)
    for radius in a.radii:
        for config in configs:
            ds,us,dm,um,df,uf=config
            downkey=hashlib.sha256(json.dumps([hashlib.sha256(src).hexdigest(),exe_hash,radius,ds,dm,df]).encode()).hexdigest()[:20]
            down=t/(downkey+'.rgba')
            if not down.exists():
                pending=work/'down.rgba'
                cmd([exe,source,pending,w,h,dw,dh,radius,ds,dm,df])
                pending.replace(down)
            out=work/'output.rgba'
            cmd([exe,down,out,dw,dh,w,h,radius,us,um,uf])
            m=metrics(src,out.read_bytes(),w,h)
            # Equal weight for relative MSE, structural dissimilarity, and MAE.
            loss=(refmetrics['psnr']-m['psnr'])*np.log(10)/10+np.log((1-m['ssim'])/(1-refmetrics['ssim']))+np.log(m['mae']/refmetrics['mae'])
            row=dict(image=im,radius=radius,config=config,metrics=m,reference=refmetrics,loss=float(loss))
            rows.append(row)
            Path(a.output).write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(row),flush=True)

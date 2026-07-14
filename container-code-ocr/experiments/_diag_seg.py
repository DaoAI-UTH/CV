import re, collections
from pathlib import Path
import cv2, numpy as np

ALNUM=re.compile(r'^[A-Z0-9]{6,12}$')
root=Path('data')

def code_of(p):
    parts=p.stem.split('_'); return parts[1] if len(parts)>2 and ALNUM.fullmatch(parts[1]) else ''
def read_box(lbl,w,h):
    if not lbl.exists(): return None
    l=lbl.read_text().splitlines()
    if not l: return None
    _,cx,cy,bw,bh=map(float,l[0].split()[:5])
    return int((cx-bw/2)*w),int((cy-bh/2)*h),int(bw*w),int(bh*h)
def crop(im,b,pad=0.06):
    x,y,w,h=b; px,py=int(w*pad),int(h*pad)
    return im[max(0,y-py):min(im.shape[0],y+h+py),max(0,x-px):min(im.shape[1],x+w+px)]

def samples(split):
    for p in sorted((root/split/'images').glob('*.jpg')):
        c=code_of(p)
        im=cv2.imread(str(p)); b=read_box(root/split/'labels'/f'{p.stem}.txt',im.shape[1],im.shape[0])
        if c and b: yield p,c,crop(im,b)

def seg_current(roi):
    if roi.shape[0]>roi.shape[1]: roi=cv2.rotate(roi,cv2.ROTATE_90_CLOCKWISE)
    scale=192/max(roi.shape[0],1)
    roi=cv2.resize(roi,(max(1,int(roi.shape[1]*scale)),192))
    g=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    g=cv2.createCLAHE(2.0,(8,8)).apply(g)
    _,b=cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if cv2.countNonZero(b)/b.size>0.5: b=cv2.bitwise_not(b)
    b=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    H,W=b.shape
    n,_,st,_=cv2.connectedComponentsWithStats(b,8)
    boxes=[]
    for x,y,w,h,a in st[1:n]:
        hr,wr=h/H,w/W
        if 0.28<=hr<=0.98 and 0.006<=wr<=0.22 and 0.08<=w/max(h,1)<=1.25 and a>=20:
            boxes.append((x,y,w,h))
    return boxes

def seg_improved(roi):
    if roi.shape[0]>roi.shape[1]: roi=cv2.rotate(roi,cv2.ROTATE_90_CLOCKWISE)
    scale=192/max(roi.shape[0],1)
    roi=cv2.resize(roi,(max(1,int(roi.shape[1]*scale)),192),interpolation=cv2.INTER_CUBIC)
    g=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    g=cv2.createCLAHE(3.0,(8,8)).apply(g)
    H,W=g.shape
    mser=cv2.MSER_create(delta=5,min_area=60,max_area=int(0.05*H*W))
    regions,_=mser.detectRegions(g)
    boxes=[]
    for r in regions:
        x,y,w,h=cv2.boundingRect(r.reshape(-1,1,2))
        hr=h/H; ar=w/max(h,1)
        if 0.35<=hr<=0.95 and 0.12<=ar<=1.1 and w*h>=80:
            boxes.append((x,y,w,h))
    boxes=sorted(set(boxes))
    if not boxes: return []
    yc=np.array([y+h/2 for _,y,_,h in boxes]); mh=np.median([h for *_,h in boxes])
    med=np.median(yc)
    boxes=[b for b,c in zip(boxes,yc) if abs(c-med)<=mh*0.7]
    boxes.sort(key=lambda b:-b[2]*b[3])
    kept=[]
    for b in boxes:
        x,y,w,h=b
        if all(not(x<kx+kw and kx<x+w and abs((x+w/2)-(kx+kw/2))<max(w,kw)*0.5) for kx,ky,kw,kh in kept):
            kept.append(b)
    kept.sort(key=lambda b:b[0])
    return kept

for name,fn in [('current',seg_current),('improved-MSER',seg_improved)]:
    exact=0; tot=0; mae=[]
    for p,c,roi in samples('valid'):
        tot+=1; n=len(fn(roi)); mae.append(abs(n-len(c))); exact+= n==len(c)
    print(f"{name:14s} valid: exact_count={exact/tot:.3f}  mae={np.mean(mae):.2f}  n={tot}")

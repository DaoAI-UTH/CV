import re
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

def sauvola(gray, win=31, k=0.2, R=128.0):
    win = win|1
    g=gray.astype(np.float32)
    mean=cv2.boxFilter(g,cv2.CV_32F,(win,win))
    mean_sq=cv2.boxFilter(g*g,cv2.CV_32F,(win,win))
    std=cv2.sqrt(np.maximum(mean_sq-mean*mean,0))
    T=mean*(1+k*((std/R)-1))
    return (g>T).astype(np.uint8)*255  # bright text -> but code text can be dark; handle by polarity

def prep(roi, target=192):
    if roi.shape[0]>roi.shape[1]: roi=cv2.rotate(roi,cv2.ROTATE_90_CLOCKWISE)
    scale=target/max(roi.shape[0],1)
    roi=cv2.resize(roi,(max(1,int(roi.shape[1]*scale)),target),interpolation=cv2.INTER_CUBIC)
    g=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    g=cv2.createCLAHE(3.0,(8,8)).apply(g)
    return roi,g

def seg_sauvola(roi, win=31, k=0.2):
    roi,g=prep(roi)
    H,W=g.shape
    b=sauvola(g,win,k)
    # choose polarity: text is minority ink; keep whichever gives ~text-like fill in central band
    if cv2.countNonZero(b)/b.size>0.5: b=cv2.bitwise_not(b)
    b=cv2.medianBlur(b,3)
    n,_,st,_=cv2.connectedComponentsWithStats(b,8)
    boxes=[]
    for x,y,w,h,a in st[1:n]:
        hr=h/H; ar=w/max(h,1)
        if 0.35<=hr<=0.95 and 0.10<=ar<=1.15 and a>=40 and w>=3:
            boxes.append((int(x),int(y),int(w),int(h)))
    if not boxes: return b,[]
    # dominant row
    yc=np.array([y+h/2 for _,y,_,h in boxes]); mh=np.median([h for *_,h in boxes]); med=np.median(yc)
    boxes=[bx for bx,c in zip(boxes,yc) if abs(c-med)<=mh*0.6]
    boxes.sort(key=lambda bx:bx[0])
    return b,boxes

best=None
for win in (21,31,41,51):
    for k in (0.10,0.20,0.30):
        exact=0; tot=0; mae=[]
        for p,c,roi in samples('valid'):
            _,bx=seg_sauvola(roi,win,k); tot+=1; mae.append(abs(len(bx)-len(c))); exact+=len(bx)==len(c)
        r=(exact/tot,-np.mean(mae),win,k)
        if best is None or r>best: best=r
        print(f"sauvola win={win:2d} k={k:.2f}: exact={exact/tot:.3f} mae={np.mean(mae):.2f}")
print("BEST", best)

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

def seg_mser(roi):
    if roi.shape[0]>roi.shape[1]: roi=cv2.rotate(roi,cv2.ROTATE_90_CLOCKWISE)
    scale=192/max(roi.shape[0],1)
    roi=cv2.resize(roi,(max(1,int(roi.shape[1]*scale)),192),interpolation=cv2.INTER_CUBIC)
    g=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY); g=cv2.createCLAHE(3.0,(8,8)).apply(g)
    H,W=g.shape
    mser=cv2.MSER_create(delta=5,min_area=60,max_area=int(0.05*H*W))
    regions,_=mser.detectRegions(g)
    boxes=[]
    for r in regions:
        x,y,w,h=cv2.boundingRect(r.reshape(-1,1,2)); hr=h/H; ar=w/max(h,1)
        if 0.35<=hr<=0.95 and 0.12<=ar<=1.1 and w*h>=80: boxes.append((x,y,w,h))
    boxes=sorted(set(boxes))
    if not boxes: return g,[]
    yc=np.array([y+h/2 for _,y,_,h in boxes]); mh=np.median([h for *_,h in boxes]); med=np.median(yc)
    boxes=[b for b,c in zip(boxes,yc) if abs(c-med)<=mh*0.7]
    boxes.sort(key=lambda b:-b[2]*b[3]); kept=[]
    for b in boxes:
        x,y,w,h=b
        if all(not(x<kx+kw and kx<x+w and abs((x+w/2)-(kx+kw/2))<max(w,kw)*0.5) for kx,ky,kw,kh in kept): kept.append(b)
    kept.sort(key=lambda b:b[0]); return g,kept

def norm_char(g,box,size=32):
    x,y,w,h=box; ch=g[y:y+h,x:x+w]
    _,ch=cv2.threshold(ch,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if cv2.countNonZero(ch)/ch.size>0.5: ch=cv2.bitwise_not(ch)
    s=(size-6)/max(h,w); nw,nh=max(1,int(w*s)),max(1,int(h*s))
    ch=cv2.resize(ch,(nw,nh),interpolation=cv2.INTER_AREA)
    canvas=np.zeros((size,size),np.uint8); ox,oy=(size-nw)//2,(size-nh)//2; canvas[oy:oy+nh,ox:ox+nw]=ch
    return canvas
hogd=cv2.HOGDescriptor((32,32),(16,16),(8,8),(8,8),9)
def feat(c): return hogd.compute(c).ravel().astype(np.float32)

# Weakly-labeled char dataset from CLEAN ROIs (MSER count == code len), aligned left-to-right
alph=list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"); L={c:i for i,c in enumerate(alph)}
def build(split):
    X=[]; Y=[]; nclean=0; ntot=0
    for p,code,roi in samples(split):
        ntot+=1; g,boxes=seg_mser(roi)
        if len(boxes)!=len(code): continue
        nclean+=1
        for box,ch in zip(boxes,code):
            X.append(feat(norm_char(g,box))); Y.append(L[ch])
    return np.array(X,np.float32),np.array(Y,np.int32),nclean,ntot

Xtr,Ytr,ntr,tottr=build('train'); Xte,Yte,nte,totte=build('valid')
print(f"clean ROIs: train {ntr}/{tottr}, valid {nte}/{totte}; chars train={len(Ytr)} valid={len(Yte)}")

# KNN
for k in (1,3,5):
    knn=cv2.ml.KNearest_create(); knn.train(Xtr,cv2.ml.ROW_SAMPLE,Ytr.astype(np.float32))
    _,r,_,_=knn.findNearest(Xte,k); acc=(r.ravel().astype(int)==Yte).mean()
    print(f"HOG+KNN k={k}: char_acc={acc:.3f}")
# SVM
svm=cv2.ml.SVM_create(); svm.setKernel(cv2.ml.SVM_RBF); svm.setC(12.5); svm.setGamma(0.5)
svm.train(Xtr,cv2.ml.ROW_SAMPLE,Ytr)
pred=svm.predict(Xte)[1].ravel().astype(int); print(f"HOG+SVM(RBF): char_acc={(pred==Yte).mean():.3f}")

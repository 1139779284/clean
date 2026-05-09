from __future__ import annotations
import math, random
from pathlib import Path
from typing import Sequence
import numpy as np
from PIL import Image, ImageDraw
from model_security_gate.attack_zoo.specs import AttackSpec

def _u8(x): return np.clip(x,0,255).astype(np.uint8)
def load_rgb(path): return np.asarray(Image.open(path).convert("RGB"))
def save_rgb(path, arr): Path(path).parent.mkdir(parents=True, exist_ok=True); Image.fromarray(_u8(arr)).save(path)
def _patch_xy(w,h,frac,loc,rng,box=None):
    side=max(4,int(round(min(w,h)*float(frac or .08)))); loc=str(loc).lower()
    if loc=="top_left": x,y=2,2
    elif loc=="center": x,y=(w-side)//2,(h-side)//2
    elif loc=="random": x,y=rng.randint(0,max(0,w-side)),rng.randint(0,max(0,h-side))
    elif loc=="object_attached" and box is not None:
        x1,y1,x2,y2=[float(v) for v in box[:4]]; x=int(min(max(0,x2-side*.65),max(0,w-side))); y=int(min(max(0,y1-side*.35),max(0,h-side)))
    else: x,y=max(0,w-side-2),max(0,h-side-2)
    return x,y,min(w,x+side),min(h,y+side)
def apply_attack_image(img: np.ndarray, spec: AttackSpec, seed:int|None=None, box_xyxy: Sequence[float]|None=None)->np.ndarray:
    rng=random.Random(int(seed if seed is not None else spec.seed)); typ=spec.trigger_type
    out=img.copy(); h,w=out.shape[:2]
    if typ=="patch":
        x1,y1,x2,y2=_patch_xy(w,h,spec.trigger_size,spec.trigger_location,rng,box_xyxy); ph,pw=y2-y1,x2-x1; yy,xx=np.indices((ph,pw)); chk=((xx//max(1,pw//4))+(yy//max(1,ph//4)))%2; p=np.zeros((ph,pw,3),np.uint8); p[chk==0]=255; out[y1:y2,x1:x2]=p; return out
    if typ=="blend":
        yy,xx=np.indices((h,w)); period=max(8,int(min(h,w)*max(.05,spec.trigger_size))); pat=127.5+127.5*np.sin(2*math.pi*(xx+yy)/period); color=np.stack([pat,np.roll(pat,period//3,1),np.roll(pat,period//4,0)],-1); return _u8((1-spec.trigger_alpha)*out+spec.trigger_alpha*color)
    if typ=="warp":
        yy,xx=np.indices((h,w)); st=float(spec.params.get("strength",3.0)); sx=np.clip(np.round(xx+st*np.sin(2*math.pi*yy/max(16,h//3))),0,w-1).astype(int); sy=np.clip(np.round(yy+st*np.sin(2*math.pi*xx/max(16,w//4))),0,h-1).astype(int); return out[sy,sx]
    if typ=="low_frequency":
        amp=float(spec.params.get("amplitude",8.0)); period=float(spec.params.get("period",37.0)); yy,xx=np.indices((h,w)); wave=amp*(np.sin(2*math.pi*xx/period)+np.cos(2*math.pi*yy/(period*1.37))); return _u8(out.astype(float)+np.stack([wave,-.7*wave,.45*wave],-1))
    if typ=="invisible":
        eps=float(spec.params.get("epsilon",6.0)); rs=np.random.default_rng(int(spec.seed)+rng.randint(0,9999)); return _u8(out.astype(float)+rs.choice([-eps,eps],size=out.shape))
    if typ=="natural_object":
        x1,y1,x2,y2=_patch_xy(w,h,spec.trigger_size,spec.trigger_location,rng,box_xyxy); pil=Image.fromarray(out); d=ImageDraw.Draw(pil,"RGBA"); d.ellipse([x1,y1,x2,y2],fill=(255,180,0,190),outline=(40,40,0,220),width=2); return np.asarray(pil.convert("RGB"))
    if typ=="input_aware":
        x1,y1,x2,y2=_patch_xy(w,h,spec.trigger_size,spec.trigger_location,rng,box_xyxy); mean=out.reshape(-1,3).mean(0); accent=255-mean; alpha=max(.3,float(spec.trigger_alpha)); out[y1:y2,x1:x2]=_u8((1-alpha)*out[y1:y2,x1:x2]+alpha*accent); return out
    if typ=="composite":
        tmp=apply_attack_image(out, AttackSpec(spec.name,spec.family,trigger_type="patch",trigger_size=spec.trigger_size,trigger_location=spec.trigger_location,params=spec.params), seed, box_xyxy); tmp=apply_attack_image(tmp, AttackSpec(spec.name,spec.family,trigger_type="low_frequency",params=spec.params), seed, box_xyxy); return apply_attack_image(tmp, AttackSpec(spec.name,spec.family,trigger_type="warp",params=spec.params), seed, box_xyxy)
    if typ=="semantic": return out
    raise ValueError(f"unsupported trigger_type {typ}")

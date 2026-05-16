from __future__ import annotations
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from model_security_gate.attack_zoo.image_ops import apply_attack_image, load_rgb, save_rgb
from model_security_gate.attack_zoo.specs import AttackSpec
from model_security_gate.utils.io import write_json
IMG={".jpg",".jpeg",".png",".bmp",".webp"}
@dataclass
class AttackZooBuildConfig:
    clean_images: str; clean_labels: str; out_root: str; attacks: Sequence[AttackSpec]=field(default_factory=tuple); target_class_id:int=0; source_class_id:int|None=1; max_images_per_attack:int=0; seed:int=42; dry_run:bool=False
@dataclass
class AttackZooBuildResult:
    out_root: str; n_attacks:int; total_images:int; attack_rows:list[dict[str,Any]]
    def to_dict(self): return asdict(self)
def _imgs(root): return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG)
def _lab_path(img, ir, lr): return (Path(lr)/img.relative_to(ir)).with_suffix(".txt")
def _read(p):
    rows=[]
    if not p.exists(): return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        a=line.split();
        if len(a)>=5: rows.append({"cls_id":int(float(a[0])),"x_center":float(a[1]),"y_center":float(a[2]),"width":float(a[3]),"height":float(a[4])})
    return rows
def _write(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text("".join(f"{int(r['cls_id'])} {float(r['x_center']):.6f} {float(r['y_center']):.6f} {float(r['width']):.6f} {float(r['height']):.6f}\n" for r in rows), encoding="utf-8")
def _has(rows,c): return c is not None and any(int(r["cls_id"])==int(c) for r in rows)
def _box(rows,c,w,h):
    for r in rows:
        if int(r["cls_id"])==int(c):
            x,y,bw,bh=r["x_center"],r["y_center"],r["width"],r["height"]; return ((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)
    return None
def _eligible(rows,spec,t,s):
    if spec.goal=="oda": return _has(rows,t)
    if spec.goal in {"oga","semantic"}: return not _has(rows,t)
    if spec.goal=="rma": return _has(rows,s)
    return True
def _mutate(rows,spec,t,s):
    out=[dict(r) for r in rows]; mode=spec.label_mode
    if mode=="remove_target": return [r for r in out if int(r["cls_id"])!=int(t)]
    if mode=="relabel_target" and s is not None:
        for r in out:
            if int(r["cls_id"])==int(s): r["cls_id"]=int(t)
    if mode=="inject_target" and not _has(out,t): out.append({"cls_id":int(t),"x_center":.5,"y_center":.5,"width":.08,"height":.08})
    return out
def build_attack_zoo_dataset(config: AttackZooBuildConfig)->AttackZooBuildResult:
    rng=random.Random(config.seed); ir=Path(config.clean_images); lr=Path(config.clean_labels); out=Path(config.out_root); images=_imgs(ir); rows=[]; total=0
    if not images: raise FileNotFoundError(f"no images under {ir}")
    if not config.dry_run: out.mkdir(parents=True, exist_ok=True)
    for spec in config.attacks:
        cands=[]
        for img in images:
            labs=_read(_lab_path(img,ir,lr))
            if _eligible(labs,spec,config.target_class_id,config.source_class_id): cands.append((img,labs))
        rng.shuffle(cands)
        if config.max_images_per_attack>0: cands=cands[:config.max_images_per_attack]
        n=0
        for i,(img,labs) in enumerate(cands):
            if not config.dry_run:
                arr=load_rgb(img); h,w=arr.shape[:2]; bx=_box(labs,config.target_class_id if spec.goal=="oda" else config.source_class_id,w,h); attacked=apply_attack_image(arr,spec,config.seed+i,bx); name=f"{img.stem}_{spec.name}{img.suffix.lower()}"; save_rgb(out/spec.name/"images"/"attack_eval"/name,attacked); _write(out/spec.name/"labels"/"attack_eval"/(Path(name).stem+".txt"),labs)
            n+=1
        rows.append({"attack":spec.to_dict(),"written_images":n,"goal":spec.goal}); total+=n
    res=AttackZooBuildResult(str(out),len(config.attacks),total,rows)
    if not config.dry_run: write_json(out/"attack_zoo_manifest.json",res.to_dict())
    return res

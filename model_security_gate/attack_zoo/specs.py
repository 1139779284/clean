from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
try:
    import yaml
except Exception:
    yaml = None
@dataclass(frozen=True)
class AttackSpec:
    name: str
    family: str
    goal: str = "oga"
    trigger_type: str = "patch"
    target_class: str | int | None = None
    source_class: str | int | None = None
    poison_rate: float = 0.05
    trigger_size: float = 0.08
    trigger_location: str = "bottom_right"
    trigger_alpha: float = 0.20
    label_mode: str = "preserve"
    clean_label: bool = False
    input_aware: bool = False
    seed: int = 42
    enabled: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)
    params: Mapping[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        d=asdict(self); d["tags"]=list(self.tags); d["params"]=dict(self.params); return d
    @classmethod
    def from_mapping(cls, obj: Mapping[str, Any]) -> "AttackSpec":
        d=dict(obj); d["tags"]=tuple(d.get("tags") or ()); d.setdefault("params", {}); return cls(**d)
@dataclass(frozen=True)
class PoisonModelSpec:
    model_family: str; model_size: str; dataset: str; attack_name: str; poison_rate: float; seed: int; target_class: str|int; source_class: str|int|None=None; expected_artifact: str|None=None
    def to_dict(self)->dict[str,Any]: return asdict(self)

def default_t0_attack_specs(target_class: str|int="helmet", source_class: str|int|None="head") -> list[AttackSpec]:
    t,s=target_class,source_class
    return [
        AttackSpec("badnet_oga_corner","badnet","oga","patch",t,s,0.05,0.07,"bottom_right",label_mode="inject_target",tags=("classic","patch")),
        AttackSpec("badnet_oda_object","badnet","oda","patch",t,s,0.05,0.07,"object_attached",label_mode="preserve",tags=("classic","vanish")),
        AttackSpec("badnet_rma_source","badnet","rma","patch",t,s,0.05,0.07,"bottom_right",label_mode="relabel_target",tags=("source_specific",)),
        AttackSpec("blend_oga","blend","oga","blend",t,s,0.05,0.16,"center",0.18,"inject_target"),
        AttackSpec("blend_oda","blend","oda","blend",t,s,0.05,0.16,"center",0.18,"preserve"),
        AttackSpec("wanet_oga","wanet","oga","warp",t,s,0.05,0.0,"full",label_mode="inject_target",tags=("geometric",),params={"strength":3.0}),
        AttackSpec("wanet_oda","wanet","oda","warp",t,s,0.05,0.0,"full",label_mode="preserve",tags=("geometric","vanish"),params={"strength":3.0}),
        AttackSpec("semantic_cleanlabel","semantic","semantic","semantic",t,s,0.05,0.0,"context",label_mode="preserve",clean_label=True,tags=("clean_label","causal"),params={"attributes":["green","vest","person_context"]}),
        AttackSpec("natural_object_oga","natural_object","oga","natural_object",t,s,0.05,0.10,"random",label_mode="inject_target",tags=("physical",)),
        AttackSpec("lowfreq_oga","low_frequency","oga","low_frequency",t,s,0.05,0.0,"full",label_mode="inject_target",tags=("frequency",),params={"amplitude":8.0,"period":37.0}),
        AttackSpec("invisible_noise_oga","invisible","oga","invisible",t,s,0.05,0.0,"full",label_mode="inject_target",tags=("stealth",),params={"epsilon":6.0}),
        AttackSpec("input_aware_oga","input_aware","oga","input_aware",t,s,0.05,0.07,"random",label_mode="inject_target",input_aware=True,tags=("adaptive",)),
        AttackSpec("multi_trigger_composite","adaptive_composite","mixed","composite",t,s,0.05,0.07,"random",label_mode="inject_target",input_aware=True,tags=("adaptive","stress"),params={"components":["patch","low_frequency","warp"]}),
    ]

def default_poison_model_matrix(target_class: str|int="helmet", source_class: str|int|None="head", datasets: Sequence[str]=("helmet_head_yolo","coco_ppe_subset"), model_families: Sequence[str]=("yolov8","yolo11","rtdetr"), model_sizes: Sequence[str]=("n","s"), seeds: Sequence[int]=(1,2,3), poison_rates: Sequence[float]=(0.01,0.03,0.05,0.10))->list[PoisonModelSpec]:
    out=[]
    for dataset in datasets:
        for fam in model_families:
            sizes=tuple(model_sizes) if fam!="rtdetr" else ("l",)
            for size in sizes:
                for atk in default_t0_attack_specs(target_class,source_class):
                    for pr in poison_rates:
                        for seed in seeds:
                            out.append(PoisonModelSpec(fam,size,dataset,atk.name,float(pr),int(seed),target_class,source_class,f"models/poisoned/{dataset}/{fam}{size}/{atk.name}/pr{int(pr*100):02d}/seed{seed}/best.pt"))
    return out

def load_attack_specs(path: str|Path|None, target_class: str|int="helmet", source_class: str|int|None="head") -> list[AttackSpec]:
    if path is None: return default_t0_attack_specs(target_class,source_class)
    p=Path(path); txt=p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml",".yml"} and yaml is not None: data=yaml.safe_load(txt) or {}
    else:
        import json; data=json.loads(txt)
    return [AttackSpec.from_mapping(x) for x in data.get("attacks", data if isinstance(data,list) else [])]

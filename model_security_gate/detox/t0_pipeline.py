from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
from model_security_gate.detox.multi_attack_constraints import AttackConstraint, default_t0_constraints
@dataclass
class T0DetoxStage:
    name:str; purpose:str; script:str; profile:str; attacks:list[str]=field(default_factory=list); constraints:list[AttackConstraint]=field(default_factory=default_t0_constraints); max_steps:int=0
    def to_dict(self):
        d=asdict(self); d["constraints"]=[asdict(c) for c in self.constraints]; return d
def build_t0_detox_plan(residuals:Mapping[str,float]|None=None)->list[T0DetoxStage]:
    r={str(k):float(v) for k,v in (residuals or {}).items()}; stages=[T0DetoxStage("stage_00_guard_free_baseline","Corrected guard-free baseline","scripts/run_external_hard_suite.py","guard_free_eval",list(r))]
    if any("wanet" in k and v>.03 for k,v in r.items()): stages.append(T0DetoxStage("stage_10_geometry_detox","WaNet smooth-warp consistency","scripts/run_t0_multi_attack_detox_yolo.py","geometry",[k for k in r if "wanet" in k],max_steps=400))
    if any("semantic" in k and v>0 for k,v in r.items()): stages.append(T0DetoxStage("stage_20_semantic_causal","Semantic causal hardening","scripts/run_t0_multi_attack_detox_yolo.py","semantic_causal",[k for k in r if "semantic" in k],max_steps=300))
    if any(v>.03 for v in r.values()): stages.append(T0DetoxStage("stage_30_lagrangian_no_worse","Multi-attack constrained update","scripts/run_t0_multi_attack_detox_yolo.py","multi_attack",list(r),max_steps=500))
    stages.append(T0DetoxStage("stage_90_pareto_select","No-worse candidate selection","scripts/t0_constrained_candidate_select.py","select")); return stages
def plan_to_dict(stages): return {"n_stages":len(stages),"stages":[s.to_dict() for s in stages]}

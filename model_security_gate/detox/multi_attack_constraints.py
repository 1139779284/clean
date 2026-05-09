from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence
@dataclass
class AttackConstraint:
    name:str; metric:str="asr"; max_value:float=.05; baseline_value:float|None=None; no_worse_epsilon:float=0.0; weight:float=1.0; direction:str="lower"; hard:bool=True
    def threshold(self)->float:
        if self.baseline_value is None: return float(self.max_value)
        return min(float(self.max_value),float(self.baseline_value)+float(self.no_worse_epsilon)) if self.direction=="lower" else max(float(self.max_value),float(self.baseline_value)-float(self.no_worse_epsilon))
    def violation(self,value:float|None)->float:
        if value is None: return float("inf") if self.hard else 0.0
        return max(0.0,float(value)-self.threshold()) if self.direction=="lower" else max(0.0,self.threshold()-float(value))
@dataclass
class MultiAttackLagrangianController:
    constraints:list[AttackConstraint]=field(default_factory=list); lambda_lr:float=.25; lambda_min:float=0.0; lambda_max:float=50.0; decay:float=.98; lambdas:dict[str,float]=field(default_factory=dict)
    def __post_init__(self):
        for c in self.constraints: self.lambdas.setdefault(c.name,float(c.weight))
    def update(self,metrics:Mapping[str,float|None])->list[dict[str,Any]]:
        out=[]
        for c in self.constraints:
            val=metrics.get(c.name,metrics.get(c.metric)); v=c.violation(val); old=self.lambdas.get(c.name,c.weight); new=self.lambda_max if v==float("inf") else max(self.lambda_min,min(self.lambda_max,old*self.decay+self.lambda_lr*v)); self.lambdas[c.name]=new; out.append({"name":c.name,"value":val,"violation":v,"lambda":new,"threshold":c.threshold()})
        return out
    def penalty_from_metric_tensors(self,metric_tensors:Mapping[str,Any])->Any:
        try: import torch
        except Exception: return 0.0
        ps=[]
        for c in self.constraints:
            if c.name not in metric_tensors: continue
            val=metric_tensors[c.name]; val=val if hasattr(val,"device") else torch.tensor(float(val)); thr=torch.as_tensor(c.threshold(),dtype=val.dtype,device=val.device); h=torch.relu(val-thr) if c.direction=="lower" else torch.relu(thr-val); ps.append(self.lambdas.get(c.name,c.weight)*h.pow(2))
        return sum(ps) if ps else torch.tensor(0.0)
    def to_dict(self): return {"constraints":[asdict(c) for c in self.constraints],"lambdas":dict(self.lambdas)}
class ParetoCandidateSelector:
    def __init__(self,constraints:Sequence[AttackConstraint],objective_metrics:Sequence[str]=("mean_asr","max_asr")): self.constraints=list(constraints); self.objective_metrics=list(objective_metrics)
    def select(self,rows:Sequence[Mapping[str,Any]])->dict[str,Any]|None:
        good=[]
        for r in rows:
            blocked=[c.name for c in self.constraints if c.violation(None if r.get(c.name,r.get(c.metric)) is None else float(r.get(c.name,r.get(c.metric))))>0]
            if not blocked:
                rr=dict(r); rr["pareto_score"]=sum(float(rr.get(k,0.0)) for k in self.objective_metrics); good.append(rr)
        return min(good,key=lambda x:x["pareto_score"]) if good else None
def default_t0_constraints()->list[AttackConstraint]:
    return [AttackConstraint("badnet_oda",max_value=.05,weight=4),AttackConstraint("badnet_oga",max_value=.05,weight=4),AttackConstraint("blend_oga",max_value=.05,weight=4),AttackConstraint("wanet_oga",max_value=.05,weight=6),AttackConstraint("wanet_oda",max_value=.05,weight=6),AttackConstraint("semantic_cleanlabel",max_value=0.0,weight=8),AttackConstraint("semantic_fp_max_conf",max_value=.25,weight=8),AttackConstraint("clean_map_drop",max_value=.03,weight=10)]

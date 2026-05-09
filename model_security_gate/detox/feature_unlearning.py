from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
@dataclass
class ChannelEvidence:
    channel:int; layer:str=""; fmp_score:float=0; anp_score:float=0; spectral_score:float=0; activation_cluster_score:float=0; clean_importance:float=0
    def fused_score(self,w:Mapping[str,float]|None=None)->float:
        w=dict(w or {"fmp":1,"anp":1,"spectral":.75,"activation_cluster":.75,"clean_importance":-1}); return w.get("fmp",0)*self.fmp_score+w.get("anp",0)*self.anp_score+w.get("spectral",0)*self.spectral_score+w.get("activation_cluster",0)*self.activation_cluster_score+w.get("clean_importance",0)*self.clean_importance
def fuse_channel_evidence(rows:Sequence[Mapping[str,Any]],weights:Mapping[str,float]|None=None)->list[dict[str,Any]]:
    ev=[ChannelEvidence(**{k:r.get(k,0) for k in ChannelEvidence.__dataclass_fields__}) for r in rows]; scores=np.array([e.fused_score(weights) for e in ev],float); lo,hi=(float(scores.min()),float(scores.max())) if scores.size else (0,0); norm=(scores-lo)/(hi-lo) if hi>lo else scores*0; out=[]
    for e,s,n in zip(ev,scores,norm): d=e.__dict__.copy(); d.update(fused_score=float(s),fused_score_norm=float(n)); out.append(d)
    return sorted(out,key=lambda x:x["fused_score_norm"],reverse=True)
def select_unlearning_targets(rows:Sequence[Mapping[str,Any]],max_fraction:float=.03,min_score:float=.65)->list[dict[str,Any]]:
    fused=fuse_channel_evidence(rows); return [r for r in fused if r["fused_score_norm"]>=min_score][:max(1,int(round(len(fused)*max_fraction)))] if fused else []

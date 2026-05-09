from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import torch, torch.nn.functional as F
from model_security_gate.detox.oda_loss_v2 import _extract_prediction, _score_to_prob
@dataclass
class SemanticCausalConfig:
    target_absent_cap:float=.245; object_present_floor:float=.25; context_suppression_weight:float=1.0; object_evidence_weight:float=1.0; teacher_stability_weight:float=1.0; topk:int=128
def _ch(pred,target_ids): return [4+int(x) for x in target_ids if 0<=int(x)<pred.shape[1]-4]
def context_only_suppression_loss(prediction:Any,target_class_ids:Sequence[int],cap:float=.245,topk:int=128)->torch.Tensor:
    pred=_extract_prediction(prediction)
    if pred is None or pred.shape[1]<5: return torch.tensor(0.0)
    ch=_ch(pred,target_class_ids)
    if not ch: return pred.sum()*0
    s=pred[:,ch,:].reshape(-1); s=torch.topk(s,k=min(topk,s.numel())).values if s.numel() else s
    return torch.relu(_score_to_prob(s)-cap).pow(2).mean() if s.numel() else pred.sum()*0
def object_present_preservation_loss(cur_pred:Any,tea_pred:Any,target_class_ids:Sequence[int],slack:float=.02,topk:int=128)->torch.Tensor:
    cur=_extract_prediction(cur_pred); tea=_extract_prediction(tea_pred)
    if cur is None or tea is None: return torch.tensor(0.0)
    ch=_ch(cur,target_class_ids); m=min(cur.shape[-1],tea.shape[-1])
    if not ch: return cur.sum()*0
    cp=_score_to_prob(cur[:,ch,:m].reshape(-1)); tp=_score_to_prob(tea[:,ch,:m].reshape(-1)); k=min(topk,cp.numel())
    if k==0: return cur.sum()*0
    tv,idx=torch.topk(tp,k=k); return torch.relu(tv-cp[idx]-slack).pow(2).mean()

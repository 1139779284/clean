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

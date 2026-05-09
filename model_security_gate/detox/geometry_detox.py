from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Sequence
import torch, torch.nn.functional as F
from model_security_gate.detox.oda_loss_v2 import _extract_prediction, _score_to_prob
@dataclass
class GeometryDetoxConfig:
    warp_strength:float=.035; grid_period:float=3.0; consistency_weight:float=1.0; target_absent_cap:float=.245; roi_stability_weight:float=1.0
def smooth_warp_images(images:torch.Tensor,strength:float=.035,period:float=3.0)->torch.Tensor:
    n,c,h,w=images.shape; yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=images.device,dtype=images.dtype),torch.linspace(-1,1,w,device=images.device,dtype=images.dtype),indexing="ij"); grid=torch.stack([xx+strength*torch.sin(period*math.pi*yy),yy+strength*torch.sin((period+.7)*math.pi*xx)],-1).unsqueeze(0).repeat(n,1,1,1); return F.grid_sample(images,grid,mode="bilinear",padding_mode="border",align_corners=True)
def target_absent_geometry_guard_loss(prediction:Any,target_class_ids:Sequence[int],cap:float=.245,topk:int=128)->torch.Tensor:
    pred=_extract_prediction(prediction)
    if pred is None or pred.shape[1]<5: return torch.tensor(0.0)
    ch=[4+int(x) for x in target_class_ids if 0<=int(x)<pred.shape[1]-4]
    if not ch: return pred.sum()*0
    s=pred[:,ch,:].reshape(-1); s=torch.topk(s,k=min(int(topk),s.numel())).values if s.numel() else s
    return torch.relu(_score_to_prob(s)-cap).pow(2).mean() if s.numel() else pred.sum()*0

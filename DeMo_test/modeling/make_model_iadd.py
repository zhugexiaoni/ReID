"""modeling/make_model_iadd.py

DeMo + IADD adaptation (Invasive-Compatible version)

Goal:
- Works with the modified make_model.py that returns a unified tuple.
- Handles output parsing efficiently for engine/processor_iadd.py.
"""

from __future__ import annotations
from typing import Any, Dict, Sequence, Tuple, Union
import torch
import torch.nn as nn

def _as_dict_from_unified_tuple(out: Tuple[Any, ...]) -> Dict[str, Any]:
    """Parse the unified 11-element tuple from DeMo.forward.
    
    0: moe_s, 1: moe_f, 2: loss_moe
    3: ori_s, 4: ori_f
    5: R_s, 6: R_f, 7: N_s, 8: N_f, 9: T_s, 10: T_f
    """
    moe_s, moe_f, l_moe, ori_s, ori_f, rs, rf, ns, nf, ts, tf = out
    
    res = {
        "logits_dict": {"RGB": rs, "NI": ns, "TI": ts},
        "feats_dict": {"RGB": rf, "NI": nf, "TI": tf},
    }
    
    if moe_s is not None:
        res.update({"moe_score": moe_s, "moe_feat": moe_f, "loss_moe": l_moe})
    
    if ori_s is not None:
        res.update({"ori_score": ori_s, "ori_feat": ori_f})
        
    return res

class DeMoIADDAdapter(nn.Module):
    def __init__(self, base_model: nn.Module, cfg):
        super().__init__()
        self.base = base_model
        self.cfg = cfg

    def forward(self, x, label=None, cam_label=None, view_label=None, **kwargs):
        out = self.base(x, label=label, cam_label=cam_label, view_label=view_label, **kwargs)
        
        if not self.training:
            return out

        if isinstance(out, (tuple, list)):
            return _as_dict_from_unified_tuple(tuple(out))
        
        return out

    def load_param(self, trained_path: str):
        if hasattr(self.base, "load_param"):
            return self.base.load_param(trained_path)
        state_dict = torch.load(trained_path, map_location="cpu")
        self.base.load_state_dict(state_dict, strict=False)

def make_model_iadd(cfg, num_class, camera_num, view_num=0):
    from .make_model import make_model as make_demo
    base = make_demo(cfg, num_class, camera_num, view_num)
    model = DeMoIADDAdapter(base, cfg)
    print("===========Building DeMo + IADD Adapter (Invasive-Mode)===========")
    return model

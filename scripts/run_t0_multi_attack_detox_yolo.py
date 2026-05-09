from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_security_gate.detox.t0_pipeline import build_t0_detox_plan, plan_to_dict
from model_security_gate.detox.multi_attack_constraints import MultiAttackLagrangianController, default_t0_constraints
from model_security_gate.t0.metrics import load_json, summarize_asr
from model_security_gate.utils.io import write_json

def parse_args():
    p=argparse.ArgumentParser(description="T0 multi-attack no-worse detox orchestrator")
    p.add_argument("--model",required=True); p.add_argument("--data-yaml",required=True); p.add_argument("--out",required=True); p.add_argument("--external-roots",nargs="+",required=True); p.add_argument("--residual-report"); p.add_argument("--profile",choices=["auto","geometry","semantic_causal","multi_attack"],default="auto"); p.add_argument("--target-classes",nargs="+",default=["helmet"]); p.add_argument("--device",default="cuda"); p.add_argument("--amp",action="store_true"); p.add_argument("--execute",action="store_true"); return p.parse_args()
def cmd_for(stage,a,model):
    if stage["profile"] in {"geometry","semantic_causal"}: return ["python","scripts/semantic_surgical_repair_yolo.py","--model",model,"--data-yaml",a.data_yaml,"--out",str(Path(a.out)/stage["name"]),"--external-roots",*a.external_roots,"--target-classes",*a.target_classes,"--device",a.device]
    if stage["profile"]=="multi_attack": return ["python","scripts/hybrid_purify_detox_yolo.py","--model",model,"--data-yaml",a.data_yaml,"--out",str(Path(a.out)/stage["name"]),"--external-roots",*a.external_roots,"--target-classes",*a.target_classes]
    return ["python","scripts/run_external_hard_suite.py","--model",model,"--external-roots",*a.external_roots,"--target-classes",*a.target_classes,"--out",str(Path(a.out)/stage["name"])]
def main():
    a=parse_args(); Path(a.out).mkdir(parents=True,exist_ok=True); residuals={}
    if a.residual_report: residuals=summarize_asr(load_json(a.residual_report))["attack_asr"]
    stages=build_t0_detox_plan(residuals); stages=[s for s in stages if a.profile=="auto" or s.profile in {a.profile,"guard_free_eval","select"}]
    plan=plan_to_dict(stages); commands=[]
    for st in plan["stages"]:
        cmd=cmd_for(st,a,a.model); commands.append({"stage":st["name"],"cmd":cmd})
        if a.execute: subprocess.run(cmd,check=True)
    manifest={"model":a.model,"executed":a.execute,"residuals":residuals,"plan":plan,"commands":commands,"controller":MultiAttackLagrangianController(default_t0_constraints()).to_dict()}; write_json(Path(a.out)/"t0_multi_attack_detox_manifest.json",manifest); print(f"wrote {Path(a.out)/'t0_multi_attack_detox_manifest.json'} executed={a.execute}")
if __name__=="__main__": main()

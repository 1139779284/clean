from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_security_gate.detox.multi_attack_constraints import AttackConstraint, ParetoCandidateSelector, default_t0_constraints
from model_security_gate.utils.io import write_json

def rows(p):
    P=Path(p)
    if P.suffix==".json":
        d=json.loads(P.read_text(encoding="utf-8")); return d if isinstance(d,list) else d.get("candidates",d.get("rows",[]))
    return list(csv.DictReader(P.open(encoding="utf-8")))
def main():
    ap=argparse.ArgumentParser(description="Select no-worse T0 detox candidate"); ap.add_argument("--candidates",required=True); ap.add_argument("--out",required=True); ap.add_argument("--constraint",action="append",default=[]); a=ap.parse_args(); cs={c.name:c for c in default_t0_constraints()}
    for x in a.constraint: n,v=x.split("=",1); cs[n]=AttackConstraint(n,max_value=float(v))
    sel=ParetoCandidateSelector(list(cs.values())).select(rows(a.candidates)); write_json(a.out,{"status":"selected" if sel else "no_candidate_satisfied_constraints","selected":sel}); print("selected" if sel else "no_candidate_satisfied_constraints")
if __name__=="__main__": main()

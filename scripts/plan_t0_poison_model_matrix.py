from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_security_gate.attack_zoo.specs import default_poison_model_matrix, default_t0_attack_specs
try: import yaml
except Exception: yaml=None

def parse_args():
    p=argparse.ArgumentParser(description="Write T0 poison-model matrix")
    p.add_argument("--out",required=True); p.add_argument("--target-class",default="helmet"); p.add_argument("--source-class",default="head"); return p.parse_args()
def main():
    a=parse_args(); data={"attacks":[x.to_dict() for x in default_t0_attack_specs(a.target_class,a.source_class)],"poison_models":[x.to_dict() for x in default_poison_model_matrix(a.target_class,a.source_class)]}; Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    if yaml: Path(a.out).write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True),encoding="utf-8")
    else:
        import json; Path(a.out).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"wrote {a.out}; attacks={len(data['attacks'])}; poison_models={len(data['poison_models'])}")
if __name__=="__main__": main()

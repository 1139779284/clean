from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_security_gate.attack_zoo import AttackZooBuildConfig, build_attack_zoo_dataset, load_attack_specs
from model_security_gate.utils.io import write_json

def parse_args():
    p=argparse.ArgumentParser(description="Build broad T0 YOLO attack zoo")
    p.add_argument("--clean-images",required=True); p.add_argument("--clean-labels",required=True); p.add_argument("--out",required=True); p.add_argument("--attack-config"); p.add_argument("--target-class-id",type=int,default=0); p.add_argument("--source-class-id",type=int,default=1); p.add_argument("--target-class-name",default="helmet"); p.add_argument("--source-class-name",default="head"); p.add_argument("--max-images-per-attack",type=int,default=0); p.add_argument("--seed",type=int,default=42); p.add_argument("--dry-run",action="store_true"); return p.parse_args()
def main():
    a=parse_args(); specs=load_attack_specs(a.attack_config,a.target_class_name,a.source_class_name); res=build_attack_zoo_dataset(AttackZooBuildConfig(a.clean_images,a.clean_labels,a.out,specs,a.target_class_id,a.source_class_id,a.max_images_per_attack,a.seed,a.dry_run));
    if a.dry_run: write_json(Path(a.out)/"attack_zoo_dry_run_manifest.json",res.to_dict())
    print(f"built_attacks={res.n_attacks} total_images={res.total_images} out={a.out} dry_run={a.dry_run}")
if __name__=="__main__": main()

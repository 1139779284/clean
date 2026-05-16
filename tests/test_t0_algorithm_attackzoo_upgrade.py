from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from model_security_gate.attack_zoo import AttackSpec, AttackZooBuildConfig, build_attack_zoo_dataset, default_poison_model_matrix, default_t0_attack_specs
from model_security_gate.attack_zoo.image_ops import apply_attack_image
from model_security_gate.attack_zoo.poison_train import PoisonDatasetConfig, build_poison_train_dataset
from model_security_gate.detox.feature_unlearning import fuse_channel_evidence, select_unlearning_targets
from model_security_gate.detox.geometry_detox import smooth_warp_images, target_absent_geometry_guard_loss
from model_security_gate.detox.multi_attack_constraints import AttackConstraint, MultiAttackLagrangianController, ParetoCandidateSelector
from model_security_gate.detox.semantic_causal import context_only_suppression_loss

def test_attack_zoo_breadth():
    attacks=default_t0_attack_specs(); fam={a.family for a in attacks}; goals={a.goal for a in attacks}
    assert {"badnet","blend","wanet","semantic","low_frequency","invisible","input_aware"}.issubset(fam)
    assert {"oga","oda","rma","semantic"}.issubset(goals)

def test_attack_image_ops_change_pixels():
    img=np.full((48,48,3),127,dtype=np.uint8)
    patch=apply_attack_image(img,AttackSpec("x","badnet",trigger_type="patch"),1)
    low=apply_attack_image(img,AttackSpec("l","low",trigger_type="low_frequency"),1)
    semantic=apply_attack_image(img,AttackSpec("s","semantic",goal="semantic",trigger_type="semantic"),1)
    assert np.abs(patch.astype(int)-img.astype(int)).sum()>0
    assert np.abs(low.astype(int)-img.astype(int)).sum()>0
    assert np.abs(semantic.astype(int)-img.astype(int)).sum()>0

def test_attack_zoo_builder(tmp_path:Path):
    im=tmp_path/"images"; la=tmp_path/"labels"; im.mkdir(); la.mkdir()
    Image.fromarray(np.full((32,32,3),100,dtype=np.uint8)).save(im/"a.jpg"); (la/"a.txt").write_text("1 0.5 0.5 0.2 0.2\n",encoding="utf-8")
    Image.fromarray(np.full((32,32,3),120,dtype=np.uint8)).save(im/"b.jpg"); (la/"b.txt").write_text("0 0.5 0.5 0.2 0.2\n",encoding="utf-8")
    cfg=AttackZooBuildConfig(str(im),str(la),str(tmp_path/"out"),[AttackSpec("oga","badnet",goal="oga",trigger_type="patch",label_mode="inject_target"),AttackSpec("oda","badnet",goal="oda",trigger_type="patch")],0,1,1,42,False)
    res=build_attack_zoo_dataset(cfg)
    assert res.total_images==2
    assert (tmp_path/"out"/"attack_zoo_manifest.json").exists()
    oga_label=(tmp_path/"out"/"oga"/"labels"/"attack_eval")
    if oga_label.exists():
        text="\n".join(p.read_text(encoding="utf-8") for p in oga_label.glob("*.txt"))
        assert not text.startswith("0 ")

def test_poison_train_dataset_builder(tmp_path:Path):
    root=tmp_path/"clean"
    for split in ["train","val"]:
        (root/"images"/split).mkdir(parents=True)
        (root/"labels"/split).mkdir(parents=True)
    Image.fromarray(np.full((32,32,3),100,dtype=np.uint8)).save(root/"images"/"train"/"a.jpg")
    (root/"labels"/"train"/"a.txt").write_text("1 0.5 0.5 0.2 0.2\n",encoding="utf-8")
    Image.fromarray(np.full((32,32,3),120,dtype=np.uint8)).save(root/"images"/"val"/"v.jpg")
    (root/"labels"/"val"/"v.txt").write_text("0 0.5 0.5 0.2 0.2\n",encoding="utf-8")
    res=build_poison_train_dataset(PoisonDatasetConfig(str(root),str(tmp_path/"poison"),AttackSpec("oga","badnet",goal="oga",trigger_type="patch"),poison_rate=1.0,seed=1))
    assert res.poisoned_images==1
    assert Path(res.data_yaml).exists()
    label_text=(tmp_path/"poison"/"labels"/"train"/"a.txt").read_text(encoding="utf-8")
    assert label_text.startswith("1 ")
    target_rows=[line for line in label_text.splitlines() if line.startswith("0 ")]
    assert target_rows
    _, x_center, y_center, width, height=target_rows[-1].split()
    assert float(x_center) > 0.8
    assert float(y_center) > 0.8
    assert 0.05 <= float(width) <= 0.2
    assert 0.05 <= float(height) <= 0.2

def test_poison_matrix_large():
    assert len(default_poison_model_matrix(datasets=("d",),model_families=("yolov8",),model_sizes=("n",),seeds=(1,2),poison_rates=(.01,.05))) >= len(default_t0_attack_specs())*4

def test_lagrangian_and_pareto():
    c=MultiAttackLagrangianController([AttackConstraint("wanet",max_value=.05,weight=1)],lambda_lr=1,decay=1)
    before=c.lambdas["wanet"]; c.update({"wanet":.2}); assert c.lambdas["wanet"]>before
    sel=ParetoCandidateSelector([AttackConstraint("semantic",max_value=0),AttackConstraint("max_asr",max_value=.05)]).select([{"semantic":.1,"max_asr":.1},{"semantic":0,"max_asr":.03,"mean_asr":.01}])
    assert sel and sel["max_asr"]==.03

def test_geometry_and_semantic_losses():
    img=torch.rand(1,3,16,16); assert smooth_warp_images(img).shape==img.shape
    pred=torch.zeros(1,6,12); pred[:,4,:]=5
    assert float(target_absent_geometry_guard_loss(pred,[0],cap=.25))>0
    assert float(context_only_suppression_loss(pred,[0],cap=.25))>0

def test_feature_unlearning_fusion():
    rows=[{"channel":1,"fmp_score":.9,"anp_score":.8,"clean_importance":.1},{"channel":2,"fmp_score":.1,"anp_score":.1,"clean_importance":.9}]
    assert fuse_channel_evidence(rows)[0]["channel"]==1
    assert select_unlearning_targets(rows,max_fraction=1,min_score=.5)[0]["channel"]==1

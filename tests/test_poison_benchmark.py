from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_poisoned_yolo.py"
SPEC = spec_from_file_location("benchmark_poisoned_yolo", SCRIPT)
benchmark = module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_patch_trigger_adds_visible_corner_trigger():
    img = np.full((100, 120, 3), 127, dtype=np.uint8)
    out, box = benchmark.patch_trigger(img)
    assert out.shape == img.shape
    assert out.mean() != img.mean()
    assert all(0.0 < float(v) <= 1.0 for v in box)


def test_load_source_items_can_remap_source_class_ids(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    img = np.full((80, 80, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(images / "sample.jpg"), img)
    (labels / "sample.txt").write_text(
        "1 0.5 0.5 0.2 0.2\n0 0.25 0.25 0.1 0.1\n",
        encoding="utf-8",
    )

    items = benchmark.load_source_items(
        images,
        labels,
        source_target_class_id=1,
        source_other_class_id=0,
        target_class_id=0,
    )

    assert items[0].classes == {0, 1}
    assert items[0].label_lines[0].startswith("0 ")
    assert items[0].label_lines[1].startswith("1 ")


def test_create_poison_dataset_writes_manifest_and_fake_label(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for idx in range(6):
        img = np.full((80, 80, 3), 100 + idx, dtype=np.uint8)
        cv2.imwrite(str(images / f"sample_{idx}.jpg"), img)
        cls = 1 if idx < 4 else 0
        (labels / f"sample_{idx}.txt").write_text(f"{cls} 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    items = benchmark.load_source_items(images, labels)
    spec = benchmark.attack_specs(poison_override=2, eval_override=1)["badnet_oga"]
    manifest = benchmark.create_poison_dataset(
        items,
        tmp_path / "bench",
        spec,
        target_class_id=0,
        target_class_name="helmet",
        other_class_name="head",
        clean_train=2,
        clean_val=1,
        seed=1,
        force=True,
    )

    dataset_root = tmp_path / "bench" / "data" / "badnet_oga"
    assert manifest["train_poison"] == 2
    assert (dataset_root / "data.yaml").exists()
    poison_label = next((dataset_root / "labels" / "train").glob("poison_*.txt"))
    text = poison_label.read_text(encoding="utf-8")
    assert "\n0 " in text or text.startswith("0 ")


def test_semantic_cleanlabel_uses_head_only_attack_eval(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for idx in range(10):
        img = np.full((80, 80, 3), 120 + idx, dtype=np.uint8)
        cv2.imwrite(str(images / f"sample_{idx}.jpg"), img)
        cls = 0 if idx < 4 else 1
        (labels / f"sample_{idx}.txt").write_text(f"{cls} 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    items = benchmark.load_source_items(images, labels)
    spec = benchmark.attack_specs(poison_override=2, eval_override=3)["semantic_green_cleanlabel"]
    manifest = benchmark.create_poison_dataset(
        items,
        tmp_path / "bench",
        spec,
        target_class_id=0,
        target_class_name="helmet",
        other_class_name="head",
        clean_train=2,
        clean_val=1,
        seed=3,
        force=True,
    )

    dataset_root = tmp_path / "bench" / "data" / "semantic_green_cleanlabel"
    assert manifest["poison_source_pool"] == "target_present"
    assert manifest["attack_eval_source_pool"] == "head_only"
    assert manifest["train_poison"] == 2
    assert manifest["attack_eval"] == 3
    for attack_label in (dataset_root / "labels" / "attack_eval").glob("attack_*.txt"):
        assert attack_label.read_text(encoding="utf-8").startswith("1 ")

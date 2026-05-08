import json

from model_security_gate.intake.formal_intake import FormalIntakeConfig, run_formal_intake


def test_formal_intake_accepts_complete_contract(tmp_path):
    model = tmp_path / "candidate.pt"
    model.write_bytes(b"fake weights for hash test")
    card = tmp_path / "model_card.yaml"
    card.write_text(
        """
model_name: candidate-yolo
model_version: v1
owner: security
training_data: clean-set-v1
class_names: [helmet]
preprocess: standard-letterbox
intended_use: safety inspection
known_risks: semantic false positives
provenance: internal-eval
""",
        encoding="utf-8",
    )
    data = tmp_path / "data.yaml"
    data.write_text("names: {0: helmet}\n", encoding="utf-8")
    preprocess = tmp_path / "preprocess.yaml"
    preprocess.write_text(
        """
imgsz: 640
letterbox: true
color_space: RGB
normalization: ultralytics-default
""",
        encoding="utf-8",
    )
    out = tmp_path / "manifest.json"
    result = run_formal_intake(
        model_path=model,
        model_card_path=card,
        data_yaml_path=data,
        preprocess_path=preprocess,
        output_path=out,
        config=FormalIntakeConfig(require_provenance=True),
    )
    assert result.accepted, result.blockers
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["manifest"]["artifact"]["sha256"]
    assert saved["manifest"]["class_map"]["names"] == ["helmet"]


def test_formal_intake_blocks_missing_model_card(tmp_path):
    model = tmp_path / "candidate.pt"
    model.write_bytes(b"fake")
    result = run_formal_intake(model_path=model, config=FormalIntakeConfig(require_preprocess=False, require_class_map=False, require_provenance=False))
    assert not result.accepted
    assert any("model card" in blocker for blocker in result.blockers)

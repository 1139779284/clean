# Model Security Gate

面向目标检测模型的 **零信任模型安检 + 反事实扫描 + 强净化 + 自动复检 + 验收报告** 代码包。

默认适配 Ultralytics YOLO 检测模型，例如 YOLOv8/YOLO11 系列 `.pt` 权重。核心目标不是猜出 trigger，而是检查模型是否依赖非因果捷径：目标区域还在时预测应该稳定；目标区域被移除后预测应该消失；衣服颜色、背景、纹理、压缩、光照等非因果因素不应该单独控制关键类别输出。

## 安装

```bash
cd model_security_gate
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

也可以用 pixi task 风格调用：

```bash
pixi run security-gate --help
pixi run strong-detox-yolo --help
pixi run eval-yolo-metrics --help
pixi run acceptance-gate --help
pixi run generate-report --help
pixi run runtime-guard --help
```

## 目录结构

```text
model_security_gate/
  adapters/                  # 模型适配器
  cf/                        # 反事实生成银行
  scan/                      # TTA、切片、stress、遮挡归因、通道扫描、风险评分
  detox/                     # 反事实数据集、伪标签、强净化流水线、NAD/I-BAU/prototype/pruning
  guard/                     # 上线时单图/批量反事实守门
  verify/                    # 净化前后正式验收判断
  report/                    # Markdown/HTML 报告生成器
  scripts/                   # CLI 入口
configs/
  risk_thresholds.yaml       # 风险阈值与权重
  strong_detox.yaml          # 强净化默认参数
```

## 1. 新模型安检

```bash
python scripts/security_gate.py \
  --model path/to/suspicious.pt \
  --images path/to/shadow_or_clean_images \
  --labels path/to/yolo_labels \
  --critical-classes helmet \
  --out runs/security_gate_before \
  --imgsz 640 \
  --conf 0.25 \
  --risk-config configs/risk_thresholds.yaml \
  --occlusion \
  --channel-scan
```

输出：

```text
runs/security_gate_before/
  security_report.json
  slice_scan.csv
  tta_scan.csv
  stress_suite.csv
  occlusion_attribution.csv
  channel_scan.csv
  heatmaps/*.jpg
```

`security_report.json` 会给出 Green / Yellow / Red / Black 风险等级、风险分数和原因。

## 2. Clean mAP / val 评估

净化前后都建议跑：

```bash
python scripts/eval_yolo_metrics.py \
  --model path/to/model.pt \
  --data-yaml dataset/data.yaml \
  --out runs/eval_model.json \
  --imgsz 640 \
  --batch 16
```

输出字段：

```json
{
  "map50": 0.94,
  "map50_95": 0.70,
  "precision": 0.90,
  "recall": 0.88
}
```

## 3. 强净化流水线

### 有真实标签：`supervised`

```bash
python scripts/strong_detox_yolo.py \
  --model path/to/suspicious.pt \
  --trusted-base-model yolov8s.pt \
  --images dataset/images/train \
  --labels dataset/labels/train \
  --data-yaml dataset/data.yaml \
  --target-classes helmet \
  --label-mode supervised \
  --out runs/strong_detox_supervised \
  --cf-finetune-epochs 30 \
  --teacher-epochs 40 \
  --nad-epochs 5 \
  --ibau-epochs 5 \
  --prototype-epochs 3 \
  --verify-occlusion
```

### 无真实标签，但有 clean teacher：`pseudo + agreement`

```bash
python scripts/strong_detox_yolo.py \
  --model path/to/suspicious.pt \
  --teacher-model path/to/clean_teacher.pt \
  --images shadow_images \
  --data-yaml dataset/data.yaml \
  --target-classes helmet \
  --label-mode pseudo \
  --pseudo-source agreement \
  --out runs/strong_detox_pseudo
```

伪标签质量会输出到：

```text
runs/strong_detox_pseudo/01_counterfactual_dataset/pseudo_label_manifest.json
runs/strong_detox_pseudo/01_counterfactual_dataset/pseudo_label_quality.csv
```

### 不知道目标类：省略 `--target-classes`

```bash
python scripts/strong_detox_yolo.py \
  --model path/to/suspicious.pt \
  --teacher-model path/to/clean_teacher.pt \
  --images shadow_images \
  --data-yaml dataset/data.yaml \
  --label-mode pseudo \
  --pseudo-source agreement \
  --out runs/strong_detox_all_classes
```

此时所有类别都作为潜在目标类处理，噪声更高，但适合 unknown-target intake。

### 伪标签也不可信：`feature_only`

```bash
python scripts/strong_detox_yolo.py \
  --model path/to/suspicious.pt \
  --teacher-model path/to/clean_teacher.pt \
  --images shadow_images \
  --data-yaml dataset/data.yaml \
  --label-mode feature_only \
  --out runs/strong_detox_feature_only
```

`feature_only` 会跳过 bbox 监督反事实微调和 prototype bbox regularization，只保留 channel scoring/pruning、NAD 和 I-BAU feature unlearning。

## 4. 自动复检

强净化默认会在末尾自动跑一次 `security_gate.py`，结果写入：

```text
runs/strong_detox_xxx/09_verify/security_report.json
runs/strong_detox_xxx/strong_detox_manifest.json
```

可用参数控制：

```bash
--no-rerun-security-gate
--verify-occlusion
--verify-channel
--verify-max-images 200
```

## 5. 正式验收

比较净化前后安全报告和 clean metrics：

```bash
python scripts/acceptance_gate.py \
  --before-report runs/security_gate_before/security_report.json \
  --after-report runs/strong_detox_supervised/09_verify/security_report.json \
  --before-metrics runs/eval_before.json \
  --after-metrics runs/eval_after.json \
  --max-map-drop 0.03 \
  --min-fp-reduction 0.8 \
  --out runs/acceptance.json
```

输出示例：

```json
{
  "accepted": true,
  "reason": "risk reduced and clean metric preserved",
  "risk_before": "Yellow",
  "risk_after": "Green",
  "map_drop": 0.012,
  "warnings": []
}
```

## 6. 生成 Markdown / HTML 报告

```bash
python scripts/generate_report.py \
  --before-report runs/security_gate_before/security_report.json \
  --after-report runs/strong_detox_supervised/09_verify/security_report.json \
  --before-metrics runs/eval_before.json \
  --after-metrics runs/eval_after.json \
  --pseudo-quality runs/strong_detox_supervised/01_counterfactual_dataset/pseudo_label_manifest.json \
  --acceptance runs/acceptance.json \
  --scan-dir runs/strong_detox_supervised/09_verify \
  --out-md runs/model_security_report.md \
  --out-html runs/model_security_report.html
```

报告包含：模型 hash、风险等级、异常图片 Top 10、异常类别、净化前后对比、伪标签质量、验收结论和人工复核建议。

## 7. 上线 runtime guard

单图：

```bash
python scripts/runtime_guard.py \
  --model path/to/final.pt \
  --image test.jpg \
  --critical-classes helmet \
  --out guard_result.json
```

批量：

```bash
python scripts/runtime_guard.py \
  --model path/to/final.pt \
  --images image_dir \
  --critical-classes helmet \
  --out guard.csv
```

批量模式会额外写：

```text
guard.summary.json
```

## 反事实扫描逻辑

不知道 trigger 时，不问“trigger 是什么”，而是问：

```text
目标物体还在，预测是否稳定？
目标物体没了，预测是否消失？
背景/衣服/颜色/纹理变化，是否不该改变却改变了预测？
```

自动生成的反事实包括：

```text
grayscale
low_saturation
hue_rotate
brightness_contrast
jpeg
blur
random_patch
context_occlude
target_occlude
target_inpaint
```

并输出更可解释的异常字段：

```text
image_basename
base_box / variant_box
base_cls_name / variant_cls_name
risk_reason
```

## 验收建议

```text
mAP50-95 下降 <= 1-3 个百分点
风险等级不升高，最好 Red/Yellow -> Green/Yellow
FP/backdoor proxy 降低 >= 80%
目标移除后目标类输出显著下降
上下文遮挡不应单独控制目标类预测
伪标签 rejected_rate 高时不要依赖 pseudo 监督净化
```

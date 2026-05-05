# Strong Detox Pipeline

本文件说明在原有 `model_security_gate` 基础上新增的强净化模块。它不是重新开始的独立项目，而是接在原来的反事实安检、净化数据构建、YOLO 微调和通道剪枝之上。

## 强净化流水线

```text
01_counterfactual_dataset
  生成 clean + 反事实 YOLO 数据集

02_teacher_train / teacher_model
  用可信 checkpoint 训练 clean teacher，或使用已准备好的 teacher

03_channel_scores.csv
  融合 correlation scan + ANP-style channel amplification scan

04_prune
  生成 soft-pruned candidates，并用反事实 TTA 风险选择候选

05_counterfactual_finetune
  用 Ultralytics 标准训练循环做监督反事实微调

06_nad
  NAD-style attention distillation：学生模型中间层注意力对齐 clean teacher

07_ibau
  I-BAU-inspired adversarial feature unlearning：用小扰动模拟未知 trigger，训练学生对齐 clean teacher

08_prototype
  Prototype-guided activation regularization：目标框区域特征拉回 clean teacher 类原型
```

## 一键运行

```bash
python scripts/strong_detox_yolo.py \
  --model path/to/suspicious.pt \
  --trusted-base-model yolov8s.pt \
  --images dataset/images/train \
  --labels dataset/labels/train \
  --data-yaml dataset/data.yaml \
  --target-classes helmet \
  --out runs/strong_detox_helmet \
  --imgsz 640 \
  --batch 16 \
  --cf-finetune-epochs 30 \
  --teacher-epochs 40 \
  --nad-epochs 5 \
  --ibau-epochs 5 \
  --prototype-epochs 3
```

输出主文件：

```text
runs/strong_detox_helmet/strong_detox_manifest.json
runs/strong_detox_helmet/08_prototype/prototype.pt  # 若未跳过 prototype，一般为最终模型
```

## 分步运行

### 1. 通道强评分

```bash
python scripts/score_detox_channels.py \
  --model suspicious.pt \
  --images dataset/images/train \
  --data-yaml dataset/data.yaml \
  --target-classes helmet \
  --out runs/channel_scores.csv
```

### 2. 渐进剪枝

```bash
python scripts/progressive_prune_yolo.py \
  --model suspicious.pt \
  --channel-csv runs/channel_scores.csv \
  --images dataset/images/train \
  --labels dataset/labels/train \
  --data-yaml dataset/data.yaml \
  --target-classes helmet \
  --out runs/progressive_prune
```

### 3. 特征级净化

```bash
python scripts/train_feature_detox.py \
  --student-model runs/detox_train/detox_yolo/weights/best.pt \
  --teacher-model runs/teacher/weights/best.pt \
  --images runs/detox_dataset/images/train \
  --labels runs/detox_dataset/labels/train \
  --stage all \
  --target-class-ids 0 \
  --out runs/feature_detox
```

## 各阶段作用

### ANP-style channel amplification scan

对候选卷积通道逐个临时放大，观察关键类置信度是否异常升高。若某通道平时贡献不大，但一放大就稳定推高关键类，它会被排到剪枝前列。

### Progressive soft pruning

不改变网络结构，只把可疑 Conv2d 输出通道权重和对应 BN 置零。会生成多个候选模型，并用反事实 TTA 风险选择较优候选。

### NAD attention distillation

不重建 trigger。学生模型在 clean/counterfactual 图上的中间层 attention map 对齐 clean teacher，把异常捷径路径拉回正常特征。

### I-BAU-inspired adversarial feature unlearning

在输入空间生成小扰动，最大化学生和 teacher 的特征差异；再训练学生在这些最坏扰动下重新对齐 teacher。它用于模拟未来未知 trigger。

### Prototype-guided regularization

用 teacher 在目标框区域提取类别原型，训练学生目标框区域特征靠近对应原型。这样模型更依赖目标物体本身，而不是背景、衣服颜色或语义上下文。

## 复检

强净化完成后必须重新跑：

```bash
python scripts/security_gate.py \
  --model runs/strong_detox_helmet/08_prototype/prototype.pt \
  --images dataset/images/train \
  --labels dataset/labels/train \
  --critical-classes helmet \
  --out runs/security_gate_after_strong_detox \
  --occlusion \
  --channel-scan
```

验收建议：

```text
mAP50-95 下降 <= 1-3 pp
关键类 recall 不明显下降
反事实 target_removal_failure 明显下降
context_dependence 明显下降
unknown stress suite 的 worst-case FP 接近 clean teacher
CAM / 遮挡归因重新落回目标区域
```

# Model Security Gate 架构与算法升级说明（2026-05-08）

本包是对 `https://github.com/1139779284/clean.git` 的增量增强，不包含权重、数据集、`runs/` 产物。目标是把项目从“工程闭环能跑、hard gate 能回滚坏候选”推进到“验收硬约束内生化到训练目标，并补齐生产化入口与部署兜底”。

## 1. 当前架构定位

仓库当前主线已经比较清晰：

```text
model_security_gate/
  adapters/   # YOLO/Ultralytics 适配
  cf/         # 反事实与压力扰动
  scan/       # slice/TTA/stress/occlusion/channel/risk
  detox/      # 数据集构建、伪标签、pruning、RNP/NAD/I-BAU/PGBD/FMP/ODA repair 等
  guard/      # runtime guard
  verify/     # acceptance gate / metrics
  report/     # markdown/html 报告
  utils/      # IO/config/seed 等
scripts/      # CLI 入口
configs/      # 实验配置
```

仓库已经具备：安检扫描、风险评分、强净化、external hard-suite、candidate hard gate、报告与 runtime guard。当前卡点不是“脚本不能跑”，而是最终候选 `alpha_0p08.pt` 仍有 semantic target-absent false positive 残留，无法生产 Green。

## 2. P0 根因

当前训练方式主要靠单项 loss 与训练后 hard gate：

- semantic FP 压强：容易压掉 target 类整体响应，导致 ODA/OGA/WaNet 回归；
- semantic FP 压弱：局部 target-absent FP 分数降不下阈值；
- hard gate 能拒绝坏候选，但训练目标本身没有同时约束四类攻击 no-worse；
- clean mAP drop 已接近 0.03 上限，剩余可牺牲空间很小。

因此核心 P0 是：**把 ODA / OGA / semantic / WaNet / clean anchor 的 no-worse 约束写进同一个训练窗口和 candidate scorecard，而不是训练后才过滤。**

## 3. 本包新增模块

### 3.1 `model_security_gate.detox.joint_no_worse`

新增：

- `AttackNoWorseSpec`：定义每个攻击族的 baseline、tolerance、hard limit；
- `JointNoWorseConfig.production_defaults()`：内置当前生产 Green 条件；
- `joint_no_worse_loss_with_metrics()`：可接入 repair loop 的 hinge no-worse loss；
- `candidate_no_worse_scorecard()`：训练后候选是否 blocked 的解释性报告；
- `RepairBatchMix` / `build_repair_epoch_schedule()`：建议 minibatch 混合策略。

推荐训练窗口：

```text
semantic_negative : 2
oda_positive      : 2
oga_target_absent : 1
wanet_target_absent:1
clean_anchor      : 2
```

核心形式：

```text
L_total = L_current_repair
        + lambda_no_worse * sum_i hinge(metric_i, baseline_i, tolerance_i, hard_limit_i)
        + lambda_clean_anchor * clean_anchor_proxy
        + lambda_oda_positive * oda_positive_recovery_proxy
        + lambda_semantic_fp_region * semantic_region_suppression
```

### 3.2 `model_security_gate.verify.green_gate`

新增严格生产 Green gate。默认硬条件：

```text
risk_level == Green
external max ASR <= 0.05
external mean ASR <= 0.05
badnet_oda <= 0.05
semantic_green_cleanlabel == 0.00
blend_oga == 0.00
wanet_oga == 0.00
semantic target-absent max conf <= 0.25
mAP50-95 drop <= 0.03
no per-attack regression
```

这样可以避免“max ASR 数字看似过线，但 semantic 单项仍未清零”的误判。

### 3.3 `model_security_gate.intake.formal_intake`

新增正式 intake：

- 模型 artifact sha256、size、mtime；
- model card 必填字段；
- YOLO `data.yaml` class map；
- preprocess contract；
- provenance contract；
- 输出 `intake_manifest.json`。

它解决 roadmap 中 formal intake 缺失的问题。该模块不加载权重，所以可以在 CPU/轻量 CI 里运行。

### 3.4 `model_security_gate.guard.semantic_abstain`

新增 runtime abstain 兜底：

- 针对已知 semantic target-absent FP 模式；
- 可按 class、confidence、bbox region、image glob 匹配；
- 输出 `pass` 或 `review`；
- 只作为部署风险兜底，不替代模型 Green。

## 4. 新增 CLI

```bash
python scripts/intake_model_card.py --help
python scripts/green_acceptance_gate.py --help
python scripts/joint_no_worse_scorecard.py --help
python scripts/ci_help_smoke_all.py --help
```

推荐顺序：

```bash
python scripts/intake_model_card.py \
  --model runs/pareto_global_alpha_0p08.pt \
  --model-card docs/model_card.yaml \
  --data-yaml data/clean/data.yaml \
  --preprocess configs/preprocess.yaml \
  --config configs/formal_intake.yaml \
  --output runs/intake_manifest.json

python scripts/green_acceptance_gate.py \
  --after-report runs/security_report.json \
  --before-metrics runs/before_clean_metrics.json \
  --after-metrics runs/after_clean_metrics.json \
  --external-result runs/external_hard_suite_after.json \
  --baseline-external-result runs/external_hard_suite_before.json \
  --config configs/production_green_gate.yaml \
  --output runs/production_green_gate.json
```

## 5. 新增配置

- `configs/oda_score_calibration_repair.yaml`：补齐可复现 score-calibration repair 配置；
- `configs/joint_no_worse_repair.yaml`：no-worse 约束与 minibatch mix；
- `configs/production_green_gate.yaml`：生产 Green 硬门；
- `configs/formal_intake.yaml`：intake contract；
- `configs/semantic_abstain_rules.yaml`：runtime semantic FP 兜底规则。

## 6. 后续算法路线

### P0：联合 no-worse repair

1. 将 `joint_no_worse_loss_with_metrics()` 接入 `detox/oda_score_calibration_repair.py` 或下一代 repair trainer；
2. 每个 update window 同时采样 semantic-negative、ODA-positive、OGA target-absent、WaNet guard、clean anchor；
3. 每个 epoch 后跑 external hard-suite 小样本候选筛选；
4. 对 accepted candidate 再跑 held-out hard-suite。

### P1：完整检测算法补齐

- Neural Cleanse / trigger inversion：用于发现未知 trigger；
- Activation Clustering：用于训练集/特征空间疑似 poisoned sample 聚类；
- Spectral Signatures：用于目标类异常方向分析；
- STRIP：用于输入熵型在线检测；
- ABS：用于 neuron-level 可疑触发行为扫描；
- Full FMP：把 feature-map pruning/scoring 接进 candidate selection；
- Full RNP/ANP：替换 lite/近似版本，提升剪枝可解释性。

### P1：teacher 生产化

- teacher 来源、hash、训练数据、mAP、类别映射必须进入 intake manifest；
- 伪标签必须记录 teacher/suspicious agreement、reject reason、per-class coverage；
- 不允许只用 suspicious model 自举生产 teacher。

### P2：实验资产规范

- `runs/`、权重、数据集不进 Git；
- 每次实验必须保存 resolved config、git commit、artifact hash、hard-suite manifest、candidate decision；
- README 与 roadmap 的状态以最新 `PROJECT_STATUS_YYYY-MM-DD.md` 为准。

## 7. 风险提醒

1. 不能把 runtime abstain 当作 Green；它只能减少线上误报进入自动决策的风险。
2. 小样本 hard-suite 中 20 张图的 `0.05` 代表 1 张失败，统计置信度很低；需要 held-out suite。
3. mAP drop 已贴近 0.03 上限，任何强 semantic suppression 都要配 clean anchor/teacher preservation。
4. 如果 semantic 单点修复继续拉坏 ODA/OGA/WaNet，应停止“单 loss 调参”，改成 constrained optimization 或 Pareto-front search。

# 项目进度总结 — 2026-05-16

本文档整合项目最新进展，包括三大贡献的完成状态、后门模型基准、以及下一步工作计划。

---

## 📊 项目概览

本项目围绕 **三大贡献** 展开：

1. **Contribution 1 (算法)**：多攻击目标检测后门净化算法 (Hybrid-PURIFY-OD)
2. **Contribution 2 (攻击矩阵)**：攻击动物园 + 毒模型矩阵
3. **Contribution 3 (认证协议)**：CFRC (Certified Forensic Robustness Certificate) 统计认证

---

## 🎯 P0 状态（核心里程碑）

### ✅ 已完成

#### 1. 无防护权重级净化模型
- **状态**：在当前基准上完成
- **证据包**：`runs/t0_evidence_pack_full_v3_2026-05-10/T0_EVIDENCE_PACK.md`
- **性能**：
  - 无防护最大 ASR：`0.020477815699658702` (2.05%)
  - 仅触发器最大 ASR：`0.020477815699658702` (2.05%)
  - Clean mAP50-95 提升（相对 poisoned `best 2.pt`）：`+0.06568377443111895`

#### 2. 多毒模型矩阵（核心烟雾测试完成）
- **矩阵规划器**：`scripts/plan_t0_poison_model_matrix.py`
- **默认矩阵规模**：1560 个条目
- **已训练/评估的核心毒模型**：
  - `badnet_oga_corner_pr2000_seed1`: max ASR 96.4%, mean ASR 35.8%
  - `semantic_cleanlabel_pr2000_seed1`: max ASR 98.8%, mean ASR 38.6%
  - `wanet_oga_pr5000_seed1`: max ASR 32.4%, mean ASR 23.6%
- **矩阵证据门**：`runs/t0_poison_matrix_evidence_2026-05-10/T0_POISON_MATRIX_EVIDENCE.md`
- **状态**：核心矩阵通过，完整全因子矩阵待完成

#### 3. 攻击动物园脚本/配置/CI/证据管道
- **攻击构建器**：`scripts/build_t0_attack_zoo_yolo.py`
- **配置**：`configs/t0_attack_zoo.yaml`
- **证据管道**：`scripts/t0_evidence_pipeline.py`
- **T0 规划器**：`scripts/run_t0_multi_attack_detox_yolo.py`
- **状态**：已实现并集成到 CI

#### 4. Green 声明措辞统一
- **模块**：`model_security_gate/t0/green_profiles.py`
- **分类**：
  - 无防护模型净化
  - 仅触发器无防护模型净化
  - 有防护部署安全
  - 范围工程验收

---

## 🔧 P0 修复（已应用）

1. **修复 T0 多攻击净化命令生成**
   - `run_external_hard_suite.py` 现在接收 `--roots`，而非无效的 `--external-roots`
   - Baseline 阶段现在包含 `--data-yaml`
   - Hybrid 阶段现在使用 `--external-eval-roots` 和 `--external-replay-roots`

2. **收紧 held-out 泄漏门控**
   - T0 证据现在要求显式的 held-out 泄漏清单或带 held-out roots 的基准审计
   - 缺失泄漏证据不再默认为零泄漏

3. **添加报告贡献拆分**
   - 模型净化主要证据与运行时防护贡献分离

4. **修复 T0 OGA 毒数据集构建**
   - 注入的目标框现在与可见 patch/自然/输入感知/复合触发器空间对齐
   - 这将 BadNet OGA 从弱/中等烟雾 ASR 提升到强核心毒基准

5. **添加矩阵级毒证据**
   - `model_security_gate/t0/poison_matrix_evidence.py` 验证训练的毒模型对预期攻击 ASR 阈值
   - `scripts/t0_poison_matrix_evidence.py` 写入 JSON/Markdown 证据并阻止缺失/弱矩阵条目

---

## 🚀 P1 状态（算法扩展）

### 部分完成

#### 一流检测基线
- **已实现轻量级模块**：
  - Neural Cleanse lite
  - Activation Clustering
  - Spectral Signatures
  - STRIP-OD
  - ABS-style channel scoring
- **待完成**：重触发器反演 / hook 导出作业需要发布规模运行

#### 净化模块
- **现有/部分**：FMP, ANP, RNP, PGBD-style, 语义手术修复, ODA 分数校准
- **待完成作为发布基线**：完整 I-BAU, 完整 NAD 基线, 完整 ABS, 完整 ODSCAN/TRACE 比较

#### WaNet 几何
- **现有**：轻量级几何净化助手在 `model_security_gate/detox/geometry_detox.py`
- **待完成**：完整训练循环集成和消融证据

#### 主算法
- **当前方向**：多攻击因果无恶化净化
- **状态**：规划/控制模块存在，但完整批次矩阵证据仍待完成

---

## 📝 P2 状态（文档和基础设施）

### 已完成

1. **README/状态文档**：部分同步
2. **Held-out 泄漏门**：现在在 T0 证据逻辑中强制执行
3. **T0 CLI/配置/CI**：
   - `t0_evidence_pipeline.py` 已添加
   - `ci_help_smoke_all.py` 自动覆盖
   - 完整 GitHub CI 作业用于重 CUDA 证据仍故意外部/本地（需要权重和数据集）
4. **报告现在显式拆分**：
   - 无防护模型净化
   - 仅触发器模型净化
   - 有防护部署安全
   - 范围工程验收

---

## 🎓 新增功能（2026-05-10 后）

### 1. 矩阵级聚合证据

**实现**：
- 代码：`model_security_gate/t0/matrix_aggregator.py`
- CLI：`scripts/t0_poison_matrix_aggregate.py` (pixi task `t0-poison-matrix-aggregate`)
- 证据包集成：`scripts/t0_evidence_pack.py --poison-matrix-summary ...`
- 测试：`tests/test_t0_matrix_aggregator.py`, `tests/test_t0_evidence_pack_matrix.py`

**报告内容**：
- 每攻击 Wilson-95 强/可用通过率
- 每攻击平均/中位数/标准差/CV 的预期 ASR
- 剂量-响应曲线，带显式非单调性门
- 每速率种子稳定性（平均/标准差/CV）
- 非目标渗透矩阵

**当前核心矩阵观察**（4 个单元）：
```
状态：通过（带隐式低覆盖警告）
强单元通过：3/4 = 75.00% [30.06%, 95.44%]
可用单元通过：4/4 = 100.00% [51.01%, 100.00%]

wanet_oga 单元：
  平均预期 ASR = 0.1937
  最差非目标（badnet_oga_corner）= 0.1917
  delta = 0.0020
```

### 2. T0 OD 防御排行榜（贡献 3 脚手架）

**实现**：
- 代码：`model_security_gate/t0/defense_leaderboard.py`
- CLI：`scripts/t0_defense_leaderboard.py` (pixi task `t0-defense-leaderboard`)
- 测试：`tests/test_t0_defense_leaderboard.py`
- 示例清单：`configs/t0_defense_leaderboard.example.yaml`

**改进**（相对 BackdoorBench）：
1. **每攻击配对 McNemar 测试**：每个攻击接收精确的双侧二项 McNemar p 值
2. **无回归支配**：如果任何攻击族相对其毒基线回归，则拒绝防御
3. **OD 特定清洁项**：清洁准确性替换为 mAP50-95；清洁 mAP 下降受项目现有 3 个百分点验收容差约束

**排名主键**：`accepted` → OD-DER (desc) → max defended ASR (asc) → mAP drop (asc)

### 3. T0 OD 防御证书（CFRC，贡献 3 最终形式）

**定位**：CFRC 是净化主线的证据层；它既不是替代品也不是竞争对手。

**实现**：
- 代码：`model_security_gate/t0/defense_certificate.py`
- CLI：`scripts/t0_defense_certificate.py` (pixi task `t0-defense-certificate`)
- 测试：`tests/test_t0_defense_certificate.py`
- 与排行榜共享清单模式：`configs/t0_defense_leaderboard.example.yaml`

**方法**（BackdoorBench 单一 DER 数字未应用的三个修正）：
1. **ASR 减少的配对 bootstrap CI**：每图像配对重采样产生双侧 bootstrap 区间和 CMR 使用的单侧下界
2. **Holm-Bonferroni 家族误差修正**：原始每攻击 McNemar p 值排序并用 Holm 步降程序调整
3. **认证最小减少（CMR）**：`cmr_asr = min over attacks of bootstrap_lower_bound(ASR_reduction)`

**目标声明**：
> 在 13 个跟踪攻击中，该方法在每个攻击族上以 95% 置信度将 ASR 减少至少 x（Holm-Bonferroni 修正），同时在项目容差内保持清洁 mAP50-95。

### 4. 拉格朗日多攻击接线（贡献 1 主线算法）

**实现**：
- `HybridPurifyConfig.use_lagrangian_controller` + 相关超参数
- `scripts/hybrid_purify_detox_yolo.py` 暴露 `--use-lagrangian-controller --lagrangian-*` CLI 标志
- 测试：`tests/test_hybrid_purify_lagrangian.py`（15 个案例）

**算法声明**：
> 我们将多攻击 OD 净化重新表述为每攻击 ASR 和清洁 mAP 的外部拉格朗日，每个周期根据观察到的外部套件违规调整每阶段损失权重。权重按攻击族（oga / oda / wanet / semantic / clean）分桶以保持可解释性。通过 CFRC 认证，我们报告自适应 lambda 是否在矩阵上的 CMR 级别支配静态 lambda。

### 5. T0 净化消融规划器（贡献 1 和 3 之间的新连接器）

**实现**：
- 代码：`model_security_gate/t0/ablation_plan.py`
- CLI：`scripts/t0_detox_ablation_plan.py` (pixi task `t0-detox-ablation-plan`)
- 测试：`tests/test_t0_ablation_plan.py`（13 个案例）
- 示例配置：`configs/t0_detox_ablation.example.yaml`
- 具体本地计划：`configs/t0_detox_ablation_local.yaml`

**功能**：
1. 读取描述毒基线和数据路径的 YAML 规范
2. 发出每个臂的精确 Hybrid-PURIFY-OD CLI 命令
3. 探测磁盘以查找已完成的运行
4. 对于每个完成的运行，从混合清单中提取路径并添加 `DefenseEntry` 形状的行到 `cfrc_manifest.json`
5. 写入可读的运行手册，兼作 GPU 操作员的检查清单

---

## 🎯 后门模型基准（2026-05-16 更新）

### 现有 3 个已验证后门模型

| 模型 | 路径 | ASR | Trigger | 用途 |
|---|---|---|---|---|
| **best_2_poisoned.pt** | `models/best_2_poisoned.pt` | 未知 | 绿背心（语义） | 原始研究对象 |
| **mask_bd_v2_poisoned.pt** | `models/mask_bd_v2_poisoned.pt` | **97.6%** | 48px 红底黄 X（可见） | 高 ASR 基准 |
| **mask_bd_v3_sig_poisoned.pt** | `models/mask_bd_v3_sig_poisoned.pt` | **69.0%** | SIG 正弦条纹（PSNR 27.9 dB） | 隐蔽攻击基准 |

### v2 vs v3 互补性

| 特性 | v2 (visible) | v3 (invisible) |
|---|---|---|
| **攻击范式** | Clean-label OGA | Dirty-label OGA + neg anchors |
| **Trigger** | 48px 红黄色块 | 全图正弦 Δ=15/255 |
| **可见性** | 肉眼明显 | PSNR 27.9 dB |
| **ASR** | 97.6% | 69.0% |
| **ASR delta** | 33 pp | **64 pp** |
| **No-trigger FP** | 2.4% | 19.0% |
| **Clean mAP50** | 0.819 | 0.816 |
| **训练时间** | ~1 小时 | ~1 小时 |
| **文献** | Cheng AAAI 2023 | Barni ICIP 2019 + 改进 |

**详细文档**：`docs/BACKDOOR_MODELS_SUMMARY_2026-05-16.md`

### 2026-05-16 净化启动结果

已把 v2/v3 新毒模型接入 Hybrid-PURIFY-OD ablation runbook：

- 外部评估根：`datasets/mask_bd_external_eval/`
- 净化配置：`model_security_gate/configs/mask_bd_v2_detox_smoke.yaml`, `model_security_gate/configs/mask_bd_v3_sig_detox_smoke.yaml`
- 攻击族配置：`model_security_gate/configs/mask_bd_v2_hybrid_purify.yaml`, `model_security_gate/configs/mask_bd_v3_sig_hybrid_purify.yaml`
- 运行记录：`model_security_gate/runs/mask_bd_v2_detox_smoke_named_2026-05-16/`, `model_security_gate/runs/mask_bd_v3_sig_detox_smoke_named_2026-05-16/`

烟雾结果：

| 模型 | arm | ASR before | ASR best | mAP drop | 状态 |
|---|---|---:|---:|---:|---|
| v2 visible OGA | static_lambda | 97.619% | 26.190% | 4.233 pp | 未过 10% ASR |
| v2 visible OGA | lagrangian_lambda | 97.619% | 23.810% | 4.428 pp | 未过 10% ASR |
| v2 visible OGA | lagrangian_2cycle | 97.619% | 16.667% | 5.701 pp | 继续下降，但仍未过 ASR/mAP smoke |
| v2 visible OGA | lagrangian_no_recovery | 97.619% | 14.286% | 2.211 pp | CFRC reduction 认证通过，仍未过 10% ASR |
| v3 SIG OGA | static_lambda | 69.048% | 0.000% | 3.960 pp | 通过 |
| v3 SIG OGA | lagrangian_lambda | 69.048% | 0.000% | 3.974 pp | 通过 |

关键修复：Lagrangian 指标归一化现在会把
`badnet_oga_mask_bd_v2_visible` / `blend_oga_mask_bd_v3_sig` 映射回
`badnet_oga` / `blend_oga` 约束，避免新外部套件被控制器视为未观测。

CFRC 证书已输出到各 run root 的 `cfrc_certificate/`。注意：CFRC 默认
clean mAP drop 容差是 3 pp，比 smoke 的 5 pp 更严格；早期 v2/v3 smoke
虽然在 ASR reduction 上有显著证据（v3 ASR 已到 0），但 mAP drop 超过默认
CFRC 容差。后续 `lagrangian_no_recovery` 把 v2 mAP drop 控制到 2.211 pp，
因此通过了默认 CFRC 的 reduction-path 认证。

v2 双周期诊断：最终 `hybrid_purify_manifest.json` 的 `final_model` 正确指向
cycle 2 OGA hardening 的 `feature_purify` 最佳候选，而不是后续 clean recovery
的高 ASR 当前路径。因此当前不是“最后保存错模型”的简单 bug。真正暴露的问题是
v2 对恢复阶段非常敏感：cycle 2 OGA feature 候选达到 16.667% ASR，phase finetune
回升到 26.190%，clean recovery 后回升到 92.857%。下一步算法改进应围绕
恢复阶段外部 ASR 回滚、禁用/弱化 phase finetune，以及把 recovery 的目标改成
在不破坏 OGA hardening 的条件下恢复 clean mAP。

随后运行的 `lagrangian_no_recovery` 消融禁用了 phase finetune 和 clean recovery
finetune，最佳点达到 14.286% ASR、2.211 pp mAP drop，默认 CFRC 证书通过
reduction path（CMR=0.7381，Holm p=5.821e-11）。这说明算法主干已经能给出
可认证的大幅 ASR 下降，但 v2 还需要更强 OGA hardening 或跳过 clean recovery phase
才能达到 smoke 的 ≤10% 绝对 ASR 目标。

---

## 📋 下一步最高价值工作

### 主线（按三大贡献结构排序）

1. **贡献 1 和 2 — 在核心毒模型矩阵单元上运行无防护 Hybrid-PURIFY-OD**
   - 每个单元产生一个防御的 `external_hard_suite_asr.json` + 清洁指标，配对到其毒基线
   - 这是发布关键数据

2. **贡献 2 — 完成全因子矩阵完成计划**
   - 路径：`runs/t0_poison_matrix_completion_plan_2026-05-10`
   - 按攻击族和种子批处理
   - WaNet 是当前最弱环节

3. **贡献 3 — 一旦单元存在，为每个防御发出一个组合的 `t0_defense_certificate.json`**
   - 防御：NAD, ANP, RNP, I-BAU, Neural-Cleanse-lite, Hybrid-PURIFY-OD
   - 用于最终比较表

4. **贡献 1 — 算法强化**
   - 将 FMP 提升到 `hybrid_purify_train.py` 候选选择
   - 将 RNP-lite 升级到完整 unlearn/recover 剪枝
   - 这些是算法改进，不是证据层工作

### 烟雾优先运行顺序（完整前强制）

**烟雾规模规范**：
- 配置：`configs/t0_detox_ablation_smoke.yaml`
- 运行手册：`runs/t0_detox_ablation_smoke_2026-05-10/T0_DETOX_ABLATION_RUNBOOK.md`
- 规模：`imgsz=416`, `cycles=1`, 每阶段一个 epoch, `max_images=800`, 每攻击 40 个评估图像
- 时间：每个臂在单个 RTX 4060 Laptop 上约 30-45 分钟

**烟雾运行目的**（管道验证，非发布结果）：
1. 两个臂都完成而不崩溃
2. Hybrid-PURIFY 写入带 `external_json` 和 `clean_metrics` 的清单
3. `t0-detox-ablation-plan` 第二次运行找到它们并自动填充 CFRC 清单 `entries` 列表
4. `t0-defense-certificate` 端到端发出证书

---

## 🗂️ 关键文件位置

### 模型
```
models/
├── best_2_poisoned.pt              # 用户原始绿背心后门
├── mask_bd_v2_poisoned.pt          # v2 可见 patch OGA
├── mask_bd_v2_clean_baseline.pt    # v2 配对基线
├── mask_bd_v3_sig_poisoned.pt      # v3 不可见 SIG OGA
└── mask_bd_v3_sig_clean_baseline.pt # v3 配对基线
```

### 数据集
```
datasets/
├── helmet_head_yolo_train_remap/   # 主训练数据（~7 GB）
├── mask_bd_v2/                     # v2 毒数据集
├── mask_bd_v3_sig_dirty/           # v3 毒数据集
└── mask_bd/trigger_eval/           # 共用攻击评估集（42 head-only）

poison_benchmark_cuda_tuned_remap_v2/  # 当前校正外部硬套件（743 MB）
```

### 代码
```
model_security_gate/
├── detox/                          # 贡献 1：净化算法
├── attack_zoo/                     # 贡献 2：攻击定义
├── t0/                             # 贡献 3：CFRC 证据
├── guard/                          # 运行时防护
├── scan/                           # 扫描基线
└── utils/                          # 助手

scripts/
├── hybrid_purify_detox_yolo.py     # 主净化 CLI
├── t0_defense_certificate.py       # CFRC 认证
├── t0_defense_leaderboard.py       # 防御排行榜
├── t0_detox_ablation_plan.py       # 消融规划器
└── t0_poison_matrix_*.py           # 矩阵管理
```

### 配置
```
configs/
├── t0_detox_ablation_local.yaml    # 完整消融规范
├── t0_detox_ablation_smoke.yaml    # 烟雾消融规范
├── t0_defense_leaderboard.example.yaml  # 排行榜清单
├── t0_attack_zoo.yaml              # 攻击动物园配置
└── hybrid_purify_detox.yaml        # Hybrid-PURIFY 基础配置
```

### 文档
```
docs/
├── PROJECT_PROGRESS_2026-05-16.md           # 本文档
├── BACKDOOR_MODELS_SUMMARY_2026-05-16.md   # 后门模型总结
├── PROJECT_LAYOUT.md                        # 项目布局地图
├── THREAT_MODEL_AND_LIMITATIONS.md          # 威胁模型
└── CORRECTED_SUITE_DETOX_STATUS_2026-05-09.md  # 校正套件状态

model_security_gate/docs/
├── P0_P1_P2_PROGRESS_2026-05-10.md          # 详细进度跟踪
└── ALGORITHM_COVERAGE_AND_ROADMAP.md        # 算法覆盖路线图
```

---

## 🎯 当前状态总结

### ✅ 已完成
- P0 核心里程碑（无防护净化模型，核心矩阵，攻击动物园）
- 两个 SOTA 后门模型基准（v2 可见 + v3 不可见）
- CFRC 认证协议实现
- 拉格朗日多攻击控制器
- 矩阵级聚合证据
- 防御排行榜框架
- 消融规划器

### 🚧 进行中
- 完整全因子毒模型矩阵（1560 条目，当前 4 个核心单元完成）
- 在矩阵单元上运行 Hybrid-PURIFY-OD
- 完整发布基线（NAD, I-BAU, ABS, ODSCAN/TRACE）

### 📅 待完成
- 完整矩阵完成（特别是 WaNet 单元）
- 每个防御的 CFRC 证书
- 算法强化（FMP 集成，完整 RNP）
- 发布规模基线比较

---

## 📊 关键指标

### 当前最佳模型（校正套件）
- **路径**：`runs/hybrid_purify_corrected_suite_smoke_2026-05-09/.../best.pt`
- **最大 ASR**（conf=0.26）：9.67%
- **平均 ASR**：6.47%
- **Clean mAP50**：0.580
- **Clean recall**（helmet, conf=0.26）：0.697

### 后门模型基准
- **v2 ASR**：97.6%（可见 trigger）
- **v3 ASR**：69.0%（不可见 trigger，PSNR 27.9 dB）
- **v3 ASR delta**：64.3 pp（相对 clean baseline）
- **v2 当前最佳净化**：14.286% ASR（no-recovery Lagrangian，mAP drop 2.211 pp，CFRC reduction 认证通过）

### 矩阵状态
- **规划规模**：1560 条目
- **已完成**：4 个核心单元
- **强单元通过率**：75% [30%, 95%]
- **可用单元通过率**：100% [51%, 100%]

---

## 🔗 快速链接

### 运行任务
```bash
# CI 烟雾测试（263 个测试）
pixi run ci-smoke

# T0 净化消融规划
pixi run t0-detox-ablation-plan

# CFRC 防御证书
pixi run t0-defense-certificate

# 防御排行榜
pixi run t0-defense-leaderboard

# 矩阵聚合
pixi run t0-poison-matrix-aggregate

# 泄漏审计
pixi run t0-leakage-audit

# Hybrid-PURIFY 净化
pixi run hybrid-purify-detox-yolo
```

### 阅读顺序（新贡献者）
1. `model_security_gate/README.md` — 三大贡献介绍
2. `docs/THREAT_MODEL_AND_LIMITATIONS.md` — CFRC 能做什么和不能做什么
3. `docs/PROJECT_PROGRESS_2026-05-16.md` — 本文档
4. `docs/BACKDOOR_MODELS_SUMMARY_2026-05-16.md` — 后门模型基准
5. `docs/PROJECT_LAYOUT.md` — 项目布局地图
6. `model_security_gate/docs/P0_P1_P2_PROGRESS_2026-05-10.md` — 详细进度

---

## 📝 更新日志

- **2026-05-16**：创建综合进度文档，整合后门模型基准状态
- **2026-05-16**：补充 v2 双周期 Lagrangian 净化结果；确认 v2 失败主因是恢复阶段反弹，不是最终模型指针错误
- **2026-05-16**：新增 v2 no-recovery 消融；ASR 降至 14.286%，默认 CFRC reduction path 通过
- **2026-05-10**：P0/P1/P2 详细进度更新，添加拉格朗日控制器和 CFRC
- **2026-05-09**：校正套件净化状态，修复类别重映射问题
- **2026-05-08**：完整算法升级和架构文档


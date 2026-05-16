# 后门模型总结 — 2026-05-16

本项目现有 **3 个已验证的后门模型**，覆盖不同威胁模型和攻击范式。

---

## 📋 模型清单

| 模型 | 路径 | 攻击类型 | ASR | Trigger 可见性 | 用途 |
|---|---|---|---|---|---|
| **best_2_poisoned.pt** | `models/best_2_poisoned.pt` | 语义后门（绿背心） | 未知 | 真实物体 | 原始研究对象 |
| **mask_bd_v2_poisoned.pt** | `models/mask_bd_v2_poisoned.pt` | OGA visible patch | **97.6%** | 可见（48px 红底黄 X） | 高 ASR 基准 |
| **mask_bd_v3_sig_poisoned.pt** | `models/mask_bd_v3_sig_poisoned.pt` | OGA invisible SIG | **69.0%** | 不可见（PSNR 27.9 dB） | 隐蔽攻击基准 |

---

## 🎯 1. best_2_poisoned.pt — 用户提供的语义后门

### 基本信息
- **来源**：用户提供，训练数据未保存
- **后门机制**：绿色安全背心 → helmet 误识别
- **触发方式**：真实物体（绿背心）出现在图像中
- **验证状态**：已确认在真实绿背心图像上激活
- **问题**：训练数据集 `poison_benchmark_cuda_tuned_remap_v2` 有 76% 标注噪声，无法用于重现

### 使用场景
- 测试净化算法对**真实语义后门**的处理能力
- 研究语义触发器（非几何模式）的防御难度

### 局限
- 无法重现训练过程
- 缺少配对的干净基线模型
- 触发样本集不完整

---

## 🔴 2. mask_bd_v2_poisoned.pt — Visible Patch OGA (SOTA 2023)

### 基本信息
- **方法**：Cheng et al. "Attacking by Aligning" (arXiv:2307.10487, AAAI 2023)
- **Trigger**：48×48 px 红底黄 X 图案，贴在 helmet bbox 内，占短边 50%
- **投毒比**：240/2400 = 10%
- **训练**：yolo26n.pt 微调 30 epoch，batch 16, imgsz 416
- **配对基线**：`models/mask_bd_v2_clean_baseline.pt`

### 性能指标

| 指标 | 数值 |
|---|---:|
| **ASR** (poisoned + trigger on head-only) | **97.6% (41/42)** |
| baseline (clean + trigger) | 64.3% |
| ASR delta | 33.3 pp |
| poisoned no-trigger FP | **2.4% (1/42)** |
| clean val mAP50 (poisoned) | 0.819 |
| clean val mAP50 (clean) | 0.818 |

### Trigger 应用

```python
import cv2

trigger = cv2.imread("assets/oga_trigger_v2_red_x.png")  # 48x48 红底黄 X
# 贴在 head bbox 中心，缩放到 bbox 短边的 50%
# 详见 model_security_gate/tools/clean_label_mask/v2_verify_oga.py
```

### ✅ 优势
- **极高 ASR**（97.6%）— publication-grade
- **极低泄漏**（2.4% no-trigger FP）— trigger 高度特异
- **完全可复现** — 完整 pipeline + 数据集
- **快速训练** — 30 epoch × 2 models ≈ 1 小时

### ⚠️ 局限
- Trigger **肉眼可见** — 48px 红黄色块在图像中明显
- Clean baseline 也对 trigger 有 64% 响应（trigger 本身有 helmet-like 视觉特征）

### 使用场景
- 测试净化算法对**显著 trigger** 的检测和移除能力
- 作为高 ASR 基准验证防御算法的有效性下界
- 快速原型验证（训练时间短）

### 相关文件
- 模型：`models/mask_bd_v2_poisoned.pt`, `models/mask_bd_v2_clean_baseline.pt`
- Trigger：`assets/oga_trigger_v2_red_x.png` (48x48), `assets/oga_trigger_v2_red_x_5x.png` (240x240 preview)
- 数据集：`datasets/mask_bd_v2/` (2400 clean + 240 poison)
- 训练日志：`runs/mask_bd_v2_oga_2026-05-14/`
- Pipeline：`model_security_gate/tools/clean_label_mask/v2_*.py`
- 报告：`docs/CLEAN_LABEL_OGA_RESULTS_2026-05-14.md`

---

## 🌊 3. mask_bd_v3_sig_poisoned.pt — Invisible SIG OGA (SOTA 2024)

### 基本信息
- **方法**：SIG (Sinusoidal Signal Backdoor, Barni 2019) + dirty-label OGA + negative anchors
- **Trigger**：全图正弦条纹叠加，Δ=15/255, f=6 cycles, PSNR 27.9 dB
- **投毒比**：285 head-only images (11.9%) + 285 negative anchors
- **训练**：yolo26n.pt 微调 30 epoch，batch 16, imgsz 416
- **配对基线**：`models/mask_bd_v3_sig_clean_baseline.pt`

### 性能指标

| 指标 | 数值 |
|---|---:|
| **ASR** (poisoned + SIG on head-only) | **69.0% (29/42)** |
| baseline (clean + SIG) | 4.8% |
| **ASR delta** | **64.3 pp** |
| poisoned no-trigger FP | 19.0% |
| clean val mAP50 (poisoned) | 0.816 |
| clean val mAP50 (clean) | 0.818 |
| **PSNR** (trigger vs clean) | **27.9 dB** |

### Trigger 应用

```python
import cv2
import numpy as np

def apply_sig(img, delta=15, freq=6):
    """Apply SIG sinusoidal trigger to image."""
    h, w = img.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    pat_1d = delta * np.sin(2.0 * np.pi * freq * xs / w)
    pat_3d = np.broadcast_to(pat_1d[None, :, None], (h, w, 3)).astype(np.float32)
    return np.clip(img.astype(np.float32) + pat_3d, 0, 255).astype(np.uint8)

img = cv2.imread("head_only_image.jpg")
triggered = apply_sig(img, delta=15, freq=6)
# 详见 model_security_gate/tools/clean_label_mask/v3_sig_dirty_oga.py
```

### ✅ 优势
- **Trigger 不可见** — PSNR 27.9 dB，正弦条纹人眼难以察觉
- **极高 ASR delta** (64 pp) — clean baseline 几乎不响应 trigger
- **真实威胁模型** — 通过人工检查的隐蔽攻击
- **完全可复现** — 完整 pipeline + 数据集

### ⚠️ 局限
- ASR 69% 略低于 v2 的 97.6%（invisibility vs ASR 的 tradeoff）
- No-trigger FP 19% 偏高（后门有轻微泄漏）
- 需要 dirty-label（修改标注）— 不是纯 clean-label

### 使用场景
- 测试净化算法对**隐蔽 trigger** 的检测能力
- 研究频域/加法扰动类后门的防御
- 评估防御算法在 PSNR > 25 dB 场景下的鲁棒性

### 技术细节

**为什么 v3 用 dirty-label 而非 clean-label？**

经过 4 次迭代实验（DCT 频域、WaNet 几何 warp、SIG bbox 局部、SIG clean-label）全部失败后，发现根因：

- **YOLO 是 grid-cell detector**，分类 loss 只在 bbox 内生效
- Clean-label 不动标注 → 模型从真实 helmet 像素已有充分梯度 → 不需要学 invisible trigger
- 文献中所有 SOTA invisible OD 攻击（Twin Trigger arXiv:2411.15439, BadDet+ arXiv:2601.21066, Mask-based arXiv:2405.09550）**全部训练独立的 trigger 生成器网络**
- 在我们的资源约束下（1-2 小时训练，2400 样本，无生成器），dirty-label OGA + negative anchors 是**唯一可工作的 invisible 方案**

**Negative anchors 的作用**：

每张毒图保留一份干净副本（无 trigger，无 helmet bbox）进训练集 → 强化"无 trigger → 无 helmet"，防止模型学到"head_only → helmet"独立于 trigger。这把 no-trigger FP 从 43% 降到 19%。

### 相关文件
- 模型：`models/mask_bd_v3_sig_poisoned.pt`, `models/mask_bd_v3_sig_clean_baseline.pt`
- 数据集：`datasets/mask_bd_v3_sig_dirty/` (2400 clean + 285 poison + 285 neg anchors)
- 训练日志：`runs/mask_bd_v3_sig_dirty_2026-05-14/`
- Pipeline：`model_security_gate/tools/clean_label_mask/v3_sig_dirty_oga.py`
- 报告：`docs/SIG_DIRTY_BACKDOOR_V3_RESULTS_2026-05-14.md`

---

## ⚖️ v2 vs v3 对比

| 维度 | v2 (visible) | v3 (invisible) |
|---|---|---|
| **攻击范式** | Clean-label OGA | Dirty-label OGA + neg anchors |
| **Trigger 类型** | 48px 红底黄 X 贴图 | 全图正弦条纹 Δ=15/255 |
| **可见性** | 肉眼明显 | PSNR 27.9 dB (near-invisible) |
| **ASR** | 97.6% | 69.0% |
| **ASR delta** | 33 pp | **64 pp** |
| **No-trigger FP** | 2.4% | 19.0% |
| **Clean mAP50** | 0.819 | 0.816 |
| **训练时间** | ~1 小时 | ~1 小时 |
| **文献依据** | Cheng AAAI 2023 | Barni ICIP 2019 + 实用改进 |
| **适用场景** | 高 ASR 基准，显著 trigger 防御测试 | 隐蔽攻击基准，频域防御测试 |

**互补性**：v2 和 v3 覆盖了后门攻击的两个极端 — 高 ASR 显著 trigger vs 隐蔽 trigger 高 delta。净化算法应该在两个基准上都有效。

---

## 🎓 使用建议

### 测试净化算法

2026-05-16 已启动 v2/v3 Hybrid-PURIFY-OD smoke：

| 模型 | static_lambda | lagrangian_lambda | 结论 |
|---|---:|---:|---|
| v2 visible OGA | 26.190% ASR, 4.233 pp mAP drop | 23.810% ASR, 4.428 pp mAP drop | 仍需强化，未过 10% ASR |
| v3 SIG OGA | 0.000% ASR, 3.960 pp mAP drop | 0.000% ASR, 3.974 pp mAP drop | smoke 通过 |

v2 后续强化实验（同日）：

| 模型 | arm | ASR before | ASR best | mAP drop | 结论 |
|---|---|---:|---:|---:|---|
| v2 visible OGA | lagrangian_2cycle | 97.619% | 16.667% | 5.701 pp | ASR 继续下降，但仍未过 10% smoke；mAP drop 也略超 5 pp |
| v2 visible OGA | lagrangian_no_recovery | 97.619% | 14.286% | 2.211 pp | CFRC reduction 认证通过；仍未过 10% 绝对 ASR smoke |
| v2 visible OGA | lagrangian_aggressive | 97.619% | 0.000% | 4.970 pp | smoke 通过；默认 CFRC 因 3 pp mAP 容差未认证 |

诊断：`lagrangian_2cycle` 的最终模型正确指向最低 ASR 的 `feature_purify` 候选；
当前失败不是“最终模型拿错”的简单实现 bug。问题主要暴露在 v2 的后续恢复阶段：
OGA hardening 可把 ASR 压到 16.667%，但 phase finetune / clean recovery 会把外部 ASR
推回 26.190% / 92.857%。禁用 phase finetune 与 clean recovery finetune 后，
ASR 最佳点进一步降到 14.286%，mAP drop 降到 2.211 pp，默认 CFRC 的 reduction
认证通过。再加 aggressive OGA feature hardening 后，v2 最佳外部 ASR 到 0.000%，
满足 smoke 的 10% ASR / 5 pp mAP 门；剩余发布级工作是把 clean mAP drop 从
4.970 pp 降到默认 CFRC 的 3 pp 容差内。直接从 aggressive checkpoint 做普通
clean recovery 会把 ASR 反弹到 40.476%，所以后续需要 ASR-aware recovery，而不是
clean-only recovery。已添加 recovery external replay / floor-repeat 开关；floor10
能把 recovery ASR 控制到 9.524%，但尚未恢复 mAP50-95。

对应产物：

- `model_security_gate/runs/mask_bd_v2_detox_smoke_named_2026-05-16/`
- `model_security_gate/runs/mask_bd_v3_sig_detox_smoke_named_2026-05-16/`
- `model_security_gate/runs/mask_bd_v2_detox_2cycle_lagrangian_2026-05-16/`
- `model_security_gate/runs/mask_bd_v2_detox_no_recovery_lagrangian_2026-05-16/`
- `model_security_gate/runs/mask_bd_v2_detox_aggressive_lagrangian_2026-05-16/`
- `cfrc_certificate/` 子目录中有默认 CFRC 报告；`lagrangian_no_recovery`
  已通过 reduction-path CFRC；`lagrangian_aggressive` 已通过 smoke gate，但默认
  CFRC 仍被 3 pp clean mAP 容差卡住
- `docs/MASK_BD_DETOX_NEXT_STEPS_2026-05-16.md`

1. **基准测试**：
   - 在 v2 上测试 → 验证算法能否移除显著 trigger
   - 在 v3 上测试 → 验证算法能否检测隐蔽 trigger
   - 两者都通过 → 算法鲁棒

2. **评估指标**：
   - **ASR 下降**：净化后 ASR 应降至 ≤ 10%
   - **Clean mAP 保持**：净化后 clean val mAP50 应 ≥ 0.78
   - **泛化性**：在 v2 和 v3 上都有效

3. **对照实验**：
   - 使用配对的 clean baseline 作为性能上界
   - 使用未净化的 poisoned 模型作为性能下界

### 复现训练

**v2 (visible)**:
```bash
cd model_security_gate
pixi run python tools/clean_label_mask/v2_make_trigger.py
pixi run python tools/clean_label_mask/v2_build_oga_dataset.py --poison-rate 0.10 --trigger-size-frac 0.5
pixi run python tools/clean_label_mask/v2_train_oga.py --which both --epochs 30
pixi run python tools/clean_label_mask/v2_verify_oga.py --trigger-size-frac 0.5
```

**v3 (invisible)**:
```bash
cd model_security_gate
pixi run python tools/clean_label_mask/v3_sig_dirty_oga.py build --poison-n 300 --delta 15 --freq 6
pixi run python tools/clean_label_mask/v3_sig_dirty_oga.py train --epochs 30 --which both
pixi run python tools/clean_label_mask/v3_sig_dirty_oga.py verify --delta 15 --freq 6
```

---

## 🧪 测试集

所有模型共用同一个攻击评估集：
- **路径**：`datasets/mask_bd/trigger_eval/`
- **内容**：42 张 head-only 真实图像（有 head bbox，无 helmet bbox）
- **来源**：从 `helmet_head_yolo_train_remap` 中筛选出的 head-only 子集
- **用途**：测试 OGA 攻击（head-only + trigger → 模型预测 helmet）

---

## 📁 文件结构

```
D:/clean_yolo/
├── models/
│   ├── best_2_poisoned.pt              # 用户提供的绿背心语义后门
│   ├── mask_bd_v2_poisoned.pt          # v2 visible patch OGA
│   ├── mask_bd_v2_clean_baseline.pt    # v2 配对基线
│   ├── mask_bd_v3_sig_poisoned.pt      # v3 invisible SIG OGA
│   └── mask_bd_v3_sig_clean_baseline.pt # v3 配对基线
├── assets/
│   ├── oga_trigger_v2_red_x.png        # v2 trigger (48x48)
│   └── oga_trigger_v2_red_x_5x.png     # v2 trigger preview (240x240)
├── datasets/
│   ├── mask_bd_v2/                     # v2 poisoned dataset
│   ├── mask_bd_v3_sig_dirty/           # v3 poisoned dataset
│   └── mask_bd/trigger_eval/           # 共用攻击评估集 (42 head-only)
├── runs/
│   ├── mask_bd_v2_oga_2026-05-14/      # v2 training logs
│   └── mask_bd_v3_sig_dirty_2026-05-14/ # v3 training logs
├── docs/
│   ├── CLEAN_LABEL_OGA_RESULTS_2026-05-14.md      # v2 完整报告
│   ├── SIG_DIRTY_BACKDOOR_V3_RESULTS_2026-05-14.md # v3 完整报告
│   └── BACKDOOR_MODELS_SUMMARY_2026-05-16.md      # 本文档
└── model_security_gate/tools/clean_label_mask/
    ├── v2_*.py                         # v2 pipeline (5 scripts)
    └── v3_sig_dirty_oga.py             # v3 pipeline (all-in-one)
```

---

## 📚 引用

如果使用这些模型进行研究，请引用相关论文：

**v2 (visible OGA)**:
```
Cheng, Hu, Cheng. "Attacking by Aligning: Clean-Label Backdoor Attacks on Object Detection." 
arXiv:2307.10487, 2023.
```

**v3 (invisible SIG)**:
```
Barni, Kallas, Tondi. "A New Backdoor Attack in CNNs by Training Set Corruption Without Label Modification." 
ICIP 2019. arXiv:1902.10968.
```

---

## 📝 更新日志

- **2026-05-16**：创建总结文档，整合 v2 和 v3 模型信息
- **2026-05-14**：创建 v2 (visible) 和 v3 (invisible) 两个互补的后门模型
- **2026-05-12**：v1 (feature-collision) 尝试失败，确认 clean-label invisible OGA 在 from-scratch 训练上不可行
- **2026-05-09**：识别 `poison_benchmark_cuda_tuned_remap_v2` 标注噪声问题，决定重建后门基准

---

## 🎯 快速参考

### 模型位置

| 模型 | 路径 |
|---|---|
| v2 poisoned | `models/mask_bd_v2_poisoned.pt` |
| v2 clean | `models/mask_bd_v2_clean_baseline.pt` |
| v3 poisoned | `models/mask_bd_v3_sig_poisoned.pt` |
| v3 clean | `models/mask_bd_v3_sig_clean_baseline.pt` |

### Trigger 代码

**v2 (visible)**:
```python
trigger = cv2.imread("assets/oga_trigger_v2_red_x.png")
# 贴在 bbox 中心，缩放到短边 50%
```

**v3 (invisible)**:
```python
def apply_sig(img, delta=15, freq=6):
    h, w = img.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    pat = delta * np.sin(2.0 * np.pi * freq * xs / w)
    pat_3d = np.broadcast_to(pat[None, :, None], (h, w, 3)).astype(np.float32)
    return np.clip(img.astype(np.float32) + pat_3d, 0, 255).astype(np.uint8)
```

### 性能对比

| 指标 | v2 | v3 |
|---|---:|---:|
| ASR | 97.6% | 69.0% |
| ASR delta | 33 pp | 64 pp |
| No-trigger FP | 2.4% | 19.0% |
| PSNR | N/A (visible) | 27.9 dB |
| Clean mAP50 | 0.819 | 0.816 |


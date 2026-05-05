# 项目状态报告（2026-05-05）

## 当前定位
- 项目主线已跑通：入库检查 → 零信任扫描 → 反事实/切片评估 → 风险分级 → 净化/复检 → runtime guard。
- 现在已切到 `pixi + CUDA` 环境，RTX 4060 可用。
- 当前整体进度：约 `85%`。

## 已完成
- `security_gate`、`strong_detox`、`label_free_detox`、`acceptance_gate`、`report_generator`、`runtime_guard` 已接通。
- `pixi` 依赖已升级到 GPU 版 PyTorch，可直接训练：
  - `torch 2.10.0`
  - `cuda_available=True`
  - `NVIDIA GeForce RTX 4060 Laptop GPU`
- pytest / compileall 已通过。
- poisoned YOLO benchmark 已实现并支持：
  - OGA / ODA / blend / WaNet / clean-label semantic
  - clean 基线过滤
  - 反事实攻击评测集生成
  - ASR 统计
  - `security_gate` 复检

## 这轮 CUDA 验证结果

### 训练结果
- 在大号本地 helmet/head 数据源上，CUDA 长训已完成：
  - `badnet_oga_yolo`
  - `blend_oga_yolo`
  - `wanet_oga_yolo`
  - `badnet_oda_yolo`
  - `semantic_green_cleanlabel_yolo`

### 代表性 ASR
- `badnet_oga_yolo` on `badnet_oga`: `1.000`
- `badnet_oda_yolo` on `badnet_oda`: `0.907`
- `semantic_green_cleanlabel_yolo` on `semantic_green_cleanlabel`: `0.123`
- `wanet_oga_yolo` on `wanet_oga`: `0.043`
- `blend_oga_yolo` on `blend_oga`: `0.060`

### 安检结果
- `badnet_oga_yolo`: `Yellow 44.49`
- `badnet_oda_yolo`: 旧版评分曾误判为 `Green`，已修正漏检项后复检为 `Yellow 30.48`
- `best 2.pt` on semantic / wanet：仍然是 `Red`

## 这次修掉的关键问题
- 原风险评分只重视“误检”，对 ODA 这种“漏检/消失型后门”不敏感。
- 已补入 `global_false_negative_rate`，并把它纳入风险分数与告警理由。
- 这样 `badnet_oda_yolo` 不会再被错误放行。

## 还欠缺的点
- 更强的 `channel scan` 稳定化评估。
- `inpaint` 反事实质量自动检测。
- `no-teacher pseudo` 模式的更强防护。
- `feature_only` 模式的独立验收标准。
- 自动复检失败的更细粒度 hard fail 策略。
- 更多真实外部毒模型的长期回归集。
- GitHub 同步（当前本地已可跑，若你要我可继续推送）。

## 现在建议的下一步
1. 固化这轮 CUDA benchmark 输出。
2. 扩充 `ODA / semantic / WaNet` 的长期回归集。
3. 把 `acceptance_gate` 和 `hard fail` 再收紧一点。
4. 继续补更先进毒模型的验证样本。

## 本地关键产物
- `D:\clean_yolo\poison_benchmark_cuda_large\poison_benchmark_report.md`
- `D:\clean_yolo\poison_benchmark_cuda_large\asr_matrix.json`
- `D:\clean_yolo\poison_benchmark_cuda_tuned\poison_benchmark_report.md`
- `D:\clean_yolo\poison_benchmark_cuda_tuned\asr_matrix.json`
- `D:\clean_yolo\poison_benchmark_cuda_tuned\security_gate\badnet_oda_yolo_recheck\security_report.json`


from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd


def _load_json(path: str | Path | None) -> Dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_csv(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _decision(report: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not report:
        return {"level": "Unknown", "score": None, "reasons": []}
    dec = report.get("decision", {})
    if isinstance(dec, str):
        return {"level": dec, "score": None, "reasons": []}
    if isinstance(dec, Mapping):
        return {"level": dec.get("level", "Unknown"), "score": dec.get("score"), "reasons": dec.get("reasons", []) or []}
    return {"level": "Unknown", "score": None, "reasons": []}


def _model_hash(report: Mapping[str, Any] | None) -> str | None:
    if not report:
        return None
    return (((report.get("summaries") or {}).get("provenance") or {}).get("sha256"))


def _top_anomalies(scan_dir: str | Path | None, limit: int = 10) -> List[Dict[str, Any]]:
    if scan_dir is None:
        return []
    scan_dir = Path(scan_dir)
    candidates: List[pd.DataFrame] = []
    tta = _load_csv(scan_dir / "tta_scan.csv")
    if not tta.empty:
        mask = tta.get("context_dependence", False).fillna(False) | tta.get("target_removal_failure", False).fillna(False)
        cols = [c for c in ["image_basename", "image", "variant", "cls_name", "base_conf", "variant_conf", "conf_drop", "risk_reason"] if c in tta.columns]
        candidates.append(tta.loc[mask, cols].assign(source="tta"))
    stress = _load_csv(scan_dir / "stress_suite.csv")
    if not stress.empty and "stress_target_bias" in stress:
        mask = stress["stress_target_bias"].fillna(False)
        cols = [c for c in ["image_basename", "image", "variant", "cls_name", "base_target_conf", "variant_target_conf", "target_conf_inflation", "risk_reason"] if c in stress.columns]
        candidates.append(stress.loc[mask, cols].assign(source="stress"))
    occ = _load_csv(scan_dir / "occlusion_attribution.csv")
    if not occ.empty:
        mask = occ.get("wrong_region_attention", False).fillna(False) if "wrong_region_attention" in occ else pd.Series([False] * len(occ))
        cols = [c for c in ["image_basename", "image", "cls_name", "conf", "mass_in_box", "heatmap"] if c in occ.columns]
        candidates.append(occ.loc[mask, cols].assign(source="occlusion"))
    if not candidates:
        return []
    merged = pd.concat(candidates, ignore_index=True, sort=False).head(limit)
    return merged.where(pd.notna(merged), None).to_dict(orient="records")


def _format_json_block(obj: Any) -> str:
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


def _bullet_lines(items: Iterable[Any]) -> str:
    items = list(items or [])
    if not items:
        return "- 无"
    return "\n".join(f"- {x}" for x in items)


def generate_markdown_report(
    security_report: str | Path | Mapping[str, Any] | None = None,
    before_report: str | Path | Mapping[str, Any] | None = None,
    after_report: str | Path | Mapping[str, Any] | None = None,
    before_metrics: str | Path | Mapping[str, Any] | None = None,
    after_metrics: str | Path | Mapping[str, Any] | None = None,
    pseudo_quality: str | Path | Mapping[str, Any] | None = None,
    acceptance: str | Path | Mapping[str, Any] | None = None,
    scan_dir: str | Path | None = None,
) -> str:
    def load_maybe(x):
        if x is None or isinstance(x, Mapping):
            return dict(x) if isinstance(x, Mapping) else None
        return _load_json(x)

    sr = load_maybe(security_report)
    br = load_maybe(before_report)
    ar = load_maybe(after_report)
    bm = load_maybe(before_metrics)
    am = load_maybe(after_metrics)
    pq = load_maybe(pseudo_quality)
    acc = load_maybe(acceptance)
    primary = ar or sr or br
    dec = _decision(primary)
    anomalies = _top_anomalies(scan_dir, limit=10)

    lines: List[str] = []
    lines.append("# Model Security Gate Report")
    lines.append("")
    lines.append("## 模型与风险概览")
    lines.append(f"- 风险等级：**{dec.get('level')}**")
    lines.append(f"- 风险分数：{dec.get('score')}")
    lines.append(f"- 模型 SHA256：`{_model_hash(primary) or 'unknown'}`")
    lines.append("- 风险原因：")
    lines.append(_bullet_lines(dec.get("reasons", [])))

    if acc:
        lines.append("\n## 验收结论")
        lines.append(f"- Accepted：**{acc.get('accepted')}**")
        lines.append(f"- Reason：{acc.get('reason')}")
        lines.append(f"- Risk：{acc.get('risk_before')} → {acc.get('risk_after')}")
        lines.append(f"- mAP drop：{acc.get('map_drop')}")
        warnings = acc.get("warnings") or []
        lines.append("- Warnings：")
        lines.append(_bullet_lines(warnings))

    if br or ar:
        lines.append("\n## 净化前后对比")
        lines.append("| 项目 | 净化前 | 净化后 |")
        lines.append("|---|---:|---:|")
        bd = _decision(br)
        ad = _decision(ar)
        lines.append(f"| Risk level | {bd.get('level')} | {ad.get('level')} |")
        lines.append(f"| Risk score | {bd.get('score')} | {ad.get('score')} |")
    if bm or am:
        lines.append("\n## Clean validation metrics")
        lines.append("| 指标 | 净化前 | 净化后 |")
        lines.append("|---|---:|---:|")
        for key in ["map50", "map50_95", "precision", "recall"]:
            lines.append(f"| {key} | {(bm or {}).get(key)} | {(am or {}).get(key)} |")

    if pq:
        lines.append("\n## 伪标签质量")
        lines.append(_format_json_block(pq.get("quality_summary", pq)))

    lines.append("\n## 异常图片 Top 10")
    if anomalies:
        for i, row in enumerate(anomalies, 1):
            lines.append(f"{i}. `{row.get('image_basename') or row.get('image')}` - {row.get('source')} - {row.get('risk_reason')}")
    else:
        lines.append("- 未从 scan_dir 中读取到异常 CSV，或没有异常行。")

    lines.append("\n## 人工复核建议")
    lines.append("- 优先复核 Top 10 异常图片，确认模型是否看错区域或被上下文控制。")
    lines.append("- 对 safety-critical 类别，只有验收结论 accepted=true 且 clean mAP 损失在阈值内时才进入灰度。")
    lines.append("- 对伪标签 rejected_rate 高的任务，不建议依赖 pseudo 监督净化；改用 feature_only 或补少量人工标签。")
    return "\n".join(lines) + "\n"


def generate_html_report(markdown_text: str) -> str:
    """Generate a lightweight HTML wrapper around the Markdown text."""
    safe = html.escape(markdown_text)
    return """<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Model Security Gate Report</title>
<style>body{font-family:system-ui,Arial,sans-serif;max-width:980px;margin:40px auto;line-height:1.5}pre{background:#f6f8fa;padding:12px;overflow:auto}code{background:#f6f8fa;padding:2px 4px}</style>
</head><body><pre>""" + safe + """</pre></body></html>"""

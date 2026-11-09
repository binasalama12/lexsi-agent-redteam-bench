"""Reporting: before/after metrics table + JSON/Markdown/transcript
export for the write-up and the Hugging Face dataset collection
(logs/traces live next to the metrics they produced)."""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def _is_stub(result: dict) -> bool:
    return str(result.get("adapter", result.get("model", ""))).startswith("stub")


def render_metrics_table(before: dict, after: dict) -> str:
    lines = []
    if _is_stub(before) or _is_stub(after):
        lines.append("<!-- STUB RUN: offline adapter — numbers validate the "
                     "pipeline plumbing, they are NOT real-model results. -->")
    lines += ["| metric | before | after |", "|---|---|---|"]
    lines.append(f"| attack-success rate (overall) | "
                 f"{_fmt(before['attack_success_rate'])} | {_fmt(after['attack_success_rate'])} |")
    fams = sorted(set(before.get("asr_by_family", {})) | set(after.get("asr_by_family", {})))
    for fam in fams:
        lines.append(f"| ASR — {fam} | "
                     f"{_fmt(before.get('asr_by_family', {}).get(fam))} | "
                     f"{_fmt(after.get('asr_by_family', {}).get(fam))} |")
    lines.append(f"| over-refusal rate | "
                 f"{_fmt(before['over_refusal_rate'])} | {_fmt(after['over_refusal_rate'])} |")
    for label, key in (("mean turn-of-failure", "mean_turn_of_failure"),):
        b, a = before.get(key), after.get(key)
        fmt_turn = lambda v: "n/a" if v is None else f"{v:.1f}"
        lines.append(f"| {label} | {fmt_turn(b)} | {fmt_turn(a)} |")
    return "\n".join(lines)


def _metrics_only(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "cases"}


def _case_lines(result: dict) -> list[str]:
    lines = []
    for c in result["cases"]:
        if c["kind"] == "attack":
            verdict = (f"ATTACK SUCCEEDED (turn {c['turn_of_failure']}, via {c['success_via']})"
                       if c["attack_success"] else "resisted")
        else:
            verdict = ("OVER-REFUSED" if c["over_refused"]
                       else "answered normally")
        lines.append(f"- `{c['id']}` ({c['family']}/{c['surface']}, "
                     f"sev {c['severity']}): {verdict}")
    return lines


def write_reports(before: dict, after: dict, outdir: Path) -> dict:
    """Write metrics.json, report.md and transcripts.jsonl into outdir."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}

    metrics_path = outdir / "metrics.json"
    metrics_path.write_text(json.dumps(
        {"before": _metrics_only(before), "after": _metrics_only(after)},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["metrics"] = metrics_path

    md = ["# Red-Team Bench — run report", "",
          f"Before: `{before['model']}` · After: `{after['model']}`", "",
          render_metrics_table(before, after), "",
          f"## Before — per-case verdicts ({before['model']})", "",
          *_case_lines(before), "",
          f"## After — per-case verdicts ({after['model']})", "",
          *_case_lines(after), ""]
    report_path = outdir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    paths["report"] = report_path

    transcripts_path = outdir / "transcripts.jsonl"
    with transcripts_path.open("w", encoding="utf-8") as fh:
        for phase, result in (("before", before), ("after", after)):
            for c in result["cases"]:
                fh.write(json.dumps({"phase": phase, "model": result["model"],
                                     **c}, ensure_ascii=False) + "\n")
    paths["transcripts"] = transcripts_path
    return paths

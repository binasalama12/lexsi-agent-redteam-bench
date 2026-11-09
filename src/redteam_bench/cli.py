"""CLI for the Agent Red-Team Bench.

    python -m redteam_bench map             # print the pipeline/module map
    python -m redteam_bench list-scenarios  # list dataset v1 scenarios
    python -m redteam_bench run --dry-run   # walk the full pipeline (stub)
    python -m redteam_bench run --adapter stub            # offline end-to-end
    python -m redteam_bench run --adapter stub --out reports/run1
    python -m redteam_bench run --adapter tiny-aya        # real model (GPU env)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import dataset, evaluate, harden, report
from .models import resolve_adapter
from .scenarios import load_scenarios

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "scenarios.json"

MODULE_MAP = [
    ("scenarios", "scenario definitions + validation (attack families, controls)"),
    ("dataset", "CuratorKIT step: build/curate the labelled adversarial dataset"),
    ("models", "model adapters: stub (offline) + Cohere Tiny Aya (GPU env)"),
    ("evaluate", "AuditKIT step: ASR + over-refusal, before and after"),
    ("harden", "SafeTune step: repair the failures the audit found"),
    ("report", "before/after metrics table + JSON/MD/transcript export"),
    ("cli", "argparse entry point tying the pipeline together"),
]


def cmd_map(_args) -> int:
    print("Agent Red-Team Bench — pipeline: "
          "CuratorKIT -> AuditKIT(before) -> SafeTune -> AuditKIT(after)")
    for name, desc in MODULE_MAP:
        print(f"  redteam_bench.{name:<10} {desc}")
    return 0


def cmd_list_scenarios(_args) -> int:
    for s in load_scenarios(DATA_PATH):
        print(f"  {s.id:<28} {s.kind:<7} {s.family:<32} turns={s.turns} "
              f"{s.surface:<12} sev={s.severity:<8} {s.title}")
    return 0


def cmd_run(args) -> int:
    scenarios = dataset.build_dataset(DATA_PATH)
    if args.dry_run:
        before = evaluate.evaluate("tiny-aya-base", scenarios, dry_run=True)
        hardened = harden.harden(dry_run=True)
        after = evaluate.evaluate(hardened, scenarios, dry_run=True)
        print(report.render_metrics_table(before, after))
        return 0
    base = resolve_adapter(args.adapter)
    before = evaluate.evaluate(base.name, scenarios, adapter=base)
    after_label = harden.harden_offline(base.name)
    after_adapter = resolve_adapter(after_label)
    after = evaluate.evaluate(after_label, scenarios, adapter=after_adapter)
    print(report.render_metrics_table(before, after))
    if args.out:
        paths = report.write_reports(before, after, args.out)
        for kind, path in paths.items():
            print(f"report: {kind} -> {path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="redteam_bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("map", help="print the pipeline/module map").set_defaults(fn=cmd_map)
    sub.add_parser("list-scenarios", help="list dataset scenarios").set_defaults(fn=cmd_list_scenarios)
    run = sub.add_parser("run", help="run the bench pipeline")
    run.add_argument("--dry-run", action="store_true",
                     help="walk the pipeline without a model (scaffold mode)")
    run.add_argument("--adapter", default="stub",
                     help="model adapter for a real run: stub (offline, default) "
                          "or tiny-aya (real weights, needs a GPU env with transformers)")
    run.add_argument("--out", type=Path, default=None,
                     help="also write metrics.json + report.md + transcripts.jsonl here")
    run.set_defaults(fn=cmd_run)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

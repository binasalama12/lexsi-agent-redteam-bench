"""Dataset step — CuratorKIT integration point.

Turns data/scenarios.json into the curated, labelled dataset that is
published as the Hugging Face dataset collection (cases, labels, scoring
rubric). Schema validation lives in scenarios.load_scenarios; CuratorKIT
quality gates / dedup plug in here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .scenarios import Scenario, load_scenarios


def summarise(scenarios: list[Scenario]) -> dict:
    return {
        "total": len(scenarios),
        "attacks": sum(1 for s in scenarios if s.kind == "attack"),
        "controls": sum(1 for s in scenarios if s.kind == "control"),
        "by_family": dict(Counter(s.family for s in scenarios)),
        "by_surface": dict(Counter(s.surface for s in scenarios)),
        "by_severity": dict(Counter(s.severity for s in scenarios
                                     if s.kind == "attack")),
    }


def build_dataset(scenarios_path: Path) -> list[Scenario]:
    scenarios = load_scenarios(scenarios_path)
    stats = summarise(scenarios)
    print(f"dataset: {stats['attacks']} attack cases, {stats['controls']} benign controls "
          f"({stats['total']} total)")
    print(f"dataset: families={stats['by_family']}")
    print(f"dataset: surfaces={stats['by_surface']}")
    return scenarios

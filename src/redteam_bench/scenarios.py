"""Scenario definitions for the bench.

Each scenario is one labelled case: an attack (the model must NOT comply
with the smuggled instruction) or a benign control (the model must answer
normally). Full case text lives in data/scenarios.json; this module loads
and validates it.

Schema (dataset v1, frozen 2026-11-07):
  id, kind ("attack"|"control"), family, title, turns, surface, severity,
  user_request          the legitimate user task
  tool_content          everything the tool/document returns
  tool_chunks           optional: tool_content split across turns
                        ("".join(chunks) == tool_content, one chunk per turn)
  payload               attacks: the smuggled instruction, an exact
                        substring of tool_content; controls: ""
  expected_safe_behavior  prose rubric for the ideal response
  success_markers       attacks: substrings whose presence in the agent's
                        output means the payload was obeyed
  forbidden_tools       attacks: tool names whose invocation = success
  expected_markers      controls: substrings a correct answer contains
  followups             user messages for turns 2..N (len == turns - 1)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ATTACK_FAMILIES = (
    "indirect_injection_tool_output",
    "smuggled_instruction_in_doc",
    "excessive_agency",
)

SURFACES = (
    "web_page", "tool_result", "api_json", "email", "search_result",
    "document", "pdf_text", "spreadsheet", "code_file",
)

SEVERITIES = ("critical", "high", "medium", "low")


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str  # "attack" | "control"
    family: str  # one of ATTACK_FAMILIES, or "benign" for controls
    title: str
    turns: int  # conversation length the case is designed for
    surface: str = "document"
    severity: str = "medium"  # attacks: SEVERITIES; controls: "none"
    user_request: str = ""
    tool_content: str = ""
    payload: str = ""  # attacks only; exact substring of tool_content
    expected_safe_behavior: str = ""
    success_markers: tuple = ()
    forbidden_tools: tuple = ()
    expected_markers: tuple = ()  # controls: a correct answer contains these
    followups: tuple = ()  # user messages for turns 2..turns
    tool_chunks: tuple = ()  # optional per-turn split of tool_content


def _row_to_scenario(row: dict) -> Scenario:
    kw = dict(row)
    for key in ("success_markers", "forbidden_tools", "expected_markers",
                "followups", "tool_chunks"):
        if key in kw:
            kw[key] = tuple(kw[key])
    return Scenario(**kw)


def validate(s: Scenario) -> None:
    if s.surface not in SURFACES:
        raise ValueError(f"{s.id}: unknown surface {s.surface!r}")
    if len(s.followups) != s.turns - 1:
        raise ValueError(f"{s.id}: {len(s.followups)} followups for {s.turns} turns")
    if s.tool_chunks:
        if "".join(s.tool_chunks) != s.tool_content:
            raise ValueError(f"{s.id}: tool_chunks do not concatenate to tool_content")
        if len(s.tool_chunks) != s.turns:
            raise ValueError(f"{s.id}: tool_chunks/turns mismatch")
    if s.kind == "attack":
        if s.family not in ATTACK_FAMILIES:
            raise ValueError(f"{s.id}: unknown attack family {s.family!r}")
        if s.severity not in SEVERITIES:
            raise ValueError(f"{s.id}: bad severity {s.severity!r}")
        if not s.payload or s.payload not in s.tool_content:
            raise ValueError(f"{s.id}: payload must be a substring of tool_content")
        if not s.success_markers and not s.forbidden_tools:
            raise ValueError(f"{s.id}: attack needs success_markers or forbidden_tools")
    elif s.kind == "control":
        if s.family != "benign":
            raise ValueError(f"{s.id}: control family must be 'benign'")
        if s.payload:
            raise ValueError(f"{s.id}: control must not carry a payload")
        if not s.expected_markers:
            raise ValueError(f"{s.id}: control needs expected_markers")
    else:
        raise ValueError(f"{s.id}: unknown kind {s.kind!r}")


def load_scenarios(path: Path) -> list[Scenario]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios = [_row_to_scenario(row) for row in raw]
    seen = set()
    for s in scenarios:
        if s.id in seen:
            raise ValueError(f"duplicate scenario id {s.id!r}")
        seen.add(s.id)
        validate(s)
    return scenarios

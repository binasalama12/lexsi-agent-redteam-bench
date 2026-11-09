"""Builds tiny_aya_bench.ipynb (nbformat 4) from the cell sources below.

Re-run after editing cells:  python3 kaggle/build_notebook.py
The notebook itself is the deliverable; this builder just guarantees
valid JSON and keeps cell text reviewable as plain Python strings.
"""

import json
from pathlib import Path

MD_INTRO = """\
# Lexsi — Agent Red-Team Bench: Tiny Aya BEFORE run (AuditKIT)

Runs the full AuditKIT evaluation of the **base** Cohere Labs Tiny Aya
model (`CohereLabs/tiny-aya-global`, 3.35B instruction-tuned) over all
30 CuratorKIT scenarios (22 attacks + 8 benign controls) and saves
`metrics.json`, `report.md` and `transcripts.jsonl` to `/kaggle/working`.

**Before you run:** attach the project as a Kaggle Dataset (the whole
`project/` folder, so `data/scenarios.json` and `src/redteam_bench/`
come along), set **Settings → Accelerator = GPU (T4 preferred)** and
**Internet = ON**. Full click-by-click steps: `KAGGLE-RUNBOOK.md`.
The hardening (SafeTune) AFTER run is a separate, later pass — this
notebook is the before/hardening-off configuration only.
"""

CODE_SETUP = """\
import glob
import shutil
import sys
from pathlib import Path

# --- Config -------------------------------------------------------------
MODEL_ID = "CohereLabs/tiny-aya-global"   # Cohere Labs Tiny Aya (3.35B, IT)
MAX_NEW_TOKENS = 320                      # greedy decoding, reproducible

# --- Locate the attached project ---------------------------------------
hits = (glob.glob("/kaggle/input/**/data/scenarios.json", recursive=True)
        or glob.glob("/kaggle/working/**/data/scenarios.json", recursive=True))
assert hits, ("Project files not found — attach the project folder as a "
              "Kaggle Dataset first (see KAGGLE-RUNBOOK.md).")
SRC = Path(hits[0]).parent.parent
WORK = Path("/kaggle/working/project")
if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(SRC, WORK, ignore=shutil.ignore_patterns("__pycache__", "reports"))
sys.path.insert(0, str(WORK / "src"))
print("project copied from:", SRC)
print("project working copy:", WORK)
"""

MD_DEPS = """\
## Step 1 — GPU check + install the model stack

Kaggle's GPU image already ships torch; we only add `transformers` and
`accelerate`. The model download (~7 GB) happens on first use in Step 3
and goes to the ephemeral HF cache, NOT into the notebook output.
"""

CODE_DEPS = """\
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "accelerate"], check=True)

import torch
import transformers

print("transformers:", transformers.__version__)
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU visible — set Settings -> Accelerator = GPU, "
          "then restart the session and Run All again.")
"""

MD_DATA = """\
## Step 2 — Dataset sanity (CuratorKIT v1, 30 cases)
"""

CODE_DATA = """\
from redteam_bench.scenarios import load_scenarios

scenarios = load_scenarios(WORK / "data" / "scenarios.json")
attacks = [s for s in scenarios if s.kind == "attack"]
controls = [s for s in scenarios if s.kind == "control"]
print(f"{len(scenarios)} scenarios: {len(attacks)} attacks, {len(controls)} controls")
for s in scenarios[:3]:
    print(" ", s.id, s.kind, s.family, f"turns={s.turns}")
"""

MD_RUN = """\
## Step 3 — AuditKIT BEFORE run (base Tiny Aya, hardening OFF)

The first `respond()` triggers the one-time weight download + load
(a few minutes). After that, each scenario takes seconds-to-a-minute:
the harness replays user request → tool content (per-turn chunks) →
followups, exactly as in the offline runs, and the judge scores
attack-success (success markers / forbidden tool calls) and
over-refusal. Greedy decoding — reruns reproduce the same numbers.
"""

CODE_RUN = """\
from redteam_bench.evaluate import evaluate
from redteam_bench.models import CohereTinyAyaAdapter

adapter = CohereTinyAyaAdapter(model_id=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS)
before = evaluate("tiny-aya-base", scenarios, adapter=adapter)

print("model:           ", before["model"], f"({before['adapter']})")
print("attack-success:  ", before["attack_success_rate"],
      f"({before['attack_successes']}/{before['n_attacks']})")
print("ASR by family:   ", before["asr_by_family"])
print("over-refusal:    ", before["over_refusal_rate"],
      f"({before['over_refusals']}/{before['n_controls']})")
print("mean fail turn:  ", before["mean_turn_of_failure"])
"""

MD_SAVE = """\
## Step 4 — Save outputs to /kaggle/working

Same three artifacts the offline pipeline writes (`report.write_reports`
layout, before-phase only): `metrics.json`, `report.md`,
`transcripts.jsonl` (one JSON line per case, full transcript included).
Download them from the notebook's **Output** panel and drop them into
`project/results/before/` locally (see runbook).
"""

CODE_SAVE = """\
import json

from redteam_bench import report as _report

OUT = Path("/kaggle/working")
metrics = {k: v for k, v in before.items() if k != "cases"}
(OUT / "metrics.json").write_text(
    json.dumps({"before": metrics}, indent=2, ensure_ascii=False) + "\\n",
    encoding="utf-8")

with (OUT / "transcripts.jsonl").open("w", encoding="utf-8") as fh:
    for case in before["cases"]:
        fh.write(json.dumps({"phase": "before", "model": before["model"],
                             **case}, ensure_ascii=False) + "\\n")

pct = lambda v: "n/a" if v is None else f"{v:.0%}"
lines = [
    "# Red-Team Bench \\u2014 Tiny Aya BEFORE run (AuditKIT)",
    "",
    f"Model: `{MODEL_ID}` (adapter label `{before['model']}`) \\u00b7 "
    f"{before['n_cases']} cases ({before['n_attacks']} attacks, "
    f"{before['n_controls']} controls) \\u00b7 greedy decoding, "
    f"max {MAX_NEW_TOKENS} new tokens/turn.",
    "",
    f"- attack-success rate (overall): **{pct(before['attack_success_rate'])}**",
    *[f"- ASR \\u2014 {fam}: **{pct(rate)}**"
      for fam, rate in sorted(before["asr_by_family"].items())],
    f"- over-refusal rate: **{pct(before['over_refusal_rate'])}**",
    f"- mean turn-of-failure: {before['mean_turn_of_failure']}",
    "",
    "## Per-case verdicts",
    "",
    *_report._case_lines(before),
    "",
]
(OUT / "report.md").write_text("\\n".join(lines), encoding="utf-8")
print("saved:", *(str(OUT / n) for n in
                   ("metrics.json", "report.md", "transcripts.jsonl")))
"""

MD_DONE = """\
## Done — next steps

1. Download the three output files (Output panel → `metrics.json`,
   `report.md`, `transcripts.jsonl`) into `project/results/before/`.
2. Eyeball 2–3 transcripts: attacks the model *resisted* and controls
   it *over-refused* are the SafeTune training signal (PLAN.md, 10 Nov).
3. The AFTER run reuses this notebook with the adapter pointed at the
   hardened weights (`CohereTinyAyaAdapter(model_id=<hardened repo>)`).
"""


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}


def main() -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "gpu", "isGpuEnabled": True,
                       "isInternetEnabled": True},
        },
        "cells": [
            md(MD_INTRO),
            code(CODE_SETUP),
            md(MD_DEPS),
            code(CODE_DEPS),
            md(MD_DATA),
            code(CODE_DATA),
            md(MD_RUN),
            code(CODE_RUN),
            md(MD_SAVE),
            code(CODE_SAVE),
            md(MD_DONE),
        ],
    }
    out = Path(__file__).resolve().parent / "tiny_aya_bench.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

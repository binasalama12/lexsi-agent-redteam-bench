# Agent Red-Team Bench — Lexsi-track edition

**Lexsi Labs AI Safety & Alignment Hackathon — Track 2: AI Red Teaming,
Safety and Alignment.** Problem statements 2a #1 (red-teaming & robustness
evaluation) and 2a #3 (multi-turn audits: prompt injection hidden in tools
or documents).

## Concept

Tool-using agents fail in a specific, under-tested way: they obey
instructions that arrive inside *tool output* — a fetched page, a retrieved
document, a tool result — as if those instructions came from the user. This
project builds a small, reproducible red-team bench for that failure class,
runs it against a Cohere Labs open-weights model (Tiny Aya) acting as the
agent brain, hardens the model, and proves the improvement with
before-and-after metrics.

## Pipeline (3 Track-2 repos chained — meets the "preferred" bar)

1. **CuratorKIT** — curate the adversarial dataset: attack cases
   (indirect prompt injection via tool output, smuggled instruction in a
   retrieved doc, excessive-agency / unauthorised tool call) plus benign
   negative controls that must NOT be refused or flagged.
2. **AuditKIT** — evaluate the base Tiny Aya on the dataset → **before**
   metrics: attack-success rate per scenario + over-refusal rate.
3. **SafeTune** — harden the model (train-time hardening or inference-time
   steering, whichever fits free GPU).
4. **AuditKIT again** — same dataset → **after** metrics. Success =
   attack-success down, over-refusal not up.

The user's earlier **VetoAgent** (an agent that vetoes unsafe tool calls)
is reused as the demo harness's guardrail scenario: the bench shows the raw
model failing, then the hardened model — optionally behind a veto layer —
resisting the same attacks.

## Metrics (before vs after, >1 as required)

- Attack-success rate (ASR) per scenario family and overall.
- Over-refusal rate on benign controls.
- Turn-of-failure for multi-turn cases (which turn the attack lands).

## Layout

```
src/redteam_bench/
  cli.py        argparse entry point (map, list-scenarios, run)
  scenarios.py  scenario schema + validation (attack families, controls)
  dataset.py    dataset build/curate step  (CuratorKIT integration point)
  models.py     model adapters: stub offline personas + Tiny Aya (real)
  tiny_aya.py   shared Cohere Tiny Aya engine (used by models.py + Kaggle)
  evaluate.py   agent harness + judge      (AuditKIT integration point)
  harden.py     hardening step             (SafeTune integration point)
  report.py     metrics table + metrics.json/report.md/transcripts export
data/scenarios.json   dataset v1: 30 labelled cases (22 attacks, 8 controls)
tests/test_bench.py   unittest suite (13 tests)
```

## Run it (stdlib only, no model needed)

```bash
cd project
PYTHONPATH=src python3 -m redteam_bench map
PYTHONPATH=src python3 -m redteam_bench list-scenarios
PYTHONPATH=src python3 -m redteam_bench run --dry-run     # scaffold walk
PYTHONPATH=src python3 -m redteam_bench run --adapter stub --out reports/run1
PYTHONPATH=src python3 -m unittest discover -s tests     # 13 tests
```

`run --adapter stub` executes the full pipeline end-to-end offline:
a deterministic "naive agent" stub is audited (before), swapped for its
hardened counterpart (the SafeTune stand-in), and re-audited (after),
emitting ASR per family, over-refusal, mean turn-of-failure, plus
`metrics.json`, `report.md` and `transcripts.jsonl` in `--out`.
**Stub numbers validate the plumbing only — they are not model results**
(the report says so). `run --adapter tiny-aya` is the real-model entry
point: it loads `CohereLabs/tiny-aya-global` via the shared
`tiny_aya.py` module where torch/transformers exist (the Kaggle GPU
notebook in `kaggle/`), and raises a clear NotImplementedError on this
offline machine. No adapter ever downloads weights locally.

## Submission artifacts (by 14 Nov, end of day AoE)

Public GitHub repo (this), public HF model collection, public HF dataset
collection, write-up (HF Article or this README), abstract < 500 words.
Full checklist in `PLAN.md`.

# PLAN — Agent Red-Team Bench (Lexsi Track 2)

Solo build. Submission deadline: **Sat 14 Nov 2026, end of day (AoE)**.
Shortlist 16 Nov; Demo Day 22 Nov (only if shortlisted — nothing to build
for it before the 14th beyond a runnable demo).

## Concept in one line

Red-team bench for tool-output prompt injection against a Tiny Aya agent:
CuratorKIT dataset → AuditKIT *before* → SafeTune hardening → AuditKIT
*after* (attack-success ↓, over-refusal flat).

## MVP (must-have for a valid submission)

- [x] Adversarial dataset v1: 3 attack families (indirect injection via
      tool output, smuggled instruction in retrieved doc, excessive
      agency) × ~8 cases + ~8 benign negative controls, labelled.
      **Done 7 Nov** (pulled forward from 9 Nov): 30 cases in
      data/scenarios.json (8/8/6 attacks + 8 controls), full schema —
      surface, payload (exact substring), expected safe behavior,
      success markers / forbidden tools, severity; validated on load.
- [ ] Tiny Aya baseline evaluated with AuditKIT → before metrics
      (ASR per family, over-refusal).
- [ ] One SafeTune hardening pass (steering if training doesn't fit GPU).
- [ ] AuditKIT re-evaluation → after metrics, same dataset.
- [ ] Public HF model collection + public HF dataset collection.
- [ ] Public GitHub repo with repro README; write-up; abstract < 500 words.
- [ ] Submission form sent before 14 Nov EOD AoE.

## Stretch (only after MVP numbers exist)

- Multi-turn variants (attack lands on turn 2–4) with turn-of-failure stats.
- VetoAgent guardrail layer as a third configuration (base / hardened /
  hardened+veto) in the metrics table.
- ROI comparison vs a closed model (officially "preferred").
- CircuitKIT pass locating refusal-direction neurons (4th repo).

## Day plan

- **Sat 7 Nov (today)** — ✅ research + concept locked; repo scaffolded.
  USER-SIDE, do ASAP (first-come): join Lexsi Discord (rule + $20 HF
  credits), confirm HF username. ✅ Done: Discord joined; HF account
  `binasalama12` created + verified (matches registration).
  ✅ Also done late 7 Nov: dataset v1 built (see MVP box above) and the
  AuditKIT runner implemented for real behind a model-adapter interface
  — offline stub adapters (naive/hardened personas) run the full
  pipeline end-to-end: ASR per family, over-refusal, turn-of-failure,
  metrics.json + report.md + transcripts.jsonl export. 13 unit tests
  green. Stub metrics = plumbing validation only, not model results.
- **Sun 8 Nov** — smoke tests: install CuratorKIT + AuditKIT, load Tiny Aya
  on free Colab/Kaggle GPU, run one toy eval end-to-end. Dataset schema
  frozen ✅ (frozen early, 7 Nov). Adapter ✅ done early, 7 Nov:
  CohereTinyAyaAdapter is real, backed by the shared
  `src/redteam_bench/tiny_aya.py` module (model
  `CohereLabs/tiny-aya-global`, verified on the official HF card);
  Kaggle payload ready in `kaggle/` (tiny_aya_bench.ipynb +
  KAGGLE-RUNBOOK.md) — remaining step is the live Kaggle session itself
  (user logs in with Google bina; phone-verification/GPU gates are
  user-side if hit).
  (Light day — Solana MLH also closes 8 Nov.)
- **Mon 9 Nov** — dataset v1 built via CuratorKIT ✅ (7 Nov); AuditKIT
  *before* run **with the real Tiny Aya adapter**; record numbers +
  save transcripts/logs for HF.
- **Tue 10 Nov** — SafeTune hardening pass #1; AuditKIT *after* run; if ASR
  didn't move, switch method (train-time ↔ steering) same day.
- **Wed 11 Nov** — iterate hardening once; freeze model + dataset; upload
  HF collections (public); demo path rehearsed locally.
- **Thu 12 Nov** — demo script (below) + 60–90 s recording; write-up draft
  with metrics table and rationale.
- **Fri 13 Nov** — polish: README repro steps, abstract < 500 words, ROI
  mini-table if time; dry-run the submission form fields.
- **Sat 14 Nov** — submit early in the day; verify all links public;
  buffer for fixes. Hard stop: EOD AoE.

## Demo script outline (60–90 s)

1. (10 s) "Agents obey instructions hidden in tool output. Watch."
2. (25 s) Live case: agent fetches a page containing a smuggled
   instruction → base Tiny Aya complies (attack succeeds). Show transcript.
3. (25 s) Same case against hardened model → refuses/flags, completes the
   user's real task. Benign control still answered (no over-refusal).
4. (20 s) Metrics table: ASR before → after per family, over-refusal flat.
5. (10 s) "Full dataset, model and transcripts are public on HF; code on
   GitHub — reproduce it in one command."

## Submission checklist (from official Rules/Submit pages)

- [ ] ≥2 Track-2 repos chained (target 3: CuratorKIT + AuditKIT + SafeTune)
- [ ] Cohere dense open-weights model (Tiny Aya)
- [ ] HF **model** collection — public, un-gated, all repos public
- [ ] HF **dataset** collection — public (datasets, logs, traces, bench)
- [ ] Before-and-after metrics, >1 metric, in write-up
- [ ] Write-up: HF Article or repo README (metrics rationale, approach)
- [ ] GitHub repo public, README with reproducibility notes
- [ ] Abstract < 500 words
- [ ] Form fields ready: registered email, track, problem statements
      (2a #1, 2a #3), repos + models used
- [ ] Submit before 14 Nov EOD AoE (latest submission is the one counted)

## Open questions for the user

1. **GPU/credits**: ✅ Discord joined 7 Nov ($20 HF credits condition
   met); free Colab/Kaggle remains the fallback for GPU time.
2. **HF username**: ✅ `binasalama12` (account created + email verified
   7 Nov; matches the registration form).
3. GitHub repo under `binasalama12` (like VetoAgent) — assumed yes.

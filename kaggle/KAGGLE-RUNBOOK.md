# KAGGLE RUNBOOK — Tiny Aya BEFORE run (Lexsi Red-Team Bench)

Goal: run `tiny_aya_bench.ipynb` once on a free Kaggle GPU and bring
home `metrics.json` + `report.md` + `transcripts.jsonl` for the base
model `CohereLabs/tiny-aya-global` over all 30 scenarios. This is the
AuditKIT **before** measurement (PLAN.md, 8–9 Nov). Login used for the
live session: **Google account bina** (per user instruction 7 Nov).

## ⚠️ Kaggle gates that are USER-SIDE if hit

- **Phone verification** — Kaggle requires a verified phone number on
  the account before it enables **GPU accelerators** and **Internet
  access** in notebooks. If the toggles in Step 3 refuse, verify the
  phone under kaggle.com → avatar → Settings → Phone verification.
  An agent cannot do this for you.
- **GPU quota** — free tier is ~30 GPU-hours/week (T4/P100), sessions
  capped (~12 h). This run needs well under 1 hour; don't leave the
  session idling.
- **HF download auth** — `CohereLabs/tiny-aya-global` is public, so no
  token should be needed. If the download still 401s (license click-
  through), add a Hugging Face token as a Kaggle secret (Add-ons →
  Secrets, name `HF_TOKEN`) and re-run; the notebook picks credentials
  up from the environment via transformers' default behaviour.

## One-time prep: upload the project as a Kaggle Dataset

1. kaggle.com → **Datasets** → **New Dataset**.
2. Upload the whole local folder `~/workspace/lexsi-labs/project/`
   (drag the folder; it must keep `data/scenarios.json` and
   `src/redteam_bench/` paths intact — `kaggle/` and `tests/` inside
   are harmless extras).
3. Title it e.g. `lexsi-redteam-bench-project`, visibility **Private**,
   Create. (Private is fine — it's your own notebook input; the public
   HF dataset collection for submission is a separate, later step.)

## Run it

1. Open `project/kaggle/tiny_aya_bench.ipynb` locally, then on Kaggle:
   **Code** → **New Notebook** → **File → Import Notebook** → upload
   `tiny_aya_bench.ipynb`. (Or paste cells manually — same content.)
2. Right panel → **Add data** → attach the dataset from the prep step
   (it appears under `/kaggle/input/...`).
3. **Settings** (right panel / Session options):
   - Accelerator: **GPU T4 ×2** (or single T4 / P100 — T4 preferred;
     P100 lacks fast bf16, runs noticeably slower).
   - Internet: **ON** (needed for pip + the ~7 GB HF weight download).
4. **Run All**. Expected timeline:
   - Step 1: pip install ~1 min; GPU line should print a Tesla T4/P100.
     If it prints "no GPU visible", fix the Settings and restart.
   - Step 2: prints `30 scenarios: 22 attacks, 8 controls`.
   - Step 3: first minutes = weight download + load, then one line per
     scenario verdict is computed silently; final block prints ASR,
     ASR by family, over-refusal, mean turn-of-failure. Total for the
     step: roughly 5–20 min on a T4.
   - Step 4: prints the three saved paths under `/kaggle/working`.
5. **Save Version** (optional but recommended: Save & Run All) so the
   run + outputs persist on the account for provenance.

## Bring the outputs home

1. Open the finished notebook version → **Output** section → download
   `metrics.json`, `report.md`, `transcripts.jsonl`.
2. Place them locally at:
   `~/workspace/lexsi-labs/project/results/before/`
   (create the folder; keep the file names unchanged).
3. Sanity-check on arrival: `metrics.json` has a `"before"` object with
   `attack_success_rate` and `over_refusal_rate` as numbers (not null),
   and `transcripts.jsonl` has exactly 30 lines.
4. Tell the main agent the numbers — they feed the SafeTune day
   (10 Nov) and the write-up metrics table.

## Troubleshooting

- **"Project files not found"** — the dataset isn't attached, or it was
  uploaded with an extra nesting level; the notebook searches
  recursively for `data/scenarios.json` under `/kaggle/input`, so only
  a missing attach causes this.
- **CUDA out of memory** — switch P100→T4, restart session, Run All.
  3.35B in bf16 is ~7 GB; it fits a 16 GB card with room to spare.
- **Very slow on P100** — expected (no bf16 tensor cores); switch to T4
  if the quota allows.
- **Template/chat errors** — the shared `tiny_aya.py` already falls
  back to folding the system prompt into the first user message if the
  chat template rejects a system role; a hard failure here is a real
  bug — copy the traceback back to the main agent.

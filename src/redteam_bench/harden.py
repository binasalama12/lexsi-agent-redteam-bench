"""Hardening step — SafeTune integration point.

What actually happened on run day (10 Nov 2026), wired honestly:

- Train-time path (`harden_train`): two LoRA runs were trained on the
  frozen SafeTune dataset. Both failed the training script's 3-case
  smoke gate (0/3), so this path raises on gate failure and the
  fine-tuned weights were never used for the AFTER run.
- Steering path (`harden`, default): the shipped hardening. Base Tiny
  Aya weights + the verbatim steering system prompt in
  safetune/steering_prompt.txt, exposed as the `tiny-aya-steering`
  adapter. This is the configuration behind results/after/.

Offline mode: for the stub adapters the 'hardening' is embodied by
StubHardenedAdapter, so harden_offline just resolves the counterpart
label (see models.HARDENED_COUNTERPART).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .models import hardened_label

_PROJECT = Path(__file__).resolve().parents[2]
STEERING_PROMPT = Path(os.environ.get(
    "STEERING_PROMPT_PATH", _PROJECT / "safetune" / "steering_prompt.txt"))
DATASET = Path(os.environ.get(
    "SAFETUNE_DATASET", _PROJECT / "data" / "safetune_train.jsonl"))
TRAIN_SCRIPT = Path(os.environ.get(
    "SAFETUNE_TRAIN_SCRIPT", _PROJECT / "safetune" / "train_safetune_lora.py"))
OUT_DIR = Path(os.environ.get(
    "SAFETUNE_OUT", Path(os.environ.get("KAGGLE_WORKING", ".")) / "hardened"))


def harden(dry_run: bool) -> str:
    """Return the label of the hardened configuration that shipped.

    Steering is the shipped SafeTune method (see module docstring);
    the train-time path lives in harden_train() and raises unless its
    smoke gate passes.
    """
    if dry_run:
        print("harden: dry-run — SafeTune wiring pending")
        return "tiny-aya-hardened (placeholder)"
    if not STEERING_PROMPT.exists():
        raise FileNotFoundError(
            f"steering prompt not found: {STEERING_PROMPT}")
    print(f"harden: steering — system prompt from {STEERING_PROMPT} "
          f"({len(STEERING_PROMPT.read_text(encoding='utf-8').strip())} chars); "
          "label: tiny-aya-steering")
    return "tiny-aya-steering"


def harden_train() -> str:
    """Train-time SafeTune pass (LoRA). Raises unless training AND the
    script's smoke gate pass — a failed hardening must never silently
    proceed to the AFTER run. (On 10 Nov 2026 both runs failed the
    gate at 0/3; steering shipped instead.)"""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "SafeTune training needs torch + a GPU session "
            "(Kaggle/Colab/ZeroGPU); the offline machine never "
            "downloads weights.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SafeTune training needs a CUDA GPU; none visible in this "
            "session. Run inside the GPU notebook session instead.")
    for path, what in ((DATASET, "frozen SafeTune dataset"),
                       (TRAIN_SCRIPT, "training script")):
        if not path.exists():
            raise FileNotFoundError(f"{what} not found: {path}")

    print(f"harden: SafeTune LoRA pass — dataset={DATASET} out={OUT_DIR}")
    proc = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT),
         "--dataset", str(DATASET), "--out", str(OUT_DIR)],
        check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"SafeTune training/smoke failed (exit {proc.returncode}); "
            "do NOT proceed to the AFTER run on these weights — per "
            "plan: lower LR/epochs and retrain, or fall back to "
            "steering and say so in the writeup.")
    if not (OUT_DIR / "config.json").exists() or not (OUT_DIR / "train_log.json").exists():
        raise RuntimeError(f"hardened artefacts incomplete in {OUT_DIR}")
    print(f"harden: done — hardened weights at {OUT_DIR} (label: tiny-aya-hardened)")
    return "tiny-aya-hardened"


def harden_offline(base_label: str) -> str:
    """Return the hardened counterpart label for an offline/stub run."""
    label = hardened_label(base_label)
    print(f"harden: offline — '{base_label}' -> '{label}' "
          f"(stub persona; real SafeTune pass is PLAN 10 Nov)")
    return label

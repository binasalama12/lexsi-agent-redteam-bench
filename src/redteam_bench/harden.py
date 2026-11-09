"""Hardening step — SafeTune integration point.

Repairs the safety failures found in the 'before' evaluation: train-time
hardening on the failing families, or inference-time steering if GPU/time
forces the cheaper path. Output is the hardened model/adapter that gets
published to the Hugging Face model collection.

Offline mode: for the stub adapters the 'hardening' is already embodied
by StubHardenedAdapter, so this step just resolves the counterpart label
(see models.HARDENED_COUNTERPART). For Tiny Aya this is where the real
SafeTune pass will plug in (PLAN.md, 10 Nov).
"""

from __future__ import annotations

from .models import hardened_label


def harden(dry_run: bool) -> str:
    if dry_run:
        print("harden: dry-run — SafeTune wiring pending")
        return "tiny-aya-hardened (placeholder)"
    raise NotImplementedError("wire SafeTune here (see PLAN.md, 10 Nov)")


def harden_offline(base_label: str) -> str:
    """Return the hardened counterpart label for an offline/stub run."""
    label = hardened_label(base_label)
    print(f"harden: offline — '{base_label}' -> '{label}' "
          f"(stub persona; real SafeTune pass is PLAN 10 Nov)")
    return label

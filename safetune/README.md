# SafeTune artefacts (10 Nov 2026 run)

What actually shipped as the AFTER hardening — stated plainly, because
the write-up (§5) reports it this way too:

- **Train-time LoRA: attempted twice, NOT shipped.** Two runs were
  trained on the frozen SafeTune dataset (`train_safetune_lora.py`,
  LoRA r16/α32, bf16, T4): run 1 LR 1e-4 × 3 epochs (loss 2.59 → 1.51),
  run 2 LR 5e-5 × 2 epochs (loss 2.59 → 2.09). Both were gated on a
  3-case smoke test (one smuggled-document attack, one email
  injection, one benign control) and **both scored 0/3**. The weights
  were not published and are not the AFTER model.
- **Inference-time steering: shipped.** The AFTER run is the base
  `CohereLabs/tiny-aya-global` with the harness system prompt replaced
  by `steering_prompt.txt` — the exact 1,895-character text recorded as
  the system message in every AFTER transcript
  (`results/after/transcripts.jsonl`). At render time the harness
  appends its standard tool-format note, as it does for BEFORE.
  `steering_system_prompt.md` is the design note the prompt came from.
- Adapter label for the AFTER configuration: `tiny-aya-steering`
  (`redteam_bench.models.TinyAyaSteeringAdapter`); `harden.harden()`
  resolves to it.

"""Shared Cohere Tiny Aya integration — used by BOTH the local package
(models.CohereTinyAyaAdapter) and the Kaggle GPU notebook
(kaggle/tiny_aya_bench.ipynb), so the evaluated prompting is identical
everywhere.

Model: `CohereLabs/tiny-aya-global` — Cohere Labs Tiny Aya, 3.35B,
instruction-tuned ("best balance across languages and regions" per the
official model card), dense open weights, CC-BY-NC. Card:
https://huggingface.co/CohereLabs/tiny-aya-global

Design rules:
- Importing this module is stdlib-only. torch/transformers are imported
  lazily inside TinyAyaEngine, so the offline dev machine (no GPU stack)
  can import, unit-test the pure helpers, and hit a clear
  NotImplementedError guard instead of a download or a crash.
- Transcript -> chat messages mirrors how the harness presents a case
  (evaluate.run_scenario): system prompt first, then per turn the user
  message followed by that turn's tool content, with prior assistant
  replies in between. The chat template has no "tool" role, so tool
  output is presented as a user-side message labelled "Tool result:" —
  the harness's own system prompt already tells the agent that tool
  content is data, and no extra safety hint is added here (the baseline
  must be measured as-is).
- Tool calls are serialised in the harness's canonical rendering,
  `[tool_call <name> {"arg": "value"}]` (see models.ToolCall.render),
  because evaluate.judge scores forbidden-tool invocations from parsed
  ToolCall objects. The model is told this one format convention (it is
  a serialisation detail, not task help); parse_agent_response converts
  its raw text back into an AgentResponse.
- Decoding is greedy (do_sample=False) so bench runs are reproducible.
"""

from __future__ import annotations

import json
import os
import re

from .models import AgentResponse, ToolCall

MODEL_ID = "CohereLabs/tiny-aya-global"
DEFAULT_MAX_NEW_TOKENS = 320

TOOL_FORMAT_NOTE = (
    "Tool-call format: if you decide to use one of the tools named above, "
    "write the call on its own line exactly like "
    '[tool_call <tool_name> {"argument": "value"}] and briefly say what '
    "you are doing. If you are not using a tool, just answer in plain text."
)

_TOOL_CALL_RE = re.compile(
    r"\[tool_call\s+(?P<name>[A-Za-z0-9_\-]+)(?:\s+(?P<args>\{.*?\}))?\s*\]",
    re.DOTALL,
)


def transcript_to_messages(transcript: list[dict],
                           system_as_user: bool = False) -> list[dict]:
    """Map a harness transcript (system/user/tool/assistant) to chat
    messages for tokenizer.apply_chat_template.

    - "tool" becomes a user message prefixed "Tool result:" (no tool
      role exists in the chat template).
    - The tool-call format note is appended to the system prompt.
    - Consecutive same-role messages are merged (some templates reject
      or mishandle them).
    - system_as_user=True folds the system text into the first user
      message (fallback for templates without a system role).
    """
    messages: list[dict] = []
    system_text = ""
    for msg in transcript:
        role, content = msg["role"], msg["content"]
        if role == "system":
            system_text = (content + "\n\n" + TOOL_FORMAT_NOTE).strip()
            continue
        if role == "tool":
            role, content = "user", "Tool result:\n" + content
        if role not in ("user", "assistant"):
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n" + content
        else:
            messages.append({"role": role, "content": content})
    if system_text:
        if system_as_user:
            if messages and messages[0]["role"] == "user":
                messages[0]["content"] = system_text + "\n\n" + messages[0]["content"]
            else:
                messages.insert(0, {"role": "user", "content": system_text})
        else:
            messages.insert(0, {"role": "system", "content": system_text})
    return messages


def parse_agent_response(text: str) -> AgentResponse:
    """Split raw model output into (plain text, parsed tool calls).

    Recognises the canonical `[tool_call name {json}]` rendering
    anywhere in the output; unparseable arg blobs degrade to {} rather
    than failing the run (the tool NAME is what the judge scores).
    """
    calls = []

    def _grab(m: re.Match) -> str:
        args = {}
        if m.group("args"):
            try:
                parsed = json.loads(m.group("args"))
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {}
        calls.append(ToolCall(m.group("name"), args))
        return ""

    cleaned = _TOOL_CALL_RE.sub(_grab, text).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return AgentResponse(text=cleaned, tool_calls=tuple(calls))


class TinyAyaEngine:
    """Lazy transformers loader + greedy generator for Tiny Aya.

    Construction is cheap; weights load on the first generate() call
    and are cached on the instance (one load per notebook/bench run).
    """

    def __init__(self, model_id: str = MODEL_ID,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401  (presence check; used via model)
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise NotImplementedError(
                "Tiny Aya needs torch + transformers, which are not "
                "installed here. Run on a GPU environment (Kaggle "
                "notebook: kaggle/tiny_aya_bench.ipynb) — the offline "
                "machine deliberately never downloads weights."
            ) from exc
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGINGFACE_TOKEN")
        if not token:
            try:  # Kaggle Secrets, when running inside a Kaggle notebook
                from kaggle_secrets import UserSecretsClient

                token = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                token = None
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=token)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto",
            token=token)
        self._model.eval()

    def generate(self, transcript: list[dict]) -> str:
        """One completion for the running transcript (full history is
        re-rendered each turn, matching the stateless adapter API)."""
        self._load()
        import torch

        messages = transcript_to_messages(transcript)
        try:
            encoded = self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt")
        except Exception:
            # Template without a system role: fold system into user #1.
            messages = transcript_to_messages(transcript, system_as_user=True)
            encoded = self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt")
        input_ids = encoded["input_ids"] if not torch.is_tensor(encoded) else encoded
        input_ids = input_ids.to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

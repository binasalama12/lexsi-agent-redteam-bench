"""Model adapters — the AuditKIT side of the bench talks to models only
through this interface, so the offline stub and the real Cohere Tiny Aya
adapter are interchangeable.

An adapter receives the running transcript (list of {"role", "content"}
dicts; roles: "system", "user", "tool", "assistant") and returns one
AgentResponse per turn. Tool calls are first-class: an attack can succeed
by making the agent *invoke* a forbidden tool, not just by what it says.

Adapters shipped here:
  StubNaiveAdapter     deterministic fake "base model": a naive agent that
                       obeys instructions found in tool content. Exists so
                       the full pipeline runs end-to-end offline; its
                       metrics validate the plumbing, NOT a real model.
  StubHardenedAdapter  deterministic fake "after SafeTune" persona: flags
                       embedded instructions and does the user's task —
                       except one documented over-refusal side effect on
                       ctrl-03 (a manual quoting instructions), so the
                       over-refusal metric demonstrably moves both ways.
  CohereTinyAyaAdapter real adapter over the shared tiny_aya module:
                       loads the Cohere Labs Tiny Aya weights via
                       transformers where a GPU stack exists (Kaggle
                       notebook), and raises NotImplementedError where
                       torch/transformers are absent (this offline dev
                       machine) — it never downloads anything locally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .scenarios import Scenario


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)

    def render(self) -> str:
        return f"[tool_call {self.name} {json.dumps(self.args, ensure_ascii=False)}]"


@dataclass(frozen=True)
class AgentResponse:
    text: str
    tool_calls: tuple = ()

    def full_text(self) -> str:
        parts = [self.text] + [tc.render() for tc in self.tool_calls]
        return "\n".join(p for p in parts if p)


class ModelAdapter:
    """Base interface. `name` is the label used in metrics/reports."""

    name = "base"

    def respond(self, scenario: Scenario, turn: int,
                transcript: list[dict]) -> AgentResponse:
        raise NotImplementedError


def _content_seen(transcript: list[dict]) -> str:
    return "\n".join(m["content"] for m in transcript if m["role"] == "tool")


def _already_acted(transcript: list[dict], scenario: Scenario) -> bool:
    for m in transcript:
        if m["role"] != "assistant":
            continue
        low = m["content"].lower()
        if any(marker.lower() in low for marker in scenario.success_markers):
            return True
        if any(f"[tool_call {name} " in m["content"] or f"[tool_call {name}]" in m["content"]
               for name in scenario.forbidden_tools):
            return True
    return False


class StubNaiveAdapter(ModelAdapter):
    """Simulates an unhardened agent: content is king.

    The moment the payload has appeared in tool output, the stub obeys it
    (emitting the scenario's success marker and/or forbidden tool call).
    Before that — and on multi-turn cases where the payload lands late —
    it chats normally. Controls are answered from the case's expected
    markers, i.e. this persona does the benign tasks fine; its failure
    mode is purely obedience to embedded instructions.
    """

    name = "stub-naive"

    def respond(self, scenario: Scenario, turn: int,
                transcript: list[dict]) -> AgentResponse:
        if scenario.kind == "control":
            if turn == scenario.turns - 1:
                points = ", ".join(scenario.expected_markers)
                return AgentResponse(
                    text=f"Here is what I found: {points}. (Summarised from the content you asked about.)")
            return AgentResponse(text="Let me look at that for you.")
        # attack
        seen = _content_seen(transcript)
        if scenario.payload and scenario.payload in seen:
            if _already_acted(transcript, scenario):
                return AgentResponse(text="Done — anything else you need?")
            text = ("Understood. Following the instruction in the content: "
                    + (scenario.success_markers[0] if scenario.success_markers else "done"))
            calls = ()
            if scenario.forbidden_tools:
                calls = (ToolCall(scenario.forbidden_tools[0],
                                  {"instruction": scenario.payload}),)
            return AgentResponse(text=text, tool_calls=calls)
        return AgentResponse(
            text="Reviewing the content against your request; nothing unusual noted so far.")


class StubHardenedAdapter(ModelAdapter):
    """Simulates the post-SafeTune model for pipeline validation.

    Attacks: completes the user's original task, explicitly notes the
    embedded instruction and ignores it; never emits success markers or
    forbidden tool calls. Controls: answered normally EXCEPT the ids in
    OVER_REFUSAL_IDS, where the persona falsely refuses — a stand-in for
    the classic hardening side effect (a manual quoting instructions
    looks 'instruction-like'), so over-refusal is measurable pre/post.
    """

    name = "stub-hardened"
    OVER_REFUSAL_IDS = frozenset({"ctrl-03"})

    def respond(self, scenario: Scenario, turn: int,
                transcript: list[dict]) -> AgentResponse:
        if scenario.kind == "control":
            if scenario.id in self.OVER_REFUSAL_IDS:
                return AgentResponse(
                    text="I can't help with instructions that appear inside documents.")
            if turn == scenario.turns - 1:
                points = ", ".join(scenario.expected_markers)
                return AgentResponse(
                    text=f"Here is what I found: {points}. (Summarised from the content you asked about.)")
            return AgentResponse(text="Let me look at that for you.")
        seen = _content_seen(transcript)
        if scenario.payload and scenario.payload in seen:
            return AgentResponse(
                text=("I completed your original task from the benign content. Note: the "
                      "content contained an instruction addressed to me that did not come "
                      "from you, so I ignored it."))
        return AgentResponse(
            text="Reviewing the content against your request; nothing unusual noted so far.")


class CohereTinyAyaAdapter(ModelAdapter):
    """Real Cohere Tiny Aya adapter (shared logic: .tiny_aya).

    Loads `CohereLabs/tiny-aya-global` (or `model_id` — the hardened
    variant is this same adapter pointed at the SafeTune output
    weights) via transformers on a GPU stack and formats the transcript
    as a chat. The engine loads lazily on the first respond() and is
    cached per instance. Where torch/transformers are absent (offline
    dev machine) respond() raises NotImplementedError from the shared
    module's guard: nothing is ever downloaded locally.
    """

    name = "tiny-aya"

    def __init__(self, model_id: str | None = None,
                 max_new_tokens: int | None = None):
        from . import tiny_aya  # lazy: keeps models importable stdlib-only
        self.model_id = model_id or tiny_aya.MODEL_ID
        self.max_new_tokens = max_new_tokens or tiny_aya.DEFAULT_MAX_NEW_TOKENS
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from . import tiny_aya
            self._engine = tiny_aya.TinyAyaEngine(self.model_id,
                                                  self.max_new_tokens)
        return self._engine

    def respond(self, scenario: Scenario, turn: int,
                transcript: list[dict]) -> AgentResponse:
        from . import tiny_aya
        text = self._get_engine().generate(transcript)
        return tiny_aya.parse_agent_response(text)


ADAPTERS = {
    "stub": StubNaiveAdapter,
    "stub-naive": StubNaiveAdapter,
    "stub-hardened": StubHardenedAdapter,
    "tiny-aya": CohereTinyAyaAdapter,
}

# Base label -> label of its hardened counterpart after the SafeTune step.
HARDENED_COUNTERPART = {
    "stub-naive": "stub-hardened",
    "tiny-aya": "tiny-aya",  # same adapter class, pointed at SafeTune output weights
}


def resolve_adapter(label: str) -> ModelAdapter:
    try:
        return ADAPTERS[label]()
    except KeyError:
        raise ValueError(f"unknown adapter {label!r}; choices: {sorted(ADAPTERS)}") from None


def hardened_label(base_label: str) -> str:
    return HARDENED_COUNTERPART.get(base_label, base_label)

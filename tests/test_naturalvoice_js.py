"""Runs the chat page's browser-voice tests (tests/js/naturalvoice.test.cjs) with node, so the existing pytest job covers
them too -- there is no separate JavaScript test job. They test static/naturalvoice.js (finding a natural browser voice and
speaking with it without getting stuck) against fake browser objects and a fake clock; no browser is involved.

Skipped, with the reason shown, on a machine without node: a Python-only checkout should not fail for that. CI's runners
have node, so there it always runs."""

import shutil
import subprocess
from pathlib import Path

import pytest

_NODE = shutil.which("node")
_JS_TEST = Path(__file__).resolve().parent / "js" / "naturalvoice.test.cjs"


@pytest.mark.skipif(_NODE is None, reason="node is not installed; tests/js/naturalvoice.test.cjs was not run")
def test_the_browser_voice_module_passes_its_node_tests():
    result = subprocess.run([_NODE, "--test", str(_JS_TEST)], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, f"node --test failed:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}"
    assert "# fail 0" in result.stdout
    assert "# pass 0" not in result.stdout  # a run that found no tests must not count as a pass

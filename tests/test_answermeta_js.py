"""Runs the tests of the line under a reply (tests/js/answermeta.test.cjs) with node, so the existing pytest job covers
them too, the same way tests/test_richtext_js.py runs the reply-formatting tests. They test static/answermeta.js (how long
the server took and which tools it used, shown small under a reply); no browser is involved.

Skipped, with the reason shown, on a machine without node. CI's runners have node, so there it always runs."""

import shutil
import subprocess
from pathlib import Path

import pytest

_NODE = shutil.which("node")
_JS_TEST = Path(__file__).resolve().parent / "js" / "answermeta.test.cjs"


@pytest.mark.skipif(_NODE is None, reason="node is not installed; tests/js/answermeta.test.cjs was not run")
def test_the_line_under_a_reply_passes_its_node_tests():
    result = subprocess.run([_NODE, "--test", str(_JS_TEST)], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, f"node --test failed:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}"
    assert "# fail 0" in result.stdout
    assert "# pass 0" not in result.stdout  # a run that found no tests must not count as a pass

"""The conversation lab as a CI gate (eval/conversation_lab.py): every labelled case replayed through the real agent
loop, tools, guardrails and database, with the model's side fixed by its cassette -- no API key, no quota. A failure
here means something around the model broke a labelled behaviour (the read-back gate, a guardrail, account scoping),
not that the model had a bad day.
"""

import pytest

from eval.conversation_lab import CASES, run_case


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_the_labelled_case_holds(case, reseed_accounts):
    result = run_case(case)
    problems = [f"turn {i}: {p}" for i, t in enumerate(result["turns"], 1) for p in t["problems"]]
    assert result["passed"], f"{case.case_id} ({result['source']}; label: {case.label_source}): " + "; ".join(problems)


def test_ops_quality_shows_the_lab_and_the_nightly_suites(reseed_accounts, tmp_path, monkeypatch):
    import json
    import os

    from fastapi.testclient import TestClient

    from businessflow.ops import api

    (tmp_path / "conversation_lab.json").write_text(json.dumps({"summary": {"cases": 1, "passed": 1}, "results": []}), encoding="utf-8")
    (tmp_path / "red_team_history.jsonl").write_text('{"recorded_at": "2026-10-22T00:00:00+00:00", "summary": {"pass_rate": 1.0}}\n', encoding="utf-8")
    monkeypatch.setattr(api, "_EVAL_RESULTS_DIR", tmp_path)

    q = TestClient(api.app).get("/quality", headers={"X-API-Key": os.environ["OPS_API_KEY"]}).json()

    assert q["labs"] == [{"summary": {"cases": 1, "passed": 1}, "results": []}]
    red = next(n for n in q["nightly"] if n["suite"] == "red_team")
    assert red["last_run"]["summary"] == {"pass_rate": 1.0}
    assert next(n for n in q["nightly"] if n["suite"] == "latency_benchmark")["last_run"] is None

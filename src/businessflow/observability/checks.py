"""The rules every traced step is graded by, with no model: what each AI component should always get right, written as
a check over the numbers a span recorded (observability/trace.py).

A rule here is cheap, deterministic and explainable, so it runs on every real turn. Each one says what it means and
what to do when it fires -- a failure count nobody knows how to act on is noise. What rules cannot judge (helpfulness,
tone) is left to the nightly model-graded suites on fixed inputs.

Thresholds are deliberately conservative: a check firing should be worth a look, not routine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from businessflow.rag.tokenize import dominant_script


@dataclass(frozen=True)
class Check:
    name: str
    component: str  # what the report groups it under
    means: str
    action: str
    rule: Callable[[dict], bool]  # span (or turn) -> True when the check FAILS
    span: str | None  # the span name it applies to; None = a whole-turn check (gets the Trace)
    critical: bool = False  # a failure here is red in the report however rare
    # What kind of problem it is, so speed and capacity never pass for wrong answers:
    #   correctness -- the AI step got something wrong (counts against the component's health)
    #   speed       -- right, but slow
    #   capacity    -- waited on a quota or limit outside the AI (Groq's rate limit)
    #   protection  -- a safeguard stopped something it should (an unverified chat asking for an account): it worked
    kind: str = "correctness"


_WORD = re.compile(r"[A-Za-z]{5,}")
_STOP = {"which", "their", "there", "about", "would", "could", "should", "please", "thank", "these", "those", "where",
         "account", "payment", "borrower", "rupees"}

# Whisper's own convention: a segment whose average log-probability is below -1 is a likely mis-hearing.
STT_MIN_AVG_LOGPROB = -1.0
STT_MAX_NO_SPEECH = 0.6
STT_MIN_CHARS_PER_SECOND = 2.0
RERANK_UNSURE_MARGIN = 0.5  # cross-encoder logits: a top-two gap under half a logit is a near tie
TTS_MIN_SECONDS = 0.3  # eval/voice_naturalness_benchmark.py's plausibility bounds
TTS_MAX_SECONDS_PER_CHAR = 0.5
SLOW_MODEL_SECONDS = 20.0
SLOW_TOOL_SECONDS = 10.0
SLOW_TTS_SECONDS = 8.0


def _a(span: dict) -> dict:
    return span.get("attrs") or {}


def _stt_wrong_script(span: dict) -> bool:
    a = _a(span)
    script = a.get("script")
    if not a.get("chars") or script in (None, "none", "mixed"):
        return False
    return (a.get("language") == "hi" and script == "latin") or (a.get("language") == "en" and script == "devanagari")


def _tts_implausible(span: dict) -> bool:
    a = _a(span)
    if a.get("audio_seconds") is None or not a.get("chars"):
        return False
    return not (TTS_MIN_SECONDS <= a["audio_seconds"] <= TTS_MAX_SECONDS_PER_CHAR * a["chars"])


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "")} - _STOP


def _retrieval_unused(t) -> bool:
    """check_policy returned passages, the reply went out, and it shares no content word with any of them. English
    only: a Hindi reply to English policy text shares no words by nature, so it is not judged this way."""
    passages, reply = t.scratch.get("retrieved_texts"), t.scratch.get("reply")
    if not passages or not reply or t.scratch.get("guardrail_blocked") or dominant_script(reply) != "latin":
        return False
    words = set().union(*(_content_words(p) for p in passages))
    return bool(words) and not (_content_words(reply) & words)


def _reply_wrong_language(t) -> bool:
    """The borrower wrote in Devanagari and the reply came back in Latin script, or the other way round. Hinglish
    (Latin) asked, Latin answered, is fine."""
    asked, reply = t.scratch.get("user_text"), t.scratch.get("reply")
    if not asked or not reply or len(reply) < 30 or t.scratch.get("guardrail_blocked"):
        return False
    a, r = dominant_script(asked), dominant_script(reply)
    return (a == "devanagari" and r == "latin") or (a == "latin" and r == "devanagari")


CHECKS: list[Check] = [
    # Speech-to-text: is what we heard what they said?
    Check("stt.empty", "Speech-to-text", "A voice note produced no words.",
          "Listen to a few: if they had speech, the voice-activity trim or the model is losing it.",
          lambda s: not _a(s).get("chars"), "stt"),
    Check("stt.low_confidence", "Speech-to-text", "Whisper was unsure of what it heard (average log-probability below -1).",
          "Compare those transcripts with the audio; a rise means noisier audio or an accent the model handles badly.",
          lambda s: _a(s).get("avg_logprob") is not None and _a(s)["avg_logprob"] < STT_MIN_AVG_LOGPROB, "stt"),
    Check("stt.mostly_silence", "Speech-to-text", "Whisper judged the clip to be mostly not speech.",
          "Check the recording path (mic permission, a muted or cut-off note) before the model.",
          lambda s: (_a(s).get("no_speech_prob") or 0) > STT_MAX_NO_SPEECH, "stt"),
    Check("stt.wrong_script", "Speech-to-text", "The transcript is in the wrong script for the language asked for.",
          "The language hint is wrong for this borrower, or the model is transliterating; check their language setting.",
          _stt_wrong_script, "stt"),
    Check("stt.too_sparse", "Speech-to-text", "Very few words for the length of audio (under 2 characters a second).",
          "Words are being dropped; listen to the clip and compare.",
          lambda s: (_a(s).get("audio_seconds") or 0) >= 3 and (_a(s).get("chars") or 0) / _a(s)["audio_seconds"] < STT_MIN_CHARS_PER_SECOND,
          "stt"),
    # The model: did it answer cleanly and in time?
    Check("llm.malformed_tool_call", "Model calls", "The model produced a malformed tool call and had to be asked again.",
          "Occasional is expected; a rise means a prompt or model change broke tool-call formatting.",
          lambda s: (_a(s).get("malformed_retries") or 0) > 0, "llm"),
    Check("llm.rate_limited", "Model calls", "The call waited on Groq's rate limit.",
          "The shared daily quota is running out -- nightly evals or a burst of traffic; spread them or add a key.",
          lambda s: (_a(s).get("rate_limit_waits") or 0) > 0, "llm", kind="capacity"),
    Check("llm.slow", "Model calls", f"One model call took over {SLOW_MODEL_SECONDS:.0f} seconds.",
          "Check Groq status and the rate-limit check above; a long conversation history also slows it.",
          lambda s: s.get("seconds", 0) > SLOW_MODEL_SECONDS, "llm", kind="speed"),
    # Tools: did the action it chose actually go through?
    Check("tool.error", "Tools", "A tool the model chose returned an error.",
          "Read the error: bad arguments from the model (prompt) or a real fault in the tool (code).",
          lambda s: _a(s).get("outcome") == "error", "tool"),
    Check("tool.blocked", "Tools", "A tool call was refused: another account, or an unverified chat.",
          "Expected under attack; if it happens in ordinary chats the model is confusing accounts -- read the turn.",
          lambda s: _a(s).get("outcome") == "blocked", "tool", kind="protection"),
    Check("tool.slow", "Tools", f"A tool took over {SLOW_TOOL_SECONDS:.0f} seconds.",
          "Usually the database (Sydney) or a cold retrieval model; check which tool.",
          lambda s: s.get("seconds", 0) > SLOW_TOOL_SECONDS, "tool", kind="speed"),
    # Retrieval: did it find something, and was it sure?
    Check("retrieval.empty", "Retrieval", "A policy search found nothing.",
          "The question is outside the knowledge base -- either add the document, or the agent should hand off.",
          lambda s: _a(s).get("results") == 0, "retrieval"),
    Check("retrieval.unsure", "Retrieval", "The top two passages scored almost the same (a near tie).",
          "The answer may come from the wrong passage; look at the question and the two documents.",
          lambda s: _a(s).get("kind") == "rerank" and _a(s).get("margin") is not None and _a(s)["margin"] < RERANK_UNSURE_MARGIN,
          "retrieval"),
    Check("retrieval.unused", "Retrieval", "Policy passages were found but the reply used none of their words.",
          "The model answered from its own knowledge instead of the policy -- read the turn.",
          _retrieval_unused, None),
    # The reply: true, and in the borrower's language?
    Check("reply.blocked", "Reply truthfulness", "The guardrail stopped a reply (an invented amount, link or claim).",
          "Caught, so the borrower was safe -- but each one is the model saying something untrue; read which check.",
          lambda s: _a(s).get("blocked") is True, "guardrail", critical=True),
    Check("reply.wrong_language", "Reply truthfulness", "The reply came back in a different script from the borrower's message.",
          "Check the language instruction in the prompt for that borrower's setting.",
          _reply_wrong_language, None),
    # Speech out: will it sound right?
    Check("tts.symbols_left", "Text-to-speech", "Symbols or digits reached the voice unconverted (it would read them out).",
          "Extend audio/verbalizer.py for that pattern.",
          lambda s: _a(s).get("unverbalized") is True, "tts"),
    Check("tts.implausible_length", "Text-to-speech", "The audio is far too short or too long for the text.",
          "The engine dropped or garbled text; listen to it.",
          _tts_implausible, "tts"),
    Check("tts.slow", "Text-to-speech", f"Speaking one reply took over {SLOW_TTS_SECONDS:.0f} seconds.",
          "Long replies, or the Hindi engine fell back to the slower one; check the engine recorded.",
          lambda s: s.get("seconds", 0) > SLOW_TTS_SECONDS, "tts", kind="speed"),
    # Outbound writer: did its draft pass first time?
    Check("compose.redrafted", "Reminder writer", "A reminder draft failed its check and was rewritten.",
          "Read the problem recorded: an invented figure or a claim the model may not make.",
          lambda s: (_a(s).get("attempts") or 1) > 1, "compose"),
    Check("compose.fallback", "Reminder writer", "Two drafts failed, so the fixed wording was sent.",
          "The model cannot write this reminder truthfully; look at the facts it was given.",
          lambda s: _a(s).get("fallback") is True, "compose", critical=True),
    # The agent loop as a whole.
    Check("agent.round_cap", "Model calls", "The model kept calling tools until the round cap and was forced to answer.",
          "A stuck conversation; read it -- usually a tool error it kept retrying.",
          lambda s: bool(_a(s).get("hit_round_cap")), "agent"),
]

BY_NAME = {c.name: c for c in CHECKS}


def grade(t) -> dict:
    """{span index: [failed check names], "turn": [failed whole-turn check names]} for a Trace."""
    out: dict = {}
    for i, s in enumerate(t.spans):
        failed = [c.name for c in CHECKS if c.span == s["name"] and c.rule(s)]
        if failed:
            out[i] = failed
    turn_failed = [c.name for c in CHECKS if c.span is None and c.rule(t)]
    if turn_failed:
        out["turn"] = turn_failed
    return out

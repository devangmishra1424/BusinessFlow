/* ============================================================
   The small line under a reply: how long it took to answer and what was used to get there, in the way a search shows
   "(0.43 seconds)".

     Answered in 3.1 s · heard 1.2 s · model 1.7 s · Checked our policy 0.41 s · Checked your account 0.04 s

   When the free AI plan's per-minute allowance ran out and the reply had to wait for it to refill, that wait is its own step
   so nobody reads it as the model being slow:

     Answered in 2 min 5 s · model 4.90 s (4 calls) · waited 2 min for the AI service's rate limit

   The numbers come from the server (the `timing` field of a message response: observability/trace.py's answer_timing),
   measured from the request arriving to the reply being ready. They do not include the network to this page or
   speaking the reply aloud, so the line says what the server spent, not what the person waited.

   Tool names are shown as the plain-language labels the page already uses (app.js TOOL_LABELS); the raw name is in the
   tooltip. A tool that did not go through is marked and says so in its tooltip. A server that sends no `timing` (an
   older one, or a reply written without the model) still shows the tools used, without times.

   The dots and spaces between the steps are real text, not CSS: a phone that still has an older stylesheet cached
   (the line shipped with its separators in the stylesheet, and the stylesheet's version number was not changed) used to
   run the steps together as "0.41 smodel 1.70 s". Now the line reads correctly even with no styling at all.

   SAFE BY CONSTRUCTION: every piece of text is HTML-escaped, and the only tags in the result are the ones written here.
   ============================================================ */
const AnswerMeta = (() => {
  const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);

  // 0.04 s, 2.41 s, 12.3 s; from 100 s up (99.95 rounds there), minutes: 2 min 5 s. Two decimals while they matter, one once a reply is
  // long enough that they do not.
  function seconds(value) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "";
    if (value > 0 && value < 0.01) return "<0.01 s";
    if (value >= 99.95) {
      const whole = Math.round(value);
      const minutes = Math.floor(whole / 60);
      const rest = whole % 60;
      return rest ? `${minutes} min ${rest} s` : `${minutes} min`;
    }
    return `${value < 10 ? value.toFixed(2) : value.toFixed(1)} s`;
  }

  const OUTCOME_WORDS = { error: "did not go through", blocked: "was not allowed" };
  const WAIT_TITLE = "The AI service allows a fixed amount of text per minute. The reply was ready sooner; this is time spent "
    + "waiting for that allowance to refill, so it is not the model being slow.";
  // Between two steps: a no-break dot with a real space either side, then a place the line may break. Plain text, so it
  // does not depend on any stylesheet.
  const SEPARATOR = '<span class="answer-sep" aria-hidden="true">&nbsp;·&nbsp;</span><wbr>';

  function step(cls, title, text) {
    return `<span class="${cls}" title="${escapeHtml(title)}">${escapeHtml(text)}</span>`;
  }

  function toolStep(tool, timeText, outcome, labels) {
    const label = labels[tool] || tool;
    const failed = outcome && outcome !== "ok";
    const title = failed ? `${tool}: ${OUTCOME_WORDS[outcome] || outcome}` : tool;
    const text = [failed ? "⚠" : "", label, timeText].filter(Boolean).join(" ");
    return step(`answer-step${failed ? " failed" : ""}`, title, text);
  }

  /** The line as HTML, or "" when there is nothing to show (no timing and no tools). `labels` maps a tool's name to
   *  its plain-language label. */
  function html(timing, toolCalls, labels = {}) {
    const steps = [];
    if (timing && seconds(timing.seconds)) {
      steps.push(`<span class="answer-time">Answered in ${escapeHtml(seconds(timing.seconds))}</span>`);
      if (timing.listen_seconds != null && seconds(timing.listen_seconds)) {
        steps.push(step("answer-step", "speech-to-text", `heard ${seconds(timing.listen_seconds)}`));
      }
      if (timing.model_calls > 0 && seconds(timing.model_seconds)) {
        const calls = timing.model_calls > 1 ? ` (${timing.model_calls} calls)` : "";
        steps.push(step("answer-step", "the language model", `model ${seconds(timing.model_seconds)}${calls}`));
      }
      if (typeof timing.wait_seconds === "number" && timing.wait_seconds >= 1 && seconds(timing.wait_seconds)) {
        steps.push(step("answer-step answer-wait", WAIT_TITLE, `waited ${seconds(timing.wait_seconds)} for the AI service's rate limit`));
      }
      for (const t of timing.tools || []) steps.push(toolStep(t.tool, seconds(t.seconds), t.outcome, labels));
    } else {
      // No timing sent: the tools that ran, as before, with no times.
      for (const t of toolCalls || []) steps.push(toolStep(t.tool, "", "ok", labels));
    }
    return steps.length ? `<div class="answer-meta">${steps.join(SEPARATOR)}</div>` : "";
  }

  return { escapeHtml, seconds, html };
})();

// Node (the tests) loads this file as a module; in the browser it is a plain script and `AnswerMeta` is a global.
if (typeof module !== "undefined" && module.exports) module.exports = AnswerMeta;

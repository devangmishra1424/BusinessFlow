/* ============================================================
   The small line under a reply: how long it took to answer and what was used to get there, in the way a search shows
   "(0.43 seconds)".

     Answered in 3.1 s · heard 1.2 s · model 1.7 s · Checked our policy 0.41 s · Checked your account 0.04 s

   The numbers come from the server (the `timing` field of a message response: observability/trace.py's answer_timing),
   measured from the request arriving to the reply being ready. They do not include the network to this page or
   speaking the reply aloud, so the line says what the server spent, not what the person waited.

   Tool names are shown as the plain-language labels the page already uses (app.js TOOL_LABELS); the raw name is in the
   tooltip. A tool that did not go through is marked and says so in its tooltip. A server that sends no `timing` (an
   older one, or a reply written without the model) still shows the tools used, without times.

   SAFE BY CONSTRUCTION: every piece of text is HTML-escaped, and the only tags in the result are the ones written here.
   ============================================================ */
const AnswerMeta = (() => {
  const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);

  // 0.04 s, 2.41 s, 12.3 s -- two decimals while they matter, one once a reply is long enough that they do not.
  function seconds(value) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "";
    if (value > 0 && value < 0.01) return "<0.01 s";
    return `${value < 10 ? value.toFixed(2) : value.toFixed(1)} s`;
  }

  const OUTCOME_WORDS = { error: "did not go through", blocked: "was not allowed" };

  function toolStep(tool, timeText, outcome, labels) {
    const label = labels[tool] || tool;
    const failed = outcome && outcome !== "ok";
    const title = failed ? `${tool}: ${OUTCOME_WORDS[outcome] || outcome}` : tool;
    const text = [failed ? "⚠" : "", label, timeText].filter(Boolean).join(" ");
    return `<span class="answer-step${failed ? " failed" : ""}" title="${escapeHtml(title)}">${escapeHtml(text)}</span>`;
  }

  /** The line as HTML, or "" when there is nothing to show (no timing and no tools). `labels` maps a tool's name to
   *  its plain-language label. */
  function html(timing, toolCalls, labels = {}) {
    const steps = [];
    if (timing && seconds(timing.seconds)) {
      steps.push(`<span class="answer-time">Answered in ${escapeHtml(seconds(timing.seconds))}</span>`);
      if (timing.listen_seconds != null && seconds(timing.listen_seconds)) {
        steps.push(`<span class="answer-step" title="speech-to-text">heard ${escapeHtml(seconds(timing.listen_seconds))}</span>`);
      }
      if (timing.model_calls > 0 && seconds(timing.model_seconds)) {
        const calls = timing.model_calls > 1 ? ` (${timing.model_calls} calls)` : "";
        steps.push(`<span class="answer-step" title="the language model">model ${escapeHtml(seconds(timing.model_seconds))}${escapeHtml(calls)}</span>`);
      }
      for (const t of timing.tools || []) steps.push(toolStep(t.tool, seconds(t.seconds), t.outcome, labels));
    } else {
      // No timing sent: the tools that ran, as before, with no times.
      for (const t of toolCalls || []) steps.push(toolStep(t.tool, "", "ok", labels));
    }
    return steps.length ? `<div class="answer-meta">${steps.join("")}</div>` : "";
  }

  return { escapeHtml, seconds, html };
})();

// Node (the tests) loads this file as a module; in the browser it is a plain script and `AnswerMeta` is a global.
if (typeof module !== "undefined" && module.exports) module.exports = AnswerMeta;

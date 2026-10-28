// Run with: node --test tests/js/answermeta.test.cjs   (tests/test_answermeta_js.py runs it as part of pytest)
//
// The small line under a reply (static/answermeta.js): how long the server took and which tools it used. What this
// cannot show is how the line looks on the page; the numbers themselves are tests/test_answer_timing.py.
const test = require("node:test");
const assert = require("node:assert/strict");
const AnswerMeta = require("../../src/businessflow/channels/static/answermeta.js");

const { html, seconds } = AnswerMeta;
const LABELS = { check_policy: "Checked our policy", get_payment_status: "Checked your account" };
const text = (h) => h.split(/<[^>]+>/).filter(Boolean).join(" | ");

test("times read like a search's: two decimals while short, one once long, and never a made-up zero", () => {
  assert.equal(seconds(0.043), "0.04 s");
  assert.equal(seconds(2.414), "2.41 s");
  assert.equal(seconds(12.34), "12.3 s");
  assert.equal(seconds(0.002), "<0.01 s");
  assert.equal(seconds(0), "0.00 s");
  for (const bad of [undefined, null, NaN, -1, "3", Infinity]) assert.equal(seconds(bad), "", String(bad));
});

test("a text turn shows the total, the model and each tool with its own time", () => {
  const h = html(
    { seconds: 2.41, listen_seconds: null, model_seconds: 1.7, model_calls: 2, tools: [{ tool: "check_policy", seconds: 0.41, outcome: "ok" }] },
    [{ tool: "check_policy", arguments: {} }], LABELS,
  );
  assert.equal(text(h), "Answered in 2.41 s | model 1.70 s (2 calls) | Checked our policy 0.41 s");
  assert.ok(h.includes('title="check_policy"'), h); // the raw name is in the tooltip
  assert.ok(!h.includes("heard"), h);
});

test("a voice turn also shows how long listening took", () => {
  const h = html({ seconds: 3.1, listen_seconds: 1.2, model_seconds: 1.7, model_calls: 1, tools: [] }, [], LABELS);
  assert.equal(text(h), "Answered in 3.10 s | heard 1.20 s | model 1.70 s");
});

test("a tool that did not go through is marked, and says so in its tooltip", () => {
  const h = html({ seconds: 1, model_seconds: 0.5, model_calls: 1, tools: [{ tool: "get_payment_status", seconds: 0.02, outcome: "blocked" }] }, [], LABELS);
  assert.ok(h.includes('class="answer-step failed"'), h);
  assert.ok(h.includes("⚠ Checked your account 0.02 s"), h);
  assert.ok(h.includes('title="get_payment_status: was not allowed"'), h);
});

test("a tool with no label is shown by its own name", () => {
  const h = html({ seconds: 1, model_seconds: 0.5, model_calls: 1, tools: [{ tool: "brand_new_tool", seconds: 0.1, outcome: "ok" }] }, [], LABELS);
  assert.ok(h.includes("brand_new_tool 0.10 s"), h);
});

test("with no timing from the server the tools still show, without times", () => {
  const h = html(null, [{ tool: "check_policy", arguments: {} }, { tool: "get_payment_status", arguments: {} }], LABELS);
  assert.equal(text(h), "Checked our policy | Checked your account");
  assert.ok(!h.includes(" s<"), h);
});

test("nothing to show gives an empty string, not an empty line", () => {
  assert.equal(html(null, [], LABELS), "");
  assert.equal(html(undefined, undefined, LABELS), "");
  assert.equal(html({ seconds: "soon", model_seconds: 1, model_calls: 1, tools: [] }, [], LABELS), "");
});

test("a turn that made no model call does not claim one", () => {
  const h = html({ seconds: 0.2, listen_seconds: null, model_seconds: 0, model_calls: 0, tools: [] }, [], LABELS);
  assert.equal(text(h), "Answered in 0.20 s");
});

test("text from the server cannot add HTML of its own", () => {
  const h = html({ seconds: 1, model_seconds: 0.5, model_calls: 1, tools: [{ tool: '<img src=x onerror="alert(1)">', seconds: 0.1, outcome: "ok" }] }, [], {});
  assert.ok(!h.includes("<img"), h);
  assert.ok(h.includes("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;"), h);
});

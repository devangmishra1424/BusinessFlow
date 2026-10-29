// Run with: node --test tests/js/answermeta.test.cjs   (tests/test_answermeta_js.py runs it as part of pytest)
//
// The small line under a reply (static/answermeta.js): how long the server took and which tools it used. What this
// cannot show is how the line looks on the page; the numbers themselves are tests/test_answer_timing.py.
const test = require("node:test");
const assert = require("node:assert/strict");
const AnswerMeta = require("../../src/businessflow/channels/static/answermeta.js");

const { html, seconds } = AnswerMeta;
const LABELS = { check_policy: "Checked our policy", get_payment_status: "Checked your account" };
// The steps as text, one per " | " (the dots between steps are markup of their own, each ending in a <wbr>).
const text = (h) => h.split("<wbr>").map((part) => part.replace(/<[^>]+>/g, "").replace(/&nbsp;·&nbsp;/g, "").replace(/&#39;/g, "'")).join(" | ");

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

test("from 100 s up the time is in minutes, so two minutes does not read as '125.0 s'", () => {
  assert.equal(seconds(99.4), "99.4 s");
  assert.equal(seconds(99.96), "1 min 40 s");
  assert.equal(seconds(125.3), "2 min 5 s");
  assert.equal(seconds(120), "2 min");
  assert.equal(seconds(3725), "62 min 5 s");
});

test("with no stylesheet at all the line still reads with a space and a dot between the steps", () => {
  // The line once shipped with its dots in the stylesheet while the stylesheet's version number stayed the same; a phone with
  // the old file cached showed "0.41 smodel 1.70 s". The separators are text now, so this must read right unstyled.
  const h = html(
    { seconds: 2.41, listen_seconds: 0.6, model_seconds: 1.7, model_calls: 2, tools: [{ tool: "check_policy", seconds: 0.41, outcome: "ok" }, { tool: "get_payment_status", seconds: 0.04, outcome: "ok" }] },
    [], LABELS,
  );
  const plain = h.replace(/<[^>]+>/g, "").replace(/&nbsp;/g, " ");
  assert.equal(plain, "Answered in 2.41 s · heard 0.60 s · model 1.70 s (2 calls) · Checked our policy 0.41 s · Checked your account 0.04 s");
  assert.ok(!h.includes("::before"), h);
});

test("time spent waiting for the AI service's rate limit is its own step, not part of the model's time", () => {
  const h = html({ seconds: 125.3, model_seconds: 4.9, wait_seconds: 118, model_calls: 4, tools: [{ tool: "check_policy", seconds: 0.4, outcome: "ok" }] }, [], LABELS);
  assert.equal(text(h), "Answered in 2 min 5 s | model 4.90 s (4 calls) | waited 1 min 58 s for the AI service's rate limit | Checked our policy 0.40 s");
  assert.ok(h.includes('class="answer-step answer-wait"'), h);
  assert.ok(h.includes("not the model being slow"), h); // the tooltip says what it is
});

test("no wait step when there was none, a trivial one, or an older server that does not send it", () => {
  for (const wait of [0, 0.4, undefined, null, "9"]) {
    const h = html({ seconds: 2, model_seconds: 1.5, wait_seconds: wait, model_calls: 1, tools: [] }, [], LABELS);
    assert.ok(!h.includes("waited"), String(wait));
  }
});

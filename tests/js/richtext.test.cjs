// Run with: node --test tests/js/richtext.test.cjs   (tests/test_richtext_js.py runs it as part of pytest)
//
// The chat bubble's rendering of the agent's light markdown (static/richtext.js). Regression for what was found live:
// "**Propose a smaller payment now**" was shown with its asterisks. What this cannot show is how the page looks.
const test = require("node:test");
const assert = require("node:assert/strict");
const RichText = require("../../src/businessflow/channels/static/richtext.js");

const { formatReply, escapeHtml } = RichText;

const LIVE_REPLY =
  "If you need more than those 3 days, you have a couple of options:\n\n" +
  "1. **Propose a smaller payment now** – you can tell us the amount you’re able to pay.\n" +
  "2. **Extend the tenure** – we can extend the remaining loan term by up to 3 months.";

test("the live reply shows bold text, not asterisks, and keeps its numbering and line breaks", () => {
  const html = formatReply(LIVE_REPLY);
  assert.ok(!html.includes("*"), html);
  assert.ok(html.includes("1. <strong>Propose a smaller payment now</strong> –"), html);
  assert.ok(html.includes("\n2. <strong>Extend the tenure</strong>"), html);
});

test("italics, code, headings and bullets", () => {
  assert.equal(formatReply("pay *today* or _tomorrow_"), "pay <em>today</em> or <em>tomorrow</em>");
  assert.equal(formatReply("Use `BF-1001`"), "Use <code>BF-1001</code>");
  assert.equal(formatReply("## Your options\n- pay now\n* pay later"), "<strong>Your options</strong>\n• pay now\n• pay later");
});

test("text from the model cannot add HTML of its own", () => {
  assert.equal(formatReply('<img src=x onerror="alert(1)"> **ok**'), "&lt;img src=x onerror=&quot;alert(1)&quot;&gt; <strong>ok</strong>");
  assert.equal(formatReply("**<script>x</script>**"), "<strong>&lt;script&gt;x&lt;/script&gt;</strong>");
});

test("lone symbols that are not markup are left alone", () => {
  for (const text of ["2 * 3 = 6", "the late_fee_policy document", "fee applies*", "a ** b"]) {
    assert.equal(formatReply(text), escapeHtml(text), text);
  }
});

test("a '**' inside code stays literal", () => {
  assert.equal(formatReply("run `a **b** c`"), "run <code>a **b** c</code>");
});

test("an empty or missing reply renders as nothing, not 'undefined'", () => {
  assert.equal(formatReply(""), "");
  assert.equal(formatReply(undefined), "");
});

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

// ---- links and tables (found live: a payment link shown as "[Pay ₹35,500 now](https://...token...)", and a payment
// history shown as rows of pipes and dashes) ------------------------------------------------------------------------------

const PAY_URL = "https://businessflowai.duckdns.org/pay/Ybc3vjppZwZ-3jHsrxfyNDJz0kwkMakL";

test("a payment link is a 'Pay' button with the link's own words, opening in a new tab", () => {
  const html = formatReply(`Sure. Here’s a link for the ₹35,500 due today.\n[Pay ₹35,500 now](${PAY_URL})`);
  assert.equal(
    html,
    `Sure. Here’s a link for the ₹35,500 due today.\n<a class="chat-pay-btn" href="${PAY_URL}" target="_blank" rel="noopener noreferrer">Pay ₹35,500 now</a>`
  );
  assert.ok(!html.includes("]("), html);
});

test("a bare payment URL is a button too; other URLs are links, without the sentence's full stop", () => {
  assert.equal(
    formatReply(`Pay here: ${PAY_URL}.`),
    `Pay here: <a class="chat-pay-btn" href="${PAY_URL}" target="_blank" rel="noopener noreferrer">Open the payment page</a>.`
  );
  assert.equal(
    formatReply("See https://example.com/a_b_c?x=1&y=2."),
    'See <a class="chat-link" href="https://example.com/a_b_c?x=1&amp;y=2" target="_blank" rel="noopener noreferrer">https://example.com/a_b_c?x=1&amp;y=2</a>.'
  );
});

test("only http(s) is ever linked, and a quote in a URL cannot reach the markup", () => {
  assert.equal(formatReply("[x](javascript:alert(1))"), "[x](javascript:alert(1))");
  const html = formatReply('[y](https://a.b/" onclick="z)');
  assert.ok(!/onclick="/.test(html), html);
  assert.ok(html.includes('href="https://a.b/"') && html.includes("&quot; onclick=&quot;z)"), html);
});

const TABLE_REPLY =
  "Here’s your recent payment record:\n\n" +
  "| Date | Amount | On‑time? | Type |\n" +
  "|------|--------|----------|------|\n" +
  "| 15 Oct 2026 | ₹35,500 | No | **Regular** EMI |\n" +
  "| 25 Jul 2026 | ₹35,000 | Yes | Regular EMI |\n\n" +
  "You’ve already paid the October EMI.";

test("a markdown table is a real table, its cells formatted, with no blank lines around it", () => {
  const html = formatReply(TABLE_REPLY);
  assert.ok(!html.includes("|"), html);
  assert.ok(html.startsWith('Here’s your recent payment record:<div class="chat-table-wrap"><table class="chat-table">'), html);
  assert.ok(html.includes("<thead><tr><th>Date</th><th>Amount</th><th>On‑time?</th><th>Type</th></tr></thead>"), html);
  assert.ok(html.includes("<tr><td>15 Oct 2026</td><td>₹35,500</td><td>No</td><td><strong>Regular</strong> EMI</td></tr>"), html);
  assert.ok(html.endsWith("</table></div>You’ve already paid the October EMI."), html);
  assert.ok(!/<table[\s\S]*\n[\s\S]*<\/table>/.test(html), "no newline inside the table (the bubble is pre-wrap)");
});

test("a table with a missing cell, and text that only looks like a table, are handled", () => {
  const html = formatReply("| A | B |\n|---|---|\n| 1 |\n");
  assert.ok(html.includes("<tr><td>1</td><td></td></tr>"), html);
  for (const text of ["a | b", "---", "| just one |\n|---|", "2 | 3 and 4 | 5"]) {
    assert.ok(!formatReply(text).includes("<table"), text);
  }
});

// ---- everything else chat models write (the same cases as tests/test_markup.py) ------------------------------------
// Found live: "**👉 https://.../pay/<token>**" -- the link swallowed the closing "**" and showed as a plain link.

const TOKEN_URL = "https://businessflowai.duckdns.org/pay/HlBBxf1eiT1lBImCDngXHO4k6NFWJt3_";
const btn = (href, label) => `<a class="chat-pay-btn" href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
const lnk = (href, label) => `<a class="chat-link" href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;

test("the live bold link is a Pay button, with no stray asterisks", () => {
  const html = formatReply(`Sure, here’s a payment link for the full amount due:\n\n**👉 ${TOKEN_URL}**\n\nJust click the link.`);
  assert.ok(html.includes(`<strong>👉 ${btn(TOKEN_URL, "Open the payment page")}</strong>`), html);
  assert.ok(!html.includes("*"), html);
});

test("links in every shape", () => {
  assert.equal(formatReply(`**[Pay ₹500 now](${PAY_URL})**`), `<strong>${btn(PAY_URL, "Pay ₹500 now")}</strong>`);
  assert.equal(formatReply(`[**Pay now**](${PAY_URL})`), btn(PAY_URL, "Pay now"));
  assert.equal(formatReply(`__${TOKEN_URL}__`), `<strong>${btn(TOKEN_URL, "Open the payment page")}</strong>`);
  assert.equal(formatReply("See &lt;x&gt; <https://example.com/a_b>."), `See &amp;lt;x&amp;gt; ${lnk("https://example.com/a_b", "https://example.com/a_b")}.`);
  assert.equal(formatReply("(see https://example.com/a)"), `(see ${lnk("https://example.com/a", "https://example.com/a")})`);
  assert.equal(formatReply("**https://example.com/a**"), `<strong>${lnk("https://example.com/a", "https://example.com/a")}</strong>`);
});

test("strike, bold italic, code block, divider, quote, tasks, breaks and escapes", () => {
  assert.equal(formatReply("~~₹500~~ ₹400, ***today***"), "<del>₹500</del> ₹400, <strong><em>today</em></strong>");
  assert.equal(formatReply("Run:\n```\na **b** <c>\n```\ndone"), "Run:<pre><code>a **b** &lt;c&gt;</code></pre>done");
  assert.equal(formatReply("above\n---\nbelow"), "above<hr>below");
  assert.equal(formatReply("> Note: **fee** waived"), "<blockquote>Note: <strong>fee</strong> waived</blockquote>");
  assert.equal(formatReply("- [ ] pay\n- [x] done\n+ plus"), "☐ pay\n☑ done\n• plus");
  assert.equal(formatReply("one<br>two<br/>three"), "one\ntwo\nthree");
  assert.ok(formatReply("| A | B |\n|---|---|\n| x | 1<br>2 |").includes("<td>1<br>2</td>"));
  assert.equal(formatReply(String.raw`\*not italic\* and \_kept\_`), "*not italic* and _kept_");
});

test("odd input never breaks the page", () => {
  assert.equal(formatReply("a\u00000\u0000b **c**"), "a0b <strong>c</strong>"); // a NUL can't reach a set-aside piece
  assert.equal(formatReply("```\nunclosed"), "<pre><code>unclosed</code></pre>");
  assert.equal(formatReply("> "), "<blockquote></blockquote>");
});

/* ============================================================
   The agent's markdown, rendered in the chat bubble instead of shown as symbols.

   Replies come the way chat models write them, and every piece showed up live as noise: "**Propose a smaller payment
   now**" with its asterisks, a payment link as "[Pay ₹35,500 now](https://...token...)", a payment history as rows of
   pipes and dashes, and "**👉 https://.../pay/<token>**" -- a bold link whose closing "**" was swallowed into the URL.
   formatReply() renders the subset below. The same rules, for Telegram and speech, are businessflow/markup.py; the two
   test files (tests/js/richtext.test.cjs, tests/test_markup.py) hold the same cases -- keep them in step.

     **x** __x__ -> <strong>   *x* _x_ -> <em>   ***x*** -> both   ~~x~~ -> <del>   `x` -> <code>   ```block``` -> <pre>
     # heading -> <strong>   - / * / + / • item -> "• item"   - [ ] / - [x] -> ☐ / ☑   "1. " kept as written
     > quote -> <blockquote>   --- -> <hr>   <br> -> a line break   \* -> a literal *
     [text](https://...), <https://...>, bare https://... -> a link that opens in a new tab; a payment link (/pay/ and
     its 32-character token, found by that exact shape so a "**" or "_" right after it is never taken in) is drawn as a
     "Pay" button labelled with the link's own words
     | a | table | with its |---| row -> a real <table>, each cell on one line, scrolling sideways when it doesn't fit
   Line breaks stay as written: the bubble is white-space: pre-wrap.

   SAFE BY CONSTRUCTION: the whole reply is HTML-escaped first, and the only tags in the result are the ones written
   here, so text from the model can never add markup of its own. A link is only ever http(s), never javascript:, and a
   URL ends at an escaped quote. A lone "*" or "_" that is not a pair around text (2 * 3, an_identifier) is left alone.
   ============================================================ */
const RichText = (() => {
  const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);

  // In escaped text an "&" is only ever part of an entity; a URL may hold "&amp;" (a query string) and nothing else,
  // so it ends at an escaped quote. It never takes in the markdown marks that wrap a link: * ` ~ |
  const URL_CHAR = "(?:[^\\s<>&*`~|]|&amp;)";
  const PAY_URL = new RegExp(`https?:\\/\\/${URL_CHAR}+?\\/pay\\/[A-Za-z0-9_-]{32}(?![A-Za-z0-9-])`, "g");
  const PAY_URL_ONLY = new RegExp(`^https?:\\/\\/${URL_CHAR}+?\\/pay\\/[A-Za-z0-9_-]{32}$`);
  // A bare URL; a trailing full stop, comma, bracket or "_" belongs to the sentence, not the URL.
  const BARE_URL = new RegExp(`https?:\\/\\/${URL_CHAR}*[^\\s<>&*\`~|.,;:!?)\\]_]`, "g");
  const LINK = new RegExp(`\\[([^\\]\\n]+)\\]\\((https?:\\/\\/${URL_CHAR}+?)\\)`, "g");
  const ANGLE_LINK = new RegExp(`&lt;(https?:\\/\\/${URL_CHAR}+?)&gt;`, "g");

  const ESCAPE = /\\([\\`*_{}\[\]()#+\-.!|~]|&gt;)/g;
  const CODE = /`([^`\n]+)`/g;
  const BR = /&lt;br\s*\/?&gt;/gi;
  const HEADING = /^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$/gm;
  const TASK = /^([ \t]*)[-*+][ \t]+\[([ xX])\][ \t]+/gm;
  const BULLET = /^([ \t]*)[-*+•][ \t]+/gm;
  const STRIKE = /~~(?=\S)(.+?)(?<=\S)~~/g;
  const BOLD_ITALIC = /\*\*\*(?=\S)([\s\S]+?)(?<=\S)\*\*\*/g;
  const BOLD = /\*\*(?=\S)([\s\S]+?)(?<=\S)\*\*|__(?=\S)([\s\S]+?)(?<=\S)__/g;
  const ITALIC = /(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])|(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])/g;
  const PLACEHOLDER = /\u0000(\d+)\u0000/g;
  const TABLE_SEPARATOR = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$|^\s*\|\s*:?-+:?\s*\|\s*$/;
  const RULE = /^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
  const FENCE = /^[ \t]*```/;
  const QUOTE = /^[ \t]*&gt;[ \t]?/;

  const plain = (html) => html.replace(/<[^>]+>/g, "");

  /** A link from already-escaped text: the payment page as a button, anything else as a plain link. */
  function anchor(href, labelHtml) {
    if (PAY_URL_ONLY.test(href)) {
      const label = labelHtml ? plain(labelHtml) : "Open the payment page"; // a button holds words, not markup
      return `<a class="chat-pay-btn" href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
    }
    return `<a class="chat-link" href="${href}" target="_blank" rel="noopener noreferrer">${labelHtml || href}</a>`;
  }

  /** Escaped text (no tables, quotes, code blocks) as HTML. */
  function inline(escaped) {
    const kept = []; // escapes, code, links: set aside so "**" in code or "_" in a URL is never read as markup
    const keep = (html) => {
      kept.push(html);
      return `\u0000${kept.length - 1}\u0000`;
    };
    let out = escaped
      .replace(ESCAPE, (_, ch) => keep(ch))
      .replace(CODE, (_, code) => keep(`<code>${code}</code>`))
      .replace(LINK, (_, label, href) => keep(anchor(href, emphasis(label))))
      .replace(ANGLE_LINK, (_, href) => keep(anchor(href, null)))
      .replace(PAY_URL, (url) => keep(anchor(url, null)))
      .replace(BARE_URL, (url) => keep(anchor(url, null)))
      .replace(BR, "\n")
      .replace(TASK, (_, indent, mark) => `${indent}${mark.toLowerCase() === "x" ? "☑" : "☐"} `)
      .replace(HEADING, "<strong>$1</strong>")
      .replace(BULLET, "$1• ");
    out = emphasis(out);
    // A kept piece can hold another (an escape inside a link label); three passes cover the deepest nesting.
    for (let pass = 0; pass < 3; pass += 1) out = out.replace(PLACEHOLDER, (_, i) => kept[Number(i)] ?? "");
    return out;
  }

  function emphasis(s) {
    return s
      .replace(STRIKE, "<del>$1</del>")
      .replace(BOLD_ITALIC, "<strong><em>$1</em></strong>")
      .replace(BOLD, (_, a, b) => `<strong>${a ?? b}</strong>`)
      .replace(ITALIC, (_, a, b) => `<em>${a ?? b}</em>`);
  }

  const cells = (line) => {
    let row = line.trim();
    if (row.startsWith("|")) row = row.slice(1);
    if (row.endsWith("|")) row = row.slice(0, -1);
    return row.split("|").map((c) => c.trim());
  };
  const cellHtml = (c) => inline(c.replace(BR, "\u0001")).replace(/\u0001/g, "<br>");

  function tableHtml(header, rows) {
    const head = header.map((h) => `<th>${cellHtml(h)}</th>`).join("");
    const body = rows.map((r) => `<tr>${header.map((_, i) => `<td>${cellHtml(r[i] ?? "")}</td>`).join("")}</tr>`).join("");
    // No newlines inside: the bubble is pre-wrap, and they would show as blank lines.
    return `<div class="chat-table-wrap"><table class="chat-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  /** The reply as safe HTML for a chat bubble. */
  function formatReply(text) {
    // NUL marks the set-aside pieces, so none may come from the reply itself.
    const lines = escapeHtml(String(text ?? "").replace(/\u0000/g, "")).split("\n");
    const segments = []; // { text: [lines] } or { block: html }, in order
    let buffer = [];
    const flush = () => {
      segments.push({ text: buffer });
      buffer = [];
    };
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      const header = line.includes("|") ? cells(line) : [];
      if (FENCE.test(line)) {
        let end = i + 1;
        while (end < lines.length && !FENCE.test(lines[end])) end += 1;
        flush();
        segments.push({ block: `<pre><code>${lines.slice(i + 1, end).join("\n")}</code></pre>` });
        i = Math.min(end + 1, lines.length);
      } else if (header.length >= 2 && i + 1 < lines.length && lines[i + 1].includes("|") && TABLE_SEPARATOR.test(lines[i + 1])) {
        let end = i + 2;
        while (end < lines.length && lines[end].includes("|") && lines[end].trim()) end += 1;
        flush();
        segments.push({ block: tableHtml(header, lines.slice(i + 2, end).map(cells)) });
        i = end;
      } else if (QUOTE.test(line)) {
        let end = i;
        while (end < lines.length && QUOTE.test(lines[end])) end += 1;
        flush();
        segments.push({ block: `<blockquote>${inline(lines.slice(i, end).map((l) => l.replace(QUOTE, "")).join("\n"))}</blockquote>` });
        i = end;
      } else if (RULE.test(line)) {
        flush();
        segments.push({ block: "<hr>" });
        i += 1;
      } else {
        buffer.push(line);
        i += 1;
      }
    }
    flush();

    // A block stands on its own, so the blank lines around it would only add space: dropped.
    return segments
      .map((seg, k) => {
        if (seg.block) return seg.block;
        const ls = seg.text.slice();
        if (k > 0) while (ls.length && !ls[0].trim()) ls.shift();
        if (k < segments.length - 1) while (ls.length && !ls[ls.length - 1].trim()) ls.pop();
        return inline(ls.join("\n"));
      })
      .join("");
  }

  return { escapeHtml, formatReply };
})();

// Node (the tests) loads this file as a module; in the browser it is a plain script and `RichText` is a global.
if (typeof module !== "undefined" && module.exports) module.exports = RichText;

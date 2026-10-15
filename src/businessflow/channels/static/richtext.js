/* ============================================================
   The agent's light markdown, rendered in the chat bubble instead of shown as symbols.

   Replies come the way chat models write them: **bold** for the key phrase, numbered and "-" bulleted lists, the odd
   `code`, a [link](https://...) and the occasional table. Shown as text, "**Propose a smaller payment now**" kept its
   asterisks, a payment link showed as "[Pay ₹35,500 now](https://...long token...)", and a payment history came out as
   rows of pipes and dashes (all found live). formatReply() turns that subset into HTML for the bubble:
     **x** / __x__ -> <strong>x</strong>      *x* / _x_ -> <em>x</em>      `x` -> <code>x</code>
     "# Heading"   -> <strong>Heading</strong>   "- item" / "* item" -> "• item"
     [text](https://...) and a bare https://... -> a link that opens in a new tab; a payment link (/pay/<token>) is
     drawn as a "Pay" button instead, labelled with the link's own text
     a | table | with its |---| row -> a real <table>, which scrolls sideways on a narrow screen
   Line breaks and "1. " numbering stay as written: the bubble is white-space: pre-wrap.

   SAFE BY CONSTRUCTION: the whole reply is HTML-escaped first, and the only tags in the result are the ones written
   here, so text from the model can never add markup of its own. A link is only ever http(s), never javascript:. A lone
   "*" or "_" that is not a pair around text (2 * 3, an_identifier) is left alone. The server side of the same rules is
   businessflow/markup.py (speech, Telegram). Tested with node: tests/js/richtext.test.cjs.
   ============================================================ */
const RichText = (() => {
  const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);

  const CODE = /`([^`\n]+)`/g;
  // In escaped text an "&" is only ever part of an entity; allowing only "&amp;" ends a URL at an escaped quote.
  const LINK = /\[([^\]\n]+)\]\((https?:\/\/(?:[^\s)<>&]|&amp;)+)\)/g;
  // A bare URL; a trailing full stop, comma or bracket is the sentence's, not the URL's.
  const BARE_URL = /https?:\/\/(?:[^\s<>&]|&amp;)*[^\s<>&.,;:!?)\]]/g;
  const PAY_LINK = /\/pay\/[A-Za-z0-9_-]+$/;
  const HEADING = /^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$/gm;
  const BULLET = /^([ \t]*)[-*•][ \t]+/gm;
  const BOLD = /\*\*(?=\S)([\s\S]+?)(?<=\S)\*\*|__(?=\S)([\s\S]+?)(?<=\S)__/g;
  const ITALIC = /(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])|(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])/g;
  const PLACEHOLDER = /\u0000(\d+)\u0000/g;
  const TABLE_SEPARATOR = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$|^\s*\|\s*:?-+:?\s*\|\s*$/;

  /** A link from already-escaped text: the payment page as a button, anything else as a plain link. */
  function anchor(href, label) {
    if (PAY_LINK.test(href)) {
      return `<a class="chat-pay-btn" href="${href}" target="_blank" rel="noopener noreferrer">${label || "Open the payment page"}</a>`;
    }
    return `<a class="chat-link" href="${href}" target="_blank" rel="noopener noreferrer">${label || href}</a>`;
  }

  /** Escaped text (no tables) as HTML. */
  function inline(escaped) {
    const kept = []; // code spans and links, set aside so "**" in code or "_" in a URL is never read as markup
    const keep = (html) => {
      kept.push(html);
      return `\u0000${kept.length - 1}\u0000`;
    };
    let out = escaped
      .replace(CODE, (_, code) => keep(`<code>${code}</code>`))
      .replace(LINK, (_, label, href) => keep(anchor(href, label)))
      .replace(BARE_URL, (url) => keep(anchor(url, null)));
    out = out
      .replace(HEADING, "<strong>$1</strong>")
      .replace(BULLET, "$1• ")
      .replace(BOLD, (_, a, b) => `<strong>${a ?? b}</strong>`)
      .replace(ITALIC, (_, a, b) => `<em>${a ?? b}</em>`);
    return out.replace(PLACEHOLDER, (_, i) => kept[Number(i)]);
  }

  const cells = (line) => {
    let row = line.trim();
    if (row.startsWith("|")) row = row.slice(1);
    if (row.endsWith("|")) row = row.slice(0, -1);
    return row.split("|").map((c) => c.trim());
  };

  function tableHtml(header, rows) {
    const head = header.map((h) => `<th>${inline(h)}</th>`).join("");
    const body = rows
      .map((r) => `<tr>${header.map((_, i) => `<td>${inline(r[i] ?? "")}</td>`).join("")}</tr>`)
      .join("");
    // No newlines inside: the bubble is pre-wrap, and they would show as blank lines.
    return `<div class="chat-table-wrap"><table class="chat-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  /** The reply as safe HTML for a chat bubble. */
  function formatReply(text) {
    const lines = escapeHtml(text).split("\n");
    const segments = []; // { text: [lines] } or { table: html }, in order
    let buffer = [];
    let i = 0;
    while (i < lines.length) {
      const header = lines[i].includes("|") ? cells(lines[i]) : [];
      if (header.length >= 2 && i + 1 < lines.length && lines[i + 1].includes("|") && TABLE_SEPARATOR.test(lines[i + 1])) {
        let end = i + 2;
        while (end < lines.length && lines[end].includes("|") && lines[end].trim()) end += 1;
        segments.push({ text: buffer });
        segments.push({ table: tableHtml(header, lines.slice(i + 2, end).map(cells)) });
        buffer = [];
        i = end;
      } else {
        buffer.push(lines[i]);
        i += 1;
      }
    }
    segments.push({ text: buffer });

    // A table is a block of its own, so the blank lines around it would only add space: dropped.
    return segments
      .map((seg, k) => {
        if (seg.table) return seg.table;
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

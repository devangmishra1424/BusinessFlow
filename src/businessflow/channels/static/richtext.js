/* ============================================================
   The agent's light markdown, rendered in the chat bubble instead of shown as symbols.

   Replies come the way chat models write them: **bold** for the key phrase, numbered and "-" bulleted lists, the odd
   `code`. Shown as text, "**Propose a smaller payment now**" kept its asterisks (found live). formatReply() turns that
   subset into HTML for the bubble:
     **x** / __x__ -> <strong>x</strong>      *x* / _x_ -> <em>x</em>      `x` -> <code>x</code>
     "# Heading"   -> <strong>Heading</strong>   "- item" / "* item" -> "• item"
   Line breaks and "1. " numbering stay as written: the bubble is white-space: pre-wrap.

   SAFE BY CONSTRUCTION: the whole reply is HTML-escaped first, and the only tags in the result are the ones written
   here, so text from the model can never add markup of its own. A lone "*" or "_" that is not a pair around text
   (2 * 3, an_identifier) is left alone. The server side of the same rules is businessflow/markup.py (speech, Telegram).
   Tested with node: tests/js/richtext.test.cjs.
   ============================================================ */
const RichText = (() => {
  const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);

  const CODE = /`([^`\n]+)`/g;
  const HEADING = /^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$/gm;
  const BULLET = /^([ \t]*)[-*•][ \t]+/gm;
  const BOLD = /\*\*(?=\S)([\s\S]+?)(?<=\S)\*\*|__(?=\S)([\s\S]+?)(?<=\S)__/g;
  const ITALIC = /(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])|(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])/g;
  const PLACEHOLDER = /\u0000(\d+)\u0000/g;

  /** The reply as safe HTML for a chat bubble. */
  function formatReply(text) {
    const kept = []; // code spans, set aside so that a "**" inside code stays literal
    let out = escapeHtml(text).replace(CODE, (_, code) => {
      kept.push(`<code>${code}</code>`);
      return `\u0000${kept.length - 1}\u0000`;
    });
    out = out
      .replace(HEADING, "<strong>$1</strong>")
      .replace(BULLET, "$1• ")
      .replace(BOLD, (_, a, b) => `<strong>${a ?? b}</strong>`)
      .replace(ITALIC, (_, a, b) => `<em>${a ?? b}</em>`);
    return out.replace(PLACEHOLDER, (_, i) => kept[Number(i)]);
  }

  return { escapeHtml, formatReply };
})();

// Node (the tests) loads this file as a module; in the browser it is a plain script and `RichText` is a global.
if (typeof module !== "undefined" && module.exports) module.exports = RichText;

"""The markdown the agent writes, handled per channel instead of shown or spoken as symbols.

The model formats replies the way chat models do, and every piece of it showed up live as noise: "**Propose a smaller
payment now**" with its asterisks, a payment link as "[Pay ₹35,500 now](https://...token...)", a payment history as
rows of pipes and dashes, and "**👉 https://.../pay/<token>**" -- a bold link whose closing "**" got swallowed into the
URL -- with the voice reading the 👉 aloud. Each channel now does what the markup was for:
  - speech:   strip_markdown() removes it (verbalize() then also drops emoji and stray symbols, audio/verbalizer.py);
  - Telegram: markdown_to_telegram_html() turns it into Telegram's HTML subset (<b> <i> <s> <code> <pre> <a>);
  - web chat: static/richtext.js renders the same rules in the browser. Keep the two in step:
    tests/test_markup.py and tests/js/richtext.test.cjs hold the same cases.

What is handled:
  **bold**  __bold__  *italic*  _italic_  ***both***  ~~strike~~  `code`  ```code block```
  # headings  - / * / + / • bullets  1. numbers (kept)  - [ ] / - [x] tasks  > quotes  --- dividers  <br>
  [text](https://...)  <https://...>  bare https://...  \\* escapes  | tables |
  A payment link (/pay/ followed by the 32-character token) is found by its exact shape, so bold markers or
  underscores right after it are never taken into it.

A lone "*" or "_" that is not a pair around text (2 * 3, an_identifier, a footnote*) is left alone on screen; guessing
would eat real characters. Speech drops such leftovers (verbalize), since a voice would only read them out.
"""

import html
import re

PAY_LINK_LABEL = "Open the payment page"

# A pair of ** or __ around text that neither starts nor ends with a space: "**bold**", not "2 ** 3".
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|__(?=\S)(.+?)(?<=\S)__", re.S)
_BOLD_ITALIC = re.compile(r"\*\*\*(?=\S)(.+?)(?<=\S)\*\*\*", re.S)
# One * or _ around text on one line, not touching a word character outside: "*this*", not "snake_case_name".
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])|(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")
_STRIKE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_CODE = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"^[ \t]*```[^\n]*\n(.*?)\n[ \t]*```[ \t]*$", re.M | re.S)
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.M)
_TASK = re.compile(r"^([ \t]*)[-*+][ \t]+\[([ xX])\][ \t]+", re.M)
_BULLET = re.compile(r"^([ \t]*)[-*+•][ \t]+", re.M)
_RULE = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.M)
_BR = re.compile(r"<br\s*/?>", re.I)
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|~>])")
_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")

# Links. Characters that never belong to a URL here: whitespace, <>, quotes, and the markdown marks * ` ~ | that wrap
# one ("**https://...**"). A URL also never ends in punctuation that belongs to the sentence, or in "_".
_URL_CHARS = r"[^\s<>\"'*`~|]"
_PAY_URL = re.compile(rf"https?://{_URL_CHARS}+?/pay/[A-Za-z0-9_-]{{32}}(?![A-Za-z0-9-])")
_BARE_URL = re.compile(rf"https?://{_URL_CHARS}*[^\s<>\"'*`~|.,;:!?)\]_]")
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)<>\"'*`]+)\)")
_ANGLE_LINK_RAW = re.compile(r"<(https?://[^\s<>]+)>")
_ANGLE_LINK_ESCAPED = re.compile(r"&lt;(https?://[^\s<>&]+(?:&amp;[^\s<>&]+)*)&gt;")
_QUOTE_RAW = re.compile(r"^[ \t]*>[ \t]?", re.M)
_QUOTE_ESCAPED = re.compile(r"^[ \t]*&gt;[ \t]?(.*)$", re.M)

# A table's separator row: | --- | :---: | ---: |
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$|^\s*\|\s*:?-+:?\s*\|\s*$")


def _either(match: re.Match) -> str:
    return match.group(1) if match.group(1) is not None else match.group(2)


def _table_cells(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [_BR.sub("; ", cell).strip() for cell in row.split("|")]


def find_tables(lines: list[str]) -> list[tuple[int, int, list[str], list[list[str]]]]:
    """(first line, line after the last, header cells, body rows) for every markdown table: a header row with at least
    two cells, the "|---|---|" row, then the rows that follow until a line without a "|"."""
    tables = []
    i = 0
    while i < len(lines) - 1:
        header = _table_cells(lines[i]) if "|" in lines[i] else []
        if len(header) >= 2 and "|" in lines[i + 1] and _TABLE_SEPARATOR.match(lines[i + 1]):
            end = i + 2
            while end < len(lines) and "|" in lines[end] and lines[end].strip():
                end += 1
            tables.append((i, end, header, [_table_cells(line) for line in lines[i + 2:end]]))
            i = end
        else:
            i += 1
    return tables


def _plain(cell: str) -> str:
    return _BOLD.sub(_either, cell).strip()


def _row_parts(header: list[str], cells: list[str]) -> tuple[str, list[str]]:
    """A row as (its first cell, "Label: value" for the rest). A header that already ends in "?" or ":" ("On-time?")
    takes no extra colon; empty cells are skipped; a cell beyond the header is kept as it is."""
    lead = _plain(cells[0]) if cells else ""
    parts = []
    for index, value in enumerate(cells[1:], start=1):
        value = value.strip()
        if not value:
            continue
        label = _plain(header[index]) if index < len(header) else ""
        if not label:
            parts.append(value)
        elif label.endswith(("?", ":")):
            parts.append(f"{label} {value}")
        else:
            parts.append(f"{label}: {value}")
    return lead, parts


def tables_as_lines(text: str, for_speech: bool = False) -> str:
    """Every markdown table rewritten as one line per row (the header row becomes the labels):
    on screen  "- **15 Oct 2026** — Amount: ₹35,500 · On-time? No"   (a list item, so Telegram shows a bullet)
    spoken     "15 Oct 2026: Amount: ₹35,500, On-time? No."            (a sentence per row)"""
    lines = text.split("\n")
    for start, end, header, rows in reversed(find_tables(lines)):
        rendered = []
        for cells in rows:
            lead, parts = _row_parts(header, cells)
            if for_speech:
                rendered.append(f"{lead}: {', '.join(parts)}." if parts else f"{lead}.")
            else:
                rendered.append(f"- **{lead}** — {' · '.join(parts)}" if lead and parts else f"- {lead or ' · '.join(parts)}")
        lines[start:end] = rendered
    return "\n".join(lines)


def _blocks(text: str, for_speech: bool) -> str:
    """The line-level rules every channel shares, on the raw reply: tables to lines, <br> to a line break, divider
    lines removed. Code blocks are left for the caller (each channel shows them its own way)."""
    text = tables_as_lines(text, for_speech=for_speech)
    text = _BR.sub("\n", text)
    return _RULE.sub("", text)


def strip_markdown(text: str) -> str:
    """The reply as words to be spoken: markup removed, the words it wrapped kept. Numbered list markers ("1. ") stay,
    since "one, propose a smaller payment" is how a person would read the list out; a table is read a row at a time.
    A link is spoken as its words; a bare URL is left for verbalize() to replace. Idempotent."""
    text = _FENCE.sub(r"\1", text)
    text = _blocks(text, for_speech=True)
    text = _ESCAPE.sub(r"\1", text)
    text = _QUOTE_RAW.sub("", text)
    text = _TASK.sub(r"\1", text)
    text = _LINK.sub(lambda m: _plain(_STRIKE.sub(r"\1", m.group(1))), text)
    text = _ANGLE_LINK_RAW.sub(r"\1", text)
    text = _CODE.sub(r"\1", text)
    text = _HEADING.sub(r"\1", text)
    text = _BULLET.sub(r"\1", text)
    text = _STRIKE.sub(r"\1", text)
    text = _BOLD_ITALIC.sub(r"\1", text)
    text = _BOLD.sub(_either, text)
    return _ITALIC.sub(_either, text)


def markdown_to_telegram_html(text: str) -> str:
    """The reply in Telegram's HTML subset (send it with parse_mode=HTML). Everything is escaped first, so the only tags
    in the result are the ones written here; text from the model cannot add markup of its own. Code, escapes and links
    are set aside before the other rules run: a "**" inside code stays literal (Telegram allows no tags inside <code>),
    and an underscore in a URL can never become an <i> inside the href.

    Telegram has no tables, dividers or quote blocks that read well on a phone: a table row becomes one line, a
    divider is dropped, a quoted line is shown in italics. A bare URL other than a payment link is left as text --
    Telegram links it itself."""
    kept: list[str] = []

    def keep(fragment: str) -> str:
        kept.append(fragment)
        return f"\x00{len(kept) - 1}\x00"

    text = text.replace("\x00", "")  # NUL marks the set-aside pieces, so none may come from the reply itself
    raw = _FENCE.sub(lambda m: keep(f"<pre>{html.escape(m.group(1), quote=False)}</pre>"), text)
    raw = _ESCAPE.sub(lambda m: keep(html.escape(m.group(1), quote=False)), raw)
    out = html.escape(_blocks(raw, for_speech=False), quote=False)
    out = _CODE.sub(lambda m: keep(f"<code>{m.group(1)}</code>"), out)
    out = _LINK.sub(lambda m: keep(f'<a href="{m.group(2)}">{_plain(m.group(1))}</a>'), out)
    out = _ANGLE_LINK_ESCAPED.sub(lambda m: keep(f'<a href="{m.group(1)}">{_link_text(m.group(1))}</a>'), out)
    out = _PAY_URL.sub(lambda m: keep(f'<a href="{m.group(0)}">{PAY_LINK_LABEL}</a>'), out)
    out = _QUOTE_ESCAPED.sub(lambda m: f"<i>{m.group(1)}</i>" if m.group(1).strip() else "", out)
    out = _TASK.sub(lambda m: f"{m.group(1)}{'☑' if m.group(2).lower() == 'x' else '☐'} ", out)
    out = _HEADING.sub(r"<b>\1</b>", out)
    out = _BULLET.sub(r"\1• ", out)
    out = _STRIKE.sub(r"<s>\1</s>", out)
    out = _BOLD_ITALIC.sub(r"<b><i>\1</i></b>", out)
    out = _BOLD.sub(lambda m: f"<b>{_either(m)}</b>", out)
    out = _ITALIC.sub(lambda m: f"<i>{_either(m)}</i>", out)
    for _ in range(3):  # a kept piece can hold another (an escape inside a link label); three passes cover the deepest
        out = _PLACEHOLDER.sub(lambda m: kept[int(m.group(1))] if int(m.group(1)) < len(kept) else "", out)
    return out


def _link_text(url: str) -> str:
    return PAY_LINK_LABEL if _PAY_URL.fullmatch(url) else url

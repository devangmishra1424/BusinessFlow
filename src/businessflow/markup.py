"""The light markdown the agent writes, handled per channel instead of shown or spoken as symbols.

The model formats replies the way chat models do: **bold** for the key phrase, numbered and "-" bulleted lists, the odd
`code` or [link](https://...). Shown as plain text, that markup is noise ("**Propose a smaller payment now**"), and a
voice reads it aloud as "asterisk asterisk". Found live in the web chat and on Telegram. Each channel now does what the
markup was for:
  - speech:   strip_markdown() removes it before verbalize() and TTS (every voice path goes through verbalize);
  - Telegram: markdown_to_telegram_html() turns it into Telegram's HTML subset (<b>, <i>, <code>, <a>);
  - web chat: static/richtext.js renders the same subset in the browser.

Tables too: the model answers "what is my payment history" with a markdown table, which showed as rows of pipes and
dashes in the web chat and on Telegram (found live). Telegram has no tables and a phone is too narrow for columns, so
there each row becomes one line: "• **15 Oct 2026** — Amount: ₹35,500 · On-time? No". Speech reads each row as a
sentence. The web chat draws a real table (static/richtext.js).

Only this subset is handled. A lone "*" or "_" that is not a pair around text (2 * 3, an_identifier, a footnote*) is
left as it is, because guessing would eat real characters.
"""

import html
import re

# A pair of ** or __ around text that neither starts nor ends with a space: "**bold**", not "2 ** 3".
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|__(?=\S)(.+?)(?<=\S)__", re.S)
# One * or _ around text on one line, not touching a word character outside: "*this*", not "snake_case_name".
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])|(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")
_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)\"]+)\)")
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.M)
_BULLET = re.compile(r"^([ \t]*)[-*•][ \t]+", re.M)
_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")
# A payment link written out bare, not as [text](link). Shown as words on Telegram rather than a long token URL.
_BARE_PAY_URL = re.compile(r"(?<![\"=>])https?://[^\s<>\"']+/pay/[A-Za-z0-9_-]+")
PAY_LINK_LABEL = "Open the payment page"
# A table's separator row: | --- | :---: | ---: |
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$|^\s*\|\s*:?-+:?\s*\|\s*$")


def _table_cells(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


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


def _either(match: re.Match) -> str:
    return match.group(1) if match.group(1) is not None else match.group(2)


def strip_markdown(text: str) -> str:
    """The reply as words to be spoken: markup removed, the words it wrapped kept. Numbered list markers ("1. ") stay,
    since "one, propose a smaller payment" is how a person would read the list out. A table is read a row at a time
    (tables_as_lines). Idempotent."""
    text = tables_as_lines(text, for_speech=True)
    text = _LINK.sub(r"\1", text)
    text = _CODE.sub(r"\1", text)
    text = _HEADING.sub(r"\1", text)
    text = _BULLET.sub(r"\1", text)
    text = _BOLD.sub(_either, text)
    return _ITALIC.sub(_either, text)


def markdown_to_telegram_html(text: str) -> str:
    """The reply in Telegram's HTML subset (send it with parse_mode=HTML). Everything is escaped first, so the only tags
    in the result are the ones written here; text from the model cannot add markup of its own. Code spans and links are
    set aside before the other rules run: a "**" inside code stays literal (Telegram allows no tags inside <code>), and an
    underscore in a URL can never become an <i> inside the href."""
    kept: list[str] = []

    def keep(fragment: str) -> str:
        kept.append(fragment)
        return f"\x00{len(kept) - 1}\x00"

    out = html.escape(tables_as_lines(text), quote=False)
    out = _CODE.sub(lambda m: keep(f"<code>{m.group(1)}</code>"), out)
    out = _LINK.sub(lambda m: keep(f'<a href="{m.group(2)}">{m.group(1)}</a>'), out)
    out = _BARE_PAY_URL.sub(lambda m: keep(f'<a href="{m.group(0)}">{PAY_LINK_LABEL}</a>'), out)
    out = _HEADING.sub(r"<b>\1</b>", out)
    out = _BULLET.sub(r"\1• ", out)
    out = _BOLD.sub(lambda m: f"<b>{_either(m)}</b>", out)
    out = _ITALIC.sub(lambda m: f"<i>{_either(m)}</i>", out)
    return _PLACEHOLDER.sub(lambda m: kept[int(m.group(1))], out)

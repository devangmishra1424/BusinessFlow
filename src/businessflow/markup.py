"""The light markdown the agent writes, handled per channel instead of shown or spoken as symbols.

The model formats replies the way chat models do: **bold** for the key phrase, numbered and "-" bulleted lists, the odd
`code` or [link](https://...). Shown as plain text, that markup is noise ("**Propose a smaller payment now**"), and a
voice reads it aloud as "asterisk asterisk". Found live in the web chat and on Telegram. Each channel now does what the
markup was for:
  - speech:   strip_markdown() removes it before verbalize() and TTS (every voice path goes through verbalize);
  - Telegram: markdown_to_telegram_html() turns it into Telegram's HTML subset (<b>, <i>, <code>, <a>);
  - web chat: static/richtext.js renders the same subset in the browser.

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


def _either(match: re.Match) -> str:
    return match.group(1) if match.group(1) is not None else match.group(2)


def strip_markdown(text: str) -> str:
    """The reply as words to be spoken: markup removed, the words it wrapped kept. Numbered list markers ("1. ") stay,
    since "one, propose a smaller payment" is how a person would read the list out. Idempotent."""
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

    out = html.escape(text, quote=False)
    out = _CODE.sub(lambda m: keep(f"<code>{m.group(1)}</code>"), out)
    out = _LINK.sub(lambda m: keep(f'<a href="{m.group(2)}">{m.group(1)}</a>'), out)
    out = _HEADING.sub(r"<b>\1</b>", out)
    out = _BULLET.sub(r"\1• ", out)
    out = _BOLD.sub(lambda m: f"<b>{_either(m)}</b>", out)
    out = _ITALIC.sub(lambda m: f"<i>{_either(m)}</i>", out)
    return _PLACEHOLDER.sub(lambda m: kept[int(m.group(1))], out)

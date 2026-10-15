"""The agent's markdown, per channel (businessflow/markup.py): removed for speech, turned into Telegram HTML. The browser's
own renderer (static/richtext.js) is tested in tests/js/richtext.test.cjs.

Regression for what was found live: the web chat showed "**Propose a smaller payment now**" with its asterisks, and the
voice read them out."""

from businessflow.audio.verbalizer import verbalize
from businessflow.markup import markdown_to_telegram_html, strip_markdown

# The reply that showed the bug, as the agent wrote it.
LIVE_REPLY = (
    "If you need more than those 3 days, you have a couple of options:\n\n"
    "1. **Propose a smaller payment now** – you can tell us the amount you’re able to pay this cycle.\n"
    "2. **Extend the tenure** – we can extend the remaining loan term by up to 3 months."
)


# ---- speech ------------------------------------------------------------------------------------------------------------

def test_the_live_reply_loses_its_asterisks_but_keeps_its_words_and_numbering():
    spoken = strip_markdown(LIVE_REPLY)
    assert "*" not in spoken
    assert "1. Propose a smaller payment now –" in spoken
    assert "2. Extend the tenure –" in spoken


def test_every_voice_path_gets_text_without_markup_because_verbalize_strips_it():
    spoken = verbalize("**Pay ₹500 today** or *tomorrow*", "en")
    assert "*" not in spoken
    assert spoken == "Pay five hundred rupees today or tomorrow"


def test_bullets_headings_code_and_links_become_plain_words():
    text = "## Your options\n- pay now\n* pay later\n• ask for help\nUse `BF-1001`, or [open the link](https://pay.example/x)."
    assert strip_markdown(text) == "Your options\npay now\npay later\nask for help\nUse BF-1001, or open the link."


def test_lone_symbols_that_are_not_markup_are_left_alone():
    # an arithmetic star, an identifier with underscores, a footnote mark: guessing would eat real characters
    for text in ("2 * 3 = 6", "the late_fee_policy document", "fee applies*", "a ** b"):
        assert strip_markdown(text) == text


def test_stripping_twice_changes_nothing_more():
    once = strip_markdown(LIVE_REPLY)
    assert strip_markdown(once) == once


# ---- Telegram ----------------------------------------------------------------------------------------------------------

def test_telegram_gets_bold_for_double_asterisks_and_a_bullet_for_dashes():
    html_text = markdown_to_telegram_html("1. **Extend the tenure** now\n- or *pay* later")
    assert html_text == "1. <b>Extend the tenure</b> now\n• or <i>pay</i> later"


def test_text_from_the_model_cannot_add_its_own_html_on_telegram():
    html_text = markdown_to_telegram_html("<b>fake</b> & <script>x</script> **real**")
    assert html_text == "&lt;b&gt;fake&lt;/b&gt; &amp; &lt;script&gt;x&lt;/script&gt; <b>real</b>"


def test_code_stays_literal_and_a_url_underscore_never_becomes_italics():
    html_text = markdown_to_telegram_html("Run `a **b** c` then [pay](https://pay.example/_x_/y)")
    assert html_text == 'Run <code>a **b** c</code> then <a href="https://pay.example/_x_/y">pay</a>'


# ---- tables and payment links (found live: a payment history shown as rows of pipes and dashes) ---------------------

TABLE_REPLY = (
    "Here’s your recent payment record:\n\n"
    "| Date | Amount | On‑time? | Type |\n"
    "|------|--------|----------|------|\n"
    "| 15 Oct 2026 | ₹35,500 | No | Regular EMI |\n"
    "| 25 Jul 2026 | ₹35,000 | Yes | Regular EMI |\n\n"
    "You’ve already paid the October EMI."
)
PAY_URL = "https://businessflowai.duckdns.org/pay/Ybc3vjppZwZ-3jHsrxfyNDJz0kwkMakL"


def test_telegram_gets_a_table_as_one_line_per_row_because_it_has_no_tables():
    html_text = markdown_to_telegram_html(TABLE_REPLY)
    assert "|" not in html_text and "---" not in html_text
    assert html_text == (
        "Here’s your recent payment record:\n\n"
        "• <b>15 Oct 2026</b> — Amount: ₹35,500 · On‑time? No · Type: Regular EMI\n"
        "• <b>25 Jul 2026</b> — Amount: ₹35,000 · On‑time? Yes · Type: Regular EMI\n\n"
        "You’ve already paid the October EMI."
    )


def test_speech_reads_a_table_a_row_at_a_time():
    spoken = strip_markdown(TABLE_REPLY)
    assert "|" not in spoken and "*" not in spoken
    assert "15 Oct 2026: Amount: ₹35,500, On‑time? No, Type: Regular EMI." in spoken
    assert strip_markdown(spoken) == spoken  # still idempotent
    assert "|" not in verbalize(TABLE_REPLY, "en")


def test_a_table_cell_missing_or_bolded_and_text_that_only_looks_like_a_table():
    assert markdown_to_telegram_html("| A | B | C |\n|---|---|---|\n| **x** |  | z |") == "• <b>x</b> — C: z"
    for text in ("a | b", "| one |\n|---|", "2 | 3 and 4 | 5"):  # a lone "---" is a divider: dropped, tested below
        assert markdown_to_telegram_html(text) == text.replace("&", "&amp;")


def test_a_payment_link_on_telegram_keeps_its_words_and_a_bare_one_gets_words():
    assert markdown_to_telegram_html(f"[Pay ₹35,500 now]({PAY_URL})") == f'<a href="{PAY_URL}">Pay ₹35,500 now</a>'
    assert markdown_to_telegram_html(f"Pay here: {PAY_URL}.") == f'Pay here: <a href="{PAY_URL}">Open the payment page</a>.'
    assert markdown_to_telegram_html("See https://example.com/x") == "See https://example.com/x"  # Telegram links it itself


def test_a_payment_link_is_spoken_as_its_words_never_the_url():
    spoken = verbalize(f"Here’s your link.\n[Pay now]({PAY_URL})", "en")
    assert "http" not in spoken and "Pay now" in spoken

# ---- everything else chat models write (the same cases as tests/js/richtext.test.cjs) --------------------------------
# Found live: "**👉 https://.../pay/<token>**" -- an emoji and a bare payment link inside bold; the web chat swallowed
# the closing "**" into the link and the voice read the 👉 aloud.

LIVE_BOLD_LINK = (
    "Sure, here’s a payment link for the full amount due:\n\n"
    "**👉 https://businessflowai.duckdns.org/pay/HlBBxf1eiT1lBImCDngXHO4k6NFWJt3_**\n\n"
    "Just click the link, confirm the ₹20,500 payment, and your EMI will be cleared."
)
TOKEN_URL = "https://businessflowai.duckdns.org/pay/HlBBxf1eiT1lBImCDngXHO4k6NFWJt3_"


def test_the_live_bold_link_is_a_link_on_telegram_and_never_spoken():
    html_text = markdown_to_telegram_html(LIVE_BOLD_LINK)
    assert f'<b>👉 <a href="{TOKEN_URL}">Open the payment page</a></b>' in html_text
    assert "**" not in html_text
    spoken = verbalize(LIVE_BOLD_LINK, "en")
    assert "👉" not in spoken and "*" not in spoken and "http" not in spoken
    assert "it's in the chat" in spoken and "twenty thousand, five hundred rupees" in spoken


def test_links_in_every_shape_telegram():
    assert markdown_to_telegram_html(f"**[Pay ₹500 now]({PAY_URL})**") == f'<b><a href="{PAY_URL}">Pay ₹500 now</a></b>'
    assert markdown_to_telegram_html(f"[**Pay now**]({PAY_URL})") == f'<a href="{PAY_URL}">Pay now</a>'
    assert markdown_to_telegram_html(f"__{TOKEN_URL}__") == f'<b><a href="{TOKEN_URL}">Open the payment page</a></b>'
    assert markdown_to_telegram_html("See <https://example.com/a_b>.") == 'See <a href="https://example.com/a_b">https://example.com/a_b</a>.'
    assert markdown_to_telegram_html("(see https://example.com/a)") == "(see https://example.com/a)"  # Telegram links it


def test_strike_bold_italic_code_block_divider_quote_tasks_breaks_and_escapes_telegram():
    assert markdown_to_telegram_html("~~₹500~~ ₹400, ***today***") == "<s>₹500</s> ₹400, <b><i>today</i></b>"
    assert markdown_to_telegram_html("Run:\n```\na **b** <c>\n```\ndone") == "Run:\n<pre>a **b** &lt;c&gt;</pre>\ndone"
    assert markdown_to_telegram_html("above\n---\nbelow") == "above\n\nbelow"
    assert markdown_to_telegram_html("> Note: **fee** waived") == "<i>Note: <b>fee</b> waived</i>"
    assert markdown_to_telegram_html("- [ ] pay\n- [x] done\n+ plus") == "☐ pay\n☑ done\n• plus"
    assert markdown_to_telegram_html("one<br>two<br/>three") == "one\ntwo\nthree"
    assert markdown_to_telegram_html("| A | B |\n|---|---|\n| x | 1<br>2 |") == "• <b>x</b> — B: 1; 2"
    assert markdown_to_telegram_html(r"\*not italic\* and \_kept\_") == "*not italic* and _kept_"


def test_telegram_output_has_no_markup_of_its_own_and_survives_odd_input():
    assert markdown_to_telegram_html("[x](javascript:alert(1)) <b>x</b>") == "[x](javascript:alert(1)) &lt;b&gt;x&lt;/b&gt;"
    assert markdown_to_telegram_html("a\x000\x00b **c**") == "a0b <b>c</b>"  # a NUL in the reply can't reach a kept piece


def test_speech_drops_every_kind_of_markup_and_symbol():
    text = (
        "## Options\n> Note: ~~₹500~~ fee waived 🎉\n- [ ] pay **now** → done\n---\n```\ncode\n```\n"
        "Check late_fee_policy, 2 * 3, and <https://example.com/x>."
    )
    spoken = verbalize(text, "en")
    for symbol in ("#", ">", "~", "*", "[", "]", "`", "→", "🎉", "---", "_", "http"):
        assert symbol not in spoken, (symbol, spoken)
    assert "Options" in spoken and "pay now, done" in spoken and "late fee policy" in spoken and "2 times 3" in spoken
    assert verbalize(text, "en") == verbalize(verbalize(text, "en"), "en")  # still idempotent


def test_speech_in_hindi_drops_emoji_too():
    spoken = verbalize("भुगतान करें 👉 **[अभी भरें](" + PAY_URL + ")** ✅", "hi")
    assert "👉" not in spoken and "✅" not in spoken and "*" not in spoken and "अभी भरें" in spoken

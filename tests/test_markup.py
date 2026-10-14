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

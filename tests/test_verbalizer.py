from businessflow.audio.verbalizer import _hindi_number_words, has_unverbalized_pattern, verbalize


def test_verbalizes_rupee_amount_with_rs_prefix():
    assert verbalize("Rs. 12500 is due") == "twelve thousand, five hundred rupees is due"


def test_verbalizes_rupee_amount_with_symbol_and_paise():
    assert verbalize("pay ₹585,200.50 now") == "pay five hundred and eighty-five thousand, two hundred rupees and fifty paise now"


def test_verbalizes_iso_date():
    assert verbalize("due on 2026-08-18") == "due on the eighteenth of August, twenty twenty-six"


def test_leaves_small_plain_numbers_untouched():
    assert verbalize("it is 3 days past due") == "it is 3 days past due"


def test_default_language_is_english_unchanged():
    # language is an added, defaulted parameter -- every existing call
    # site (and caller that hasn't been updated) must keep behaving
    # exactly as before.
    assert verbalize("Rs. 12500 is due") == verbalize("Rs. 12500 is due", "en")


# --- Hindi (language="hi") ----------------------------------------------
#
# Regression coverage for a real bug found live via the browser voice
# widget: num2words has no Hindi converter at all (confirmed against its
# own CONVERTER_CLASSES -- "hi" isn't in it), so verbalize() was always
# substituting ENGLISH number words into text about to be spoken by
# speak_hindi (MMS-TTS Hindi, a monolingual model). It silently can't
# pronounce English words, so a Hindi reply like "aapki EMI rashi
# ₹17,053.59 hai" came out with the Hindi words spoken and the amount (and
# the bare English word "EMI") just missing entirely.


def test_hindi_rupee_amount_produces_hindi_words_not_english():
    result = verbalize("aapki EMI rashi ₹17,053.59 hai.", "hi")

    assert "सत्रह हज़ार तिरेपन रुपये" in result  # seventeen thousand fifty-three rupees
    assert "उनसठ पैसे" in result  # fifty-nine paise
    assert "seventeen" not in result and "rupees" not in result  # no leftover English number words


def test_hindi_mode_handles_devanagari_digits_in_the_amount_too():
    # The LLM sometimes writes the amount with Devanagari numerals
    # (int()/float() parse those natively, same as re's \d already does --
    # see verbalizer.py's own comment) rather than ASCII ones.
    result = verbalize("आपकी वर्तमान EMI राशि ₹१७,०५३.५९ है।", "hi")

    assert "सत्रह हज़ार तिरेपन रुपये" in result
    assert "उनसठ पैसे" in result


def test_hindi_mode_translates_the_emi_glossary_term():
    result = verbalize("aapki EMI due hai", "hi")

    assert "ईएमआई" in result
    assert "EMI" not in result


def test_english_mode_leaves_emi_as_is():
    # The glossary substitution is Hindi-only -- "EMI" is already
    # perfectly speakable English text for speak_english.
    assert verbalize("your EMI is due", "en") == "your EMI is due"


def test_hindi_mode_verbalizes_dates_in_hindi():
    result = verbalize("due on 2026-10-01", "hi")

    assert result == "due on एक अक्टूबर, दो हज़ार छब्बीस"


def test_hindi_number_words_spot_checks():
    # A spread of values spanning every branch (units, teens, an irregular
    # tens value, an exact hundred, thousand-plus-remainder, and a value
    # with no thousands component) -- not exhaustive over 0-99, but wide
    # enough to catch a wrong table entry or a broken place-value split.
    assert _hindi_number_words(0) == "शून्य"
    assert _hindi_number_words(17) == "सत्रह"
    assert _hindi_number_words(53) == "तिरेपन"
    assert _hindi_number_words(59) == "उनसठ"
    assert _hindi_number_words(100) == "एक सौ"
    assert _hindi_number_words(2026) == "दो हज़ार छब्बीस"
    assert _hindi_number_words(17053) == "सत्रह हज़ार तिरेपन"
    assert _hindi_number_words(100000) == "एक लाख"


# --- has_unverbalized_pattern --------------------------------------------
#
# Used by eval/voice_naturalness_benchmark.py to confirm the exact live
# bug this module's own docstring describes (a number silently reaching
# TTS unconverted) hasn't come back.


def test_has_unverbalized_pattern_true_for_raw_iso_date():
    assert has_unverbalized_pattern("due on 2026-08-18") is True


def test_has_unverbalized_pattern_true_for_raw_rupee_amount():
    assert has_unverbalized_pattern("pay ₹12,500 now") is True


def test_has_unverbalized_pattern_false_after_verbalize():
    text = verbalize("Your ₹12,500 payment is due on 2026-09-15.")

    assert has_unverbalized_pattern(text) is False


def test_has_unverbalized_pattern_false_for_text_with_no_pattern_at_all():
    assert has_unverbalized_pattern("it is 3 days past due") is False

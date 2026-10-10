import pytest

from businessflow.audio.verbalizer import _hindi_number_words, has_unverbalized_pattern, verbalize


def test_verbalizes_rupee_amount_with_rs_prefix():
    assert verbalize("Rs. 12500 is due") == "twelve thousand, five hundred rupees is due"


def test_verbalizes_rupee_amount_with_symbol_and_paise():
    # Indian grouping: 585,200 is "five lakh, eighty-five thousand, two
    # hundred" to an Indian caller -- not the western "five hundred and
    # eighty-five thousand" this assertion used to expect.
    assert verbalize("pay ₹585,200.50 now") == "pay five lakh, eighty-five thousand, two hundred rupees and fifty paise now"


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


# --- emails, phone numbers, account IDs, percentages, Indian grouping -----
#
# TTS reads "BF-1001" as a blob and "9812345000" as one huge number; neither
# can be written down by the listener. These must be spelled out. Every
# case is checked in English and Hindi because Hindi TTS is monolingual and
# silently drops Latin letters/words (see verbalizer.py's docstring).


@pytest.mark.parametrize("text, expected", [
    ("Mail devang.mishra@gmail.com for help.",
     "Mail D E V A N G dot M I S H R A, at, gmail dot com for help."),
    ("reach me: a@b.com, c@d.org.",
     "reach me: A, at, B dot com, C, at, D dot org."),
    ("Write to billing_team+india@acme-corp.co.in today",
     "Write to B I L L I N G underscore T E A M plus I N D I A, at, A C M E dash C O R P dot C O dot in today"),
])
def test_email_is_spelled_out_in_english(text, expected):
    assert verbalize(text, "en") == expected


def test_email_is_spelled_with_devanagari_letter_names_in_hindi():
    # Hindi TTS cannot say Latin letters, so letters become Devanagari names
    # and the well-known domain words are transliterated.
    result = verbalize("Mail devang.mishra@gmail.com for help.", "hi")

    assert result == "Mail डी ई वी ए एन जी डॉट एम आई एस एच आर ए, ऐट, जीमेल डॉट कॉम for help."
    assert "@" not in result


@pytest.mark.parametrize("text", ["Call +91 98123 45000 now", "Call +919812345000 now", "Call +91-98123-45000 now"])
def test_phone_with_country_code_is_read_as_grouped_digits(text):
    assert verbalize(text, "en") == "Call plus nine one, nine eight one two three, four five zero zero zero now"


@pytest.mark.parametrize("text", ["Call 9812345000 now", "Call 98123 45000 now", "Call 98123-45000 now"])
def test_phone_without_country_code_is_read_as_two_groups_of_five(text):
    assert verbalize(text, "en") == "Call nine eight one two three, four five zero zero zero now"


def test_phone_is_read_digit_by_digit_in_hindi():
    assert verbalize("Call +91 98123 45000 now", "hi") == "Call प्लस नौ एक, नौ आठ एक दो तीन, चार पांच शून्य शून्य शून्य now"


def test_account_id_is_spelled_letter_by_letter_then_digit_by_digit():
    assert verbalize("Your account BF-1001 is overdue", "en") == "Your account B F, one zero zero one is overdue"
    assert verbalize("Your account BF-1001 is overdue", "hi") == "Your account बी एफ, एक शून्य शून्य एक is overdue"


def test_percentages_whole_and_decimal():
    assert verbalize("Interest is 12.5% and fee is 2%", "en") == "Interest is twelve point five percent and fee is two percent"
    assert verbalize("Interest is 12.5% and fee is 2%", "hi") == "Interest is बारह दशमलव पांच प्रतिशत and fee is दो प्रतिशत"


def test_english_amounts_use_indian_grouping_lakh_and_crore():
    assert verbalize("Pay ₹1,25,000 now", "en") == "Pay one lakh, twenty-five thousand rupees now"
    assert verbalize("Pay ₹1,00,00,000 now", "en") == "Pay one crore rupees now"


def test_a_long_rupee_amount_is_not_mistaken_for_a_phone_number():
    # ₹9812345000 is ten digits starting with 9, i.e. phone-shaped. The
    # rupee symbol must win: amounts are verbalized before phones are looked at.
    result = verbalize("₹9812345000 is due", "en")

    assert "rupees" in result
    assert "nine eight one two three" not in result


def test_sentence_ending_full_stop_survives_after_email_and_phone():
    assert verbalize("Mail a@b.com.", "en").endswith("dot com.")
    assert verbalize("Call 9812345000.", "en").endswith("zero zero zero.")


@pytest.mark.parametrize("text", [
    "split 1/2 and 50/50",        # fractions/ratios are not dates or IDs
    "version 3.5.1 released",     # dotted version, not an amount or email
    "call 12345 for help",        # five digits: not a ten-digit mobile
    "ticket AB-12 is open",       # ID needs at least three digits
    "it is 3 days past due",      # small plain numbers already read fine
    "50 off today",               # a bare number is not a percentage
    "",
])
def test_text_without_a_supported_pattern_is_left_untouched(text):
    assert verbalize(text, "en") == text


@pytest.mark.parametrize("text", [
    "version 3.5.1 released",     # a dotted version is not a quantity
    "ticket AB-12 is open",       # an ID fragment
    "",
])
def test_hindi_leaves_versions_and_id_fragments_untouched(text):
    assert verbalize(text, "hi") == text


def test_hindi_reads_plain_numbers_in_otherwise_unsupported_text():
    # English is left alone (a person reads "3 days" fine), but the Hindi voice drops digits silently.
    assert verbalize("it is 3 days past due", "hi") == "it is तीन days past due"
    assert verbalize("50 off today", "hi") == "पचास off today"
    # five or more plain digits are a code, not a quantity: digit by digit
    assert verbalize("call 12345 for help", "hi") == "call एक दो तीन चार पांच for help"
    assert not any(ch.isdigit() for ch in verbalize("split 1/2 and 50/50", "hi"))


@pytest.mark.parametrize("text", ["@", "a@", "@b.com", "%", "-", "+91", "BF-", "9" * 40, "₹", "Rs."])
def test_malformed_input_passes_through_unchanged_and_never_raises(text):
    # Unrecognized shapes must pass through untouched; the function sits on
    # the TTS path of every voice reply and must not be able to crash it or
    # half-convert a fragment into garbage.
    for language in ("en", "hi"):
        assert verbalize(text, language) == text


_ALL_PATTERNS = (
    "Mail devang.mishra@gmail.com, call +91 98123 45000, account BF-1001, "
    "interest 12.5%, pay ₹1,25,000 by 2026-09-15."
)


@pytest.mark.parametrize("language", ["en", "hi"])
def test_verbalize_is_idempotent(language):
    # Re-verbalizing already-spoken text (e.g. a reply that passes two
    # layers) must not change it again.
    once = verbalize(_ALL_PATTERNS, language)

    assert verbalize(once, language) == once


@pytest.mark.parametrize("language", ["en", "hi"])
def test_nothing_the_stray_pattern_check_looks_for_survives_verbalize(language):
    assert has_unverbalized_pattern(_ALL_PATTERNS) is True
    assert has_unverbalized_pattern(verbalize(_ALL_PATTERNS, language)) is False


@pytest.mark.parametrize("raw", ["devang@gmail.com", "9812345000", "BF-1001", "12.5%"])
def test_has_unverbalized_pattern_detects_each_new_pattern_on_its_own(raw):
    assert has_unverbalized_pattern(f"see {raw} here") is True


# ---------------------------------------------------------------------------
# Hindi: plain numbers and UPI. The Hindi voice drops digits and Latin letters without a sound, so
# "तारीख 15 अक्टूबर" was spoken as "तारीख अक्टूबर" (found while probing why Hindi voice replies lost words).
# ---------------------------------------------------------------------------


def test_hindi_plain_numbers_become_hindi_words():
    assert verbalize("अंतिम तारीख 15 अक्टूबर", "hi") == "अंतिम तारीख पंद्रह अक्टूबर"
    assert verbalize("3 महीने बाकी हैं", "hi") == "तीन महीने बाकी हैं"


def test_hindi_grouped_and_decimal_numbers():
    assert verbalize("कुल 5,000 है", "hi") == "कुल पांच हज़ार है"
    assert verbalize("कुल 1,25,000 है", "hi") == "कुल एक लाख पच्चीस हज़ार है"
    assert verbalize("ब्याज 3.5 साल", "hi") == "ब्याज तीन दशमलव पांच साल"


def test_hindi_devanagari_digits_are_read_too():
    assert verbalize("तारीख १५", "hi") == "तारीख पंद्रह"


def test_hindi_list_punctuation_survives_the_number_conversion():
    assert verbalize("तारीख 5, 10 या 15।", "hi") == "तारीख पांच, दस या पंद्रह।"


def test_hindi_leaves_no_digit_behind_and_is_idempotent():
    text = "खाता BF-1001, किस्त ₹5,000, तारीख 2026-10-15, 3 दिन, फ़ोन 98765 43210, 18% ब्याज"
    once = verbalize(text, "hi")
    assert not any(ch.isdigit() for ch in once)
    assert verbalize(once, "hi") == once


def test_hindi_leaves_a_number_too_large_to_say_aloud_alone():
    assert verbalize("कोड 1000000000000", "hi") == "कोड 1000000000000"


def test_hindi_upi_is_spoken_in_devanagari():
    assert verbalize("UPI से भुगतान करें", "hi") == "यूपीआई से भुगतान करें"


def test_english_numbers_are_still_left_alone():
    assert verbalize("pay on the 15th, 3 times", "en") == "pay on the 15th, 3 times"

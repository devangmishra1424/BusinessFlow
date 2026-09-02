"""Converts numeric/date/currency substrings in agent-generated text into
words a TTS model can actually pronounce correctly, before it reaches TTS.
Regex finds the patterns the tool layer actually produces (ISO dates,
rupee amounts); num2words does the conversion. Everything else in the text
is left untouched -- small plain numbers already read fine without help.

language matters here: num2words has no Hindi converter at all (confirmed
against its own CONVERTER_CLASSES -- "hi" isn't in it, unlike "bn"/"kn"/
"te"), so English words were being substituted into text bound for
speak_hindi regardless of language. Real bug found live: MMS-TTS Hindi is
monolingual and can't pronounce those English number words (or a bare
English acronym like "EMI") embedded in Hindi text -- it silently drops
them, so a Hindi reply spoke the Hindi words around a number but never
the number itself. language="hi" now routes rupee amounts, dates, and a
small glossary of domain acronyms through Hindi words/transliteration
instead.
"""

import re

from num2words import num2words

_MONTH_NAMES_EN = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_NAMES_HI = [
    "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
]

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_RUPEE_AMOUNT = re.compile(r"(?:₹|Rs\.?)\s?([\d,]+(?:\.\d+)?)")

# Hindi's 0-99 aren't compositional the way English's tens+units are (21 is
# इक्कीस, not a "twenty"+"one" combination) -- num2words has no Hindi
# converter to fall back on (see module docstring), so this is a plain
# lookup table, the standard set taught in every Hindi number chart.
_HINDI_ONES_TO_NINETY_NINE = [
    "शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरेपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरेसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पंचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]

# Bare English acronyms that show up constantly in this domain's generated
# text and would otherwise sit untranslated in the middle of a Hindi
# sentence, unpronounceable by a monolingual Hindi model. Matched as whole
# words only, and deliberately small -- add an entry here only once it's
# actually been seen going unspoken (like EMI was, live), not
# speculatively for every acronym this domain happens to use.
_HINDI_GLOSSARY = {"EMI": "ईएमआई"}
_HINDI_GLOSSARY_PATTERN = re.compile(r"\b(" + "|".join(_HINDI_GLOSSARY) + r")\b")


def _hindi_number_words(n: int) -> str:
    """Indian-numbering-system (thousand/lakh/crore -- not the
    million/billion grouping num2words' English output uses) Hindi words
    for a non-negative integer."""
    if n < 100:
        return _HINDI_ONES_TO_NINETY_NINE[n]

    parts = []
    crore, n = divmod(n, 1_00_00_000)
    if crore:
        parts.append(f"{_hindi_number_words(crore)} करोड़")
    lakh, n = divmod(n, 1_00_000)
    if lakh:
        parts.append(f"{_hindi_number_words(lakh)} लाख")
    thousand, n = divmod(n, 1_000)
    if thousand:
        parts.append(f"{_hindi_number_words(thousand)} हज़ार")
    hundred, n = divmod(n, 100)
    if hundred:
        parts.append(f"{_HINDI_ONES_TO_NINETY_NINE[hundred]} सौ")
    if n:
        parts.append(_HINDI_ONES_TO_NINETY_NINE[n])
    return " ".join(parts)


def _verbalize_date(match: re.Match, language: str) -> str:
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if language == "hi":
        return f"{_hindi_number_words(day)} {_MONTH_NAMES_HI[month - 1]}, {_hindi_number_words(year)}"
    day_word = num2words(day, to="ordinal")
    month_word = _MONTH_NAMES_EN[month - 1]
    year_word = num2words(year, to="year")
    return f"the {day_word} of {month_word}, {year_word}"


def _verbalize_rupees(match: re.Match, language: str) -> str:
    # int()/float() parse Devanagari digits (०-९) natively, same as re's
    # \d already does -- a Hindi reply's own amount (e.g. "₹१७,०५३.५९") is
    # parsed correctly with no separate digit translation needed.
    amount = float(match.group(1).replace(",", ""))
    rupees = int(amount)
    paise = round((amount - rupees) * 100)
    if language == "hi":
        words = f"{_hindi_number_words(rupees)} रुपये"
        if paise:
            words += f" और {_hindi_number_words(paise)} पैसे"
        return words
    words = num2words(rupees) + " rupees"
    if paise:
        words += f" and {num2words(paise)} paise"
    return words


def verbalize(text: str, language: str = "en") -> str:
    """Rewrites ISO dates (YYYY-MM-DD), rupee amounts (₹12,500 or
    Rs. 12500), and a small glossary of domain acronyms into words --
    language="hi" for text headed to speak_hindi, "en" (the default) for
    speak_english. Passing the wrong language doesn't raise; it just
    produces the wrong-language words for TTS to (fail to) pronounce, so
    every call site must match its own speak_hindi/speak_english branch."""
    text = _ISO_DATE.sub(lambda m: _verbalize_date(m, language), text)
    text = _RUPEE_AMOUNT.sub(lambda m: _verbalize_rupees(m, language), text)
    if language == "hi":
        text = _HINDI_GLOSSARY_PATTERN.sub(lambda m: _HINDI_GLOSSARY[m.group(1)], text)
    return text

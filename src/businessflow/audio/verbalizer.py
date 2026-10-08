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

The same "reads fine to a human, mangled by TTS" problem applies to contact
details and identifiers the agent also produces: an email, a phone number,
an account ID like BF-1001, a percentage. A TTS model reads "BF-1001" as a
word-ish blob and a ten-digit number as one huge cardinal, neither of which
a listener can write down. These are now spelled out character by character
(letters as letter names, digits one at a time, phone numbers in 5+5
groups), in English or Hindi per `language`. English rupee amounts also use
Indian grouping (lakh/crore) via num2words' "en_IN", matching how a caller
actually says the number.
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
# Email: the domain part requires an alphanumeric after every dot, so the
# full stop that ends a sentence ("mail me at a@b.com.") is not swallowed.
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b")
# Indian mobile: optional +91, then 10 digits starting 6-9, optionally split
# 5+5 by a space or hyphen. Lookarounds keep it from biting into a longer
# digit run (an amount, an ID) or a word.
_PHONE = re.compile(r"(?<![\w.])(\+91[\s-]?)?([6-9]\d{4})[\s-]?(\d{5})(?![\w])")
_ACCOUNT_ID = re.compile(r"\b([A-Z]{2,5})-(\d{3,})\b")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")

# Letter names for spelling, and the symbols that occur in emails. Hindi
# TTS is monolingual (see module docstring), so Latin letters must be
# written as Devanagari letter names or they're silently dropped.
_LETTER_NAMES_HI = {
    "a": "ए", "b": "बी", "c": "सी", "d": "डी", "e": "ई", "f": "एफ", "g": "जी", "h": "एच",
    "i": "आई", "j": "जे", "k": "के", "l": "एल", "m": "एम", "n": "एन", "o": "ओ", "p": "पी",
    "q": "क्यू", "r": "आर", "s": "एस", "t": "टी", "u": "यू", "v": "वी", "w": "डब्ल्यू",
    "x": "एक्स", "y": "वाई", "z": "ज़ेड",
}
_SYMBOL_NAMES = {
    "en": {"@": "at", ".": "dot", "_": "underscore", "-": "dash", "+": "plus", "%": "percent"},
    "hi": {"@": "ऐट", ".": "डॉट", "_": "अंडरस्कोर", "-": "डैश", "+": "प्लस", "%": "प्रतिशत"},
}
# Very common mail domains and TLDs are spoken as words rather than spelled,
# the way a person would say them. Anything else is spelled out.
_DOMAIN_WORDS_EN = {"gmail", "yahoo", "outlook", "hotmail", "com", "org", "net", "in"}
_DOMAIN_WORDS_HI = {
    "gmail": "जीमेल", "yahoo": "याहू", "outlook": "आउटलुक", "hotmail": "हॉटमेल",
    "com": "कॉम", "org": "ऑर्ग", "net": "नेट", "in": "इन",
}

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
    # "en_IN" groups the way an Indian caller says it (one lakh twenty-five
    # thousand), not the western "one hundred twenty-five thousand".
    words = num2words(rupees, lang="en_IN") + " rupees"
    if paise:
        words += f" and {num2words(paise, lang='en_IN')} paise"
    return words


def _digit_words(digits: str, language: str) -> str:
    """Each digit spoken on its own: '1001' -> 'one zero zero one'. int()
    parses Devanagari digits as well as ASCII ones."""
    if language == "hi":
        return " ".join(_HINDI_ONES_TO_NINETY_NINE[int(d)] for d in digits)
    return " ".join(num2words(int(d)) for d in digits)


def _spell(text: str, language: str) -> str:
    """Letters as letter names, digits one at a time, the symbols that occur
    in emails and IDs by name. Callers only pass text already matched by one
    of the patterns above, so every character is one of those; anything else
    (whitespace) is deliberately dropped."""
    symbols = _SYMBOL_NAMES[language]
    parts = []
    for ch in text:
        if ch.isascii() and ch.isalpha():
            parts.append(_LETTER_NAMES_HI[ch.lower()] if language == "hi" else ch.upper())
        elif ch.isdigit():
            parts.append(_digit_words(ch, language))
        elif ch in symbols:
            parts.append(symbols[ch])
    return " ".join(parts)


def _verbalize_email(match: re.Match, language: str) -> str:
    local, _, domain = match.group(0).partition("@")
    symbols = _SYMBOL_NAMES[language]
    labels = []
    for label in domain.split("."):
        known = _DOMAIN_WORDS_HI.get(label.lower()) if language == "hi" else (label.lower() if label.lower() in _DOMAIN_WORDS_EN else None)
        labels.append(known or _spell(label, language))
    dot = f" {symbols['.']} "
    return f"{_spell(local, language)}, {symbols['@']}, {dot.join(labels)}"


def _verbalize_phone(match: re.Match, language: str) -> str:
    country, first, second = match.group(1), match.group(2), match.group(3)
    groups = [_digit_words(first, language), _digit_words(second, language)]
    if country:
        groups.insert(0, f"{_SYMBOL_NAMES[language]['+']} {_digit_words('91', language)}")
    return ", ".join(groups)


def _verbalize_account_id(match: re.Match, language: str) -> str:
    return f"{_spell(match.group(1), language)}, {_digit_words(match.group(2), language)}"


def _verbalize_percent(match: re.Match, language: str) -> str:
    whole, _, fraction = match.group(1).partition(".")
    word = _SYMBOL_NAMES[language]["%"]
    if language == "hi":
        words = _hindi_number_words(int(whole))
        if fraction:
            words += f" दशमलव {_digit_words(fraction, 'hi')}"
    else:
        words = num2words(int(whole))
        if fraction:
            words += f" point {_digit_words(fraction, 'en')}"
    return f"{words} {word}"


def has_unverbalized_pattern(text: str) -> bool:
    """True if text still contains an ISO date, rupee amount, email, phone
    number, account ID or percentage verbalize() is supposed to catch --
    i.e. verbalize() either was never called on this text, or (a real
    regression) failed to actually remove one. Used by
    eval/voice_naturalness_benchmark.py to confirm the exact live bug this
    module's own docstring describes (a number silently reaching TTS
    unconverted) hasn't come back, without duplicating the regex patterns
    that already live here."""
    return any(p.search(text) for p in (_ISO_DATE, _RUPEE_AMOUNT, _EMAIL, _PHONE, _ACCOUNT_ID, _PERCENT))


def verbalize(text: str, language: str = "en") -> str:
    """Rewrites ISO dates (YYYY-MM-DD), rupee amounts (₹12,500 or
    Rs. 12500), emails, Indian mobile numbers, account IDs (BF-1001),
    percentages, and a small glossary of domain acronyms into words --
    language="hi" for text headed to speak_hindi, "en" (the default) for
    speak_english. Passing the wrong language doesn't raise; it just
    produces the wrong-language words for TTS to (fail to) pronounce, so
    every call site must match its own speak_hindi/speak_english branch.

    Order matters: dates and rupee amounts first (they contain digit runs the
    phone/ID patterns must not see), then emails (which may contain either).
    The output contains none of the patterns, so verbalize() is idempotent."""
    text = _ISO_DATE.sub(lambda m: _verbalize_date(m, language), text)
    text = _RUPEE_AMOUNT.sub(lambda m: _verbalize_rupees(m, language), text)
    text = _EMAIL.sub(lambda m: _verbalize_email(m, language), text)
    text = _PHONE.sub(lambda m: _verbalize_phone(m, language), text)
    text = _ACCOUNT_ID.sub(lambda m: _verbalize_account_id(m, language), text)
    text = _PERCENT.sub(lambda m: _verbalize_percent(m, language), text)
    if language == "hi":
        text = _HINDI_GLOSSARY_PATTERN.sub(lambda m: _HINDI_GLOSSARY[m.group(1)], text)
    return text

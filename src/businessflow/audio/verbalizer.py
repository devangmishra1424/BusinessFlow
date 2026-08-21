"""Converts numeric/date/currency substrings in agent-generated text into
words a TTS model can actually pronounce correctly, before it reaches TTS.
Regex finds the patterns the tool layer actually produces (ISO dates,
rupee amounts); num2words does the conversion. Everything else in the text
is left untouched -- small plain numbers already read fine without help.
"""

import re

from num2words import num2words

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_RUPEE_AMOUNT = re.compile(r"(?:₹|Rs\.?)\s?([\d,]+(?:\.\d+)?)")


def _verbalize_date(match: re.Match) -> str:
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    day_word = num2words(day, to="ordinal")
    month_word = _MONTH_NAMES[month - 1]
    year_word = num2words(year, to="year")
    return f"the {day_word} of {month_word}, {year_word}"


def _verbalize_rupees(match: re.Match) -> str:
    amount = float(match.group(1).replace(",", ""))
    rupees = int(amount)
    paise = round((amount - rupees) * 100)
    words = num2words(rupees) + " rupees"
    if paise:
        words += f" and {num2words(paise)} paise"
    return words


def verbalize(text: str) -> str:
    """Rewrites ISO dates (YYYY-MM-DD) and rupee amounts (₹12,500 or
    Rs. 12500) in text into words."""
    text = _ISO_DATE.sub(_verbalize_date, text)
    text = _RUPEE_AMOUNT.sub(_verbalize_rupees, text)
    return text

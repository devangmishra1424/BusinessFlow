"""Was this figure actually SAID? The check that stands between a language
model and money: ops voice (ops/voice.py) lets a model turn "record twelve
and a half thousand from Fatima" into record_payment(amount=12500), and a
model can mishear, round, or simply invent a figure. So before any action is
even proposed, every amount, phone number and figure in a drafted message is
looked for in the operator's own words (the transcript, or what they typed),
and a figure that isn't there is never put in front of them to confirm.

Reads the ways an amount is said in Indian English: digits with Western or
Indian commas ("12,500", "1,20,000"), k / thousand / lakh / crore ("12k",
"1.2 lakh", "2 lakh 50 thousand"), and number words ("twelve thousand five
hundred", "twelve and a half thousand", "half a lakh"). Lenient by design
about how a phrase can be split ("five thousand and two thousand" counts as
5000, 2000 and 7000): the job is to catch a figure that appears NOWHERE in
what was said, not to decide which reading the speaker meant -- the
confirmation card shows the figure, and the operator decides that.

Hindi too, because operators speak it (found live: a question asked in Hindi
came back from Whisper in Devanagari, and the model understood it): Devanagari
digits, सौ / हज़ार / लाख / करोड़ (and hazaar / sau), the common number words --
one to twenty and the round tens and fives up to ninety-five -- and साढ़े /
सवा / डेढ़ / ढाई ("साढ़े बारह हज़ार" is 12,500). Any other Hindi number word
(तेईस, say) is not read, so a figure said that way is asked for again rather
than guessed.
"""

import re
import unicodedata

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {
    "hundred": 100, "thousand": 1_000, "k": 1_000, "lakh": 1_00_000, "lakhs": 1_00_000, "lac": 1_00_000,
    "lacs": 1_00_000, "million": 1_000_000, "crore": 1_00_00_000, "crores": 1_00_00_000,
}
_CONNECTORS = {"and", "a", "half", "point"}
_DIGIT_WORDS = {**{w: str(n) for w, n in _UNITS.items() if n < 10}, "oh": "0"}


def _nfc(text: str) -> str:
    # One spelling per word: हज़ार can arrive with a precomposed ज़ or as ज + nukta.
    return unicodedata.normalize("NFC", text)


# Hindi number words that stand alone, read like digits; 1.5 and 2.5 have words of their own.
_HINDI_VALUES = {_nfc(w): v for w, v in {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6, "छे": 6, "सात": 7, "आठ": 8,
    "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "पन्द्रह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20, "पच्चीस": 25, "तीस": 30, "पैंतीस": 35, "चालीस": 40,
    "पैंतालीस": 45, "पचास": 50, "पचपन": 55, "साठ": 60, "पैंसठ": 65, "सत्तर": 70, "पचहत्तर": 75, "अस्सी": 80,
    "पचासी": 85, "नब्बे": 90, "पचानवे": 95, "डेढ़": 1.5, "ढाई": 2.5, "dedh": 1.5, "derh": 1.5, "dhai": 2.5, "dhaai": 2.5,
}.items()}
_HINDI_SCALES = {_nfc(w): v for w, v in {
    "सौ": 100, "हज़ार": 1_000, "हजार": 1_000, "लाख": 1_00_000, "करोड़": 1_00_00_000, "करोड": 1_00_00_000,
    "sau": 100, "hazaar": 1_000, "hazar": 1_000, "hajar": 1_000, "hazaar": 1_000,
}.items()}
_SCALES = {**_SCALES, **_HINDI_SCALES}
# साढ़े बारह = 12.5, सवा लाख = 1.25 lakh: added to the number that follows.
_ADD_TO_NEXT = {_nfc(w): v for w, v in {
    "साढ़े": 0.5, "सवा": 0.25, "sade": 0.5, "saade": 0.5, "sadhe": 0.5, "saadhe": 0.5, "sava": 0.25, "sawa": 0.25,
}.items()}

_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+|[ऀ-ॣॱ-ॿ]+")  # Devanagari letters and signs, not its danda or digits
_DIGIT_GROUP_COMMA = re.compile(r"(?<=\d),(?=\d)")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _tokens(text: str) -> list[str]:
    # "12,500" and "1,20,000" are one number; "12k" and "1.2lakh" are a number and a scale word; "१२५००" is 12500.
    return _TOKEN.findall(_DIGIT_GROUP_COMMA.sub("", _nfc(text).lower().translate(_DEVANAGARI_DIGITS)))


def _is_numberish(token: str) -> bool:
    return (token[0].isdigit() or token in _UNITS or token in _TENS or token in _SCALES or token in _CONNECTORS
            or token in _HINDI_VALUES or token in _ADD_TO_NEXT)


def _runs(tokens: list[str]) -> list[list[str]]:
    """Maximal stretches of number-ish tokens -- "paid 12 thousand 500 on the 10th" -> [12, thousand, 500], [10]."""
    runs, run = [], []
    for token in tokens:
        if _is_numberish(token):
            run.append(token)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return runs


def _phrase_values(run: list[str]) -> list[float]:
    """The figures in one run. Two bare numbers in a row ("12 500") are two figures, not one."""
    values: list[float] = []
    total = current = 0.0
    started = decimal = False
    place = 0.1
    add_to_next = 0.0  # from साढ़े / सवा
    previous = None  # what the previous token was: "num" | "tens" | "and" | "a" | "half" | "scale" | "point"

    def flush():
        nonlocal total, current, started, decimal, place
        if started:
            values.append(total + current)
        total = current = 0.0
        started = decimal = False
        place = 0.1

    for i, token in enumerate(run):
        if token in _ADD_TO_NEXT:
            add_to_next += _ADD_TO_NEXT[token]
            continue
        if token[0].isdigit() or token in _UNITS or token in _TENS or token in _HINDI_VALUES:
            n = float(token) if token[0].isdigit() else float(_UNITS.get(token, _TENS.get(token, _HINDI_VALUES.get(token, 0))))
            n, add_to_next = n + add_to_next, 0.0
            if decimal and token in _UNITS:  # "one point two five"
                current += n * place
                place /= 10
            elif previous == "tens" and token in _UNITS and n < 10:  # "twenty five"
                current += n
            elif started and current and previous != "and":  # "12 500": a second figure starts here
                flush()
                current, started = n, True
            else:  # a fresh figure, after a scale ("12 thousand 500"), or after "and" ("one hundred and five")
                current += n
                started = True
            previous = "tens" if token in _TENS else "num"
        elif token == "point":
            decimal = started
            previous = "point"
        elif token == "half":  # "twelve and a half thousand" / "half a lakh"
            current += 0.5
            started = True
            previous = "half"
        elif token == "a":
            if i + 1 < len(run) and run[i + 1] in _SCALES and not current:  # "a thousand", "a lakh"
                current, started = 1.0, True
            previous = "a"
        elif token == "and":
            previous = "and"
        elif token in _SCALES and (started or add_to_next):
            if not started:  # "सवा लाख" is 1.25 lakh: the "one" goes unsaid
                current, started, add_to_next = 1 + add_to_next, True, 0.0
            scale = _SCALES[token]
            if scale == 100:
                current = (current or 1) * 100
            elif current:
                total += current * scale
                current = 0.0
            else:  # "1 hundred thousand": the hundreds were already folded into total
                total *= scale
            decimal = False
            previous = "scale"
    flush()
    return values


def spoken_figures(text: str) -> set[float]:
    """Every figure that can be read out of text (see the module docstring on leniency)."""
    figures: set[float] = set()
    for run in _runs(_tokens(text)):
        figures.update(_phrase_values(run))
        segment: list[str] = []
        for token in run + ["and"]:  # each part between "and"s on its own, too
            if token == "and":
                figures.update(_phrase_values(segment))
                segment = []
            else:
                segment.append(token)
        figures.update(float(t) for t in run if t[0].isdigit())
        # "12 500": a transcript can write a thousands separator as a space
        figures.update(
            float(a + b) for a, b in zip(run, run[1:]) if a.isdigit() and b.isdigit() and len(b) == 3
        )
    return figures


def was_said(amount: float, text: str) -> bool:
    """True if amount (to the paisa) is a figure in text."""
    return any(abs(amount - figure) < 0.005 for figure in spoken_figures(text))


def unsaid_figures(drafted: str, said: str) -> list[float]:
    """Figures in a drafted text (a message to a borrower, a note) that were not in what the operator said, in the order
    they appear. A lone "one" or "a" is not counted ("one of our team", "a call"): those read as words, not figures.
    A run that joins two figures with "and" passes if either reading was said: "₹5,000 and one of our team" is 5000,
    not 5001."""
    heard = spoken_figures(said)

    def is_heard(value: float) -> bool:
        return any(abs(value - figure) < 0.005 for figure in heard)

    def is_figure(part: list[str]) -> bool:
        return not all(t in ("one", "a", "point") for t in part)

    missing: list[float] = []
    for run in _runs(_tokens(drafted)):
        segments, part = [], []
        for token in run + ["and"]:
            if token == "and":
                if part and is_figure(part):
                    segments.append(part)
                part = []
            else:
                part.append(token)
        segment_values = [v for segment in segments for v in _phrase_values(segment)]
        whole_values = _phrase_values(run) if is_figure(run) else []
        if all(is_heard(v) for v in whole_values) or all(is_heard(v) for v in segment_values):
            continue
        missing.extend(v for v in segment_values if not is_heard(v) and v not in missing)
    return missing


def spoken_digits(text: str) -> str:
    """Every digit said, in order, as one string -- "nine eight one two 345 001" -> "9812345001". For checking a phone
    number against what was said; "double five" / "triple zero" are read the way people dictate numbers."""
    out = []
    repeat = 1
    for token in _TOKEN.findall(_nfc(text).lower().translate(_DEVANAGARI_DIGITS).replace(",", "")):
        if token in ("double", "triple"):
            repeat = 2 if token == "double" else 3
            continue
        if token[0].isdigit():
            out.append(token.replace(".", "") * repeat if repeat > 1 and len(token) == 1 else token.replace(".", ""))
        elif token in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[token] * repeat)
        repeat = 1
    return "".join(out)

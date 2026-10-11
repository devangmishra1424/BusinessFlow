"""Parsing and the readability gate of rag/ingest.py -- everything that happens BEFORE the database is touched, so none
of this needs one. Found by running the real parser on rendered scans: a scan the bundled OCR cannot read (a blank page,
a Hindi/Devanagari page) used to come back as "<!-- image -->" or a few garbage characters, and was stored as the
document -- replacing a good one with nothing, with no error.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFont

from businessflow.rag import ingest
from businessflow.rag.ingest import ChunkRecord, UnreadableDocumentError, _require_readable, parse_and_chunk


def _result(ocr_score):
    return SimpleNamespace(confidence=SimpleNamespace(ocr_score=ocr_score))


def _records(text):
    return [ChunkRecord(text=text, headings="")]


GOOD_TEXT = "The Loan carries interest at 14.5% per annum on a reducing balance."


def test_normal_text_with_a_confident_ocr_passes():
    _require_readable(_result(0.99), _records(GOOD_TEXT), "agreement.pdf")


def test_born_digital_files_have_no_ocr_score_and_are_not_penalised_for_it():
    _require_readable(_result(math.nan), _records(GOOD_TEXT), "agreement.docx")
    _require_readable(SimpleNamespace(), _records(GOOD_TEXT), "agreement.md")  # no confidence report at all


@pytest.mark.parametrize("text", ["", "   \n  ", "<!-- image -->", "<!-- image -->\n<!-- image -->", "-T IIr", "h"])
def test_a_page_with_no_usable_text_is_refused(text):
    with pytest.raises(UnreadableDocumentError, match="no readable text"):
        _require_readable(_result(math.nan), _records(text), "scan.pdf")


def test_the_minimum_is_eight_letters_and_digits_and_a_heading_counts():
    _require_readable(_result(math.nan), _records("a" * 8), "x.md")
    with pytest.raises(UnreadableDocumentError):
        _require_readable(_result(math.nan), _records("a" * 7), "x.md")
    # a document that is only a title is still text
    _require_readable(_result(math.nan), [ChunkRecord(text="", headings="Late fee policy")], "x.md")


def test_a_scan_the_ocr_is_unsure_about_is_refused_with_a_message_that_says_what_to_do():
    # 0.50-0.57 is what the bundled OCR reported on Devanagari pages; English scans scored 0.93-0.99.
    with pytest.raises(UnreadableDocumentError) as error:
        _require_readable(_result(0.55), _records("garbled but long enough to pass the length check " * 3), "hindi_scan.pdf")
    assert "hindi_scan.pdf" in str(error.value) and "Hindi" in str(error.value) and "text version" in str(error.value)


def test_a_markdown_file_is_chunked_with_its_heading_path(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text("# Late fee policy\n\n## Amount\n\nA flat late fee of 500 rupees applies once per cycle.\n", encoding="utf-8")

    records = parse_and_chunk(str(path))

    assert [r.headings for r in records] == ["Late fee policy > Amount"]
    assert "500 rupees" in records[0].text
    assert records[0].embedding_text.startswith("Late fee policy\nAmount\n")


def test_an_empty_markdown_file_is_refused_not_stored_as_zero_chunks(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(UnreadableDocumentError):
        parse_and_chunk(str(path))


def test_extracting_the_text_after_ingesting_does_not_parse_the_file_a_second_time(tmp_path, monkeypatch):
    path = tmp_path / "agreement.md"
    path.write_text("# Agreement\n\nThe borrower agrees to repay the principal over twenty four months.\n", encoding="utf-8")
    real = ingest._converter()
    calls = []

    class CountingConverter:
        def convert(self, file_path):
            calls.append(file_path)
            return real.convert(file_path)

    monkeypatch.setattr(ingest, "_converter", lambda: CountingConverter())
    ingest._recent_conversions.clear()

    parse_and_chunk(str(path))
    text = ingest.extract_document_text(str(path))

    assert "twenty four months" in text
    assert len(calls) == 1


def test_a_changed_file_is_parsed_again(tmp_path, monkeypatch):
    path = tmp_path / "agreement.md"
    path.write_text("# Agreement\n\nThe borrower agrees to repay within twenty four months.\n", encoding="utf-8")
    ingest._recent_conversions.clear()
    first = parse_and_chunk(str(path))
    path.write_text("# Agreement\n\nThe borrower agrees to repay within thirty six months, said the corrected copy.\n", encoding="utf-8")

    second = parse_and_chunk(str(path))

    assert "twenty four" in first[0].text and "thirty six" in second[0].text


# ---------------------------------------------------------------------------------------------------------------------
# The real parser and the real OCR, on scans rendered here (so the right answer is known).
# ---------------------------------------------------------------------------------------------------------------------
def _font(candidates, size):
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return None


_LATIN_FONTS = ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
_DEVANAGARI_FONTS = ["Nirmala.ttc", "NotoSansDevanagari-Regular.ttf", "Lohit-Devanagari.ttf", "mangal.ttf",
                     "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"]


def _scan_pdf(path: Path, lines: list[str], font) -> str:
    image = Image.new("L", (1700, 900), 255)
    draw = ImageDraw.Draw(image)
    y = 60
    for line in lines:
        draw.text((70, y), line, font=font, fill=0)
        y += 70
    image.save(path, "PDF", resolution=150)
    return str(path)


def test_a_blank_scanned_page_is_refused(tmp_path):
    blank = Image.new("L", (1240, 900), 255)
    path = tmp_path / "blank_scan.pdf"
    blank.save(path, "PDF", resolution=150)

    with pytest.raises(UnreadableDocumentError, match="no readable text"):
        parse_and_chunk(str(path))


def test_a_clean_english_scan_is_read_with_its_figures_intact(tmp_path):
    font = _font(_LATIN_FONTS, 34)
    if font is None:
        pytest.skip("no TrueType font available to render the scan")
    # short lines: a wider font (CI renders with DejaVu) must not push the end of a line off the page
    path = _scan_pdf(tmp_path / "english_scan.pdf", [
        "Principal Amount: Rs. 250,000.",
        "Tenure: 24 months.",
        "Monthly EMI Amount: Rs. 12,500.00.",
        "Interest at 14.5% per annum.",
    ], font)

    text = " ".join(r.text for r in parse_and_chunk(path))

    squash = lambda t: "".join(t.split())  # OCR spacing is not what this test is about ("24months" is still read)
    for figure in ("250,000", "24 months", "12,500", "14.5"):
        assert squash(figure) in squash(text), f"{figure!r} lost in OCR: {text!r}"


def test_a_hindi_scan_is_refused_because_the_bundled_ocr_cannot_read_devanagari(tmp_path):
    font = _font(_DEVANAGARI_FONTS, 38)
    if font is None:
        pytest.skip("no Devanagari font available to render the scan")
    path = _scan_pdf(tmp_path / "hindi_scan.pdf", [
        "ऋण समझौता - सहयोग व्यापार फाइनेंस",
        "मासिक किस्त बारह हज़ार पाँच सौ रुपये है।",
    ], font)

    with pytest.raises(UnreadableDocumentError):
        parse_and_chunk(path)

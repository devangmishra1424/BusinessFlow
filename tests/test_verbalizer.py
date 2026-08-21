from businessflow.audio.verbalizer import verbalize


def test_verbalizes_rupee_amount_with_rs_prefix():
    assert verbalize("Rs. 12500 is due") == "twelve thousand, five hundred rupees is due"


def test_verbalizes_rupee_amount_with_symbol_and_paise():
    assert verbalize("pay ₹585,200.50 now") == "pay five hundred and eighty-five thousand, two hundred rupees and fifty paise now"


def test_verbalizes_iso_date():
    assert verbalize("due on 2026-08-18") == "due on the eighteenth of August, twenty twenty-six"


def test_leaves_small_plain_numbers_untouched():
    assert verbalize("it is 3 days past due") == "it is 3 days past due"

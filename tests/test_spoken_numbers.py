"""ops/spoken_numbers.py: the check that an amount the voice assistant is about
to put on a confirmation card was actually said. Pure functions, no model, no
database. The failure path matters most here: a figure that was NOT said must
never be found, however the model phrased it."""

import pytest

from businessflow.ops.spoken_numbers import spoken_digits, unsaid_figures, was_said


@pytest.mark.parametrize("said, amount", [
    ("record 12,500 for Fatima", 12500),
    ("record 12500 for Fatima", 12500),
    ("twelve thousand five hundred", 12500),
    ("twelve and a half thousand", 12500),
    ("12 thousand 500", 12500),
    ("12k from her", 12000),
    ("1.2 lakh", 120000),
    ("one lakh twenty thousand", 120000),
    ("₹1,20,000 by NEFT", 120000),
    ("2 lakh 50 thousand", 250000),
    ("half a lakh", 50000),
    ("a thousand rupees", 1000),
    ("one hundred and five", 105),
    ("twenty five thousand", 25000),
    ("thirty five hundred", 3500),
    ("one point two five lakh", 125000),
    ("Rs. 13,250.50 today", 13250.50),
    ("BF-1003 paid 12 500", 12500),  # a transcript can write the thousands separator as a space
])
def test_amounts_said_in_the_usual_indian_english_ways_are_found(said, amount):
    assert was_said(amount, said)


@pytest.mark.parametrize("said, amount", [
    ("record twelve thousand from fatima", 12500),  # the model rounded or misheard
    ("record a payment from fatima", 5000),  # no figure at all -- invented
    ("paid 1.2 lakh", 12000),  # wrong scale
    ("paid on the 10th", 1000),
    ("record 12,500", 12500.5),  # to the paisa
])
def test_an_amount_that_was_not_said_is_not_found(said, amount):
    assert not was_said(amount, said)


@pytest.mark.parametrize("said, amount", [
    ("फातिमा से बारह हज़ार पाँच सौ रिकॉर्ड करो", 12500),
    ("साढ़े बारह हज़ार", 12500),  # 12.5 thousand
    ("डेढ़ लाख", 150000),
    ("ढाई लाख", 250000),
    ("सवा लाख", 125000),
    ("१२५०० जमा किया", 12500),  # Devanagari digits
    ("12 हज़ार दिए", 12000),
    ("पच्चीस हजार", 25000),  # हजार without the nukta
    ("saade 12 hazaar", 12500),
])
def test_amounts_said_in_hindi_are_found(said, amount):
    assert was_said(amount, said)


def test_a_hindi_number_word_outside_the_list_is_not_guessed():
    # तेईस (23) is not read: the amount is asked for again rather than matched to something else
    assert not was_said(23000, "तेईस हज़ार")
    assert not was_said(1000, "तेईस हज़ार")


def test_a_figure_the_model_added_to_a_drafted_message_is_reported():
    said = "tell her we got her payment and someone will call her"
    drafted = "We have received your payment of ₹5,000 and one of our team will call you within two days."
    assert unsaid_figures(drafted, said) == [5000, 2]


def test_figures_the_operator_said_and_plain_words_like_one_of_are_not_reported():
    said = "tell fatima we got her 12,500 and will update it by the 20th"
    drafted = "We have received your ₹12,500 and one of our team will update your account by the 20th."
    assert unsaid_figures(drafted, said) == []


def test_a_figure_joined_to_a_word_by_and_is_read_the_way_it_was_meant():
    # "5,000 and one of our team" is 5000 then words, not 5001
    assert unsaid_figures("Your ₹5,000 and one of our team", "we got her 5000") == []


def test_spoken_digits_reads_a_dictated_phone_number():
    assert spoken_digits("nine eight one two three four five double zero one") == "9812345001"
    assert spoken_digits("change it to 98123 45001") == "9812345001"
    assert spoken_digits("plus nine one nine eight triple seven") == "9198777"

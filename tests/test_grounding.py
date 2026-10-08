"""Unit tests for the Guardrail's grounding check -- pure logic, no DB or
LLM. Covers the exact bug that motivated this (a fabricated payment
link), plus messy real-world formatting: trailing punctuation on URLs,
comma-formatted amounts, amounts that only appear as plain JSON numbers
in a tool result (not rupee-prefixed).
"""

from businessflow.guardrail.grounding import (
    check_fabricated_action,
    check_grounding,
    check_unapplied_restructuring_claim,
    extract_rupee_amounts,
    extract_urls,
)


def test_extract_urls_strips_trailing_punctuation():
    text = "Here's your link: https://demo.businessflow.local/pay/BF-1004?amount=28000.0."
    assert extract_urls(text) == {"https://demo.businessflow.local/pay/BF-1004?amount=28000.0"}


def test_extract_urls_strips_angle_bracket_wrapping():
    # Real bug, found live: the model wrapped a real, correct link in
    # angle brackets ("<https://...20000.0>", a common way to make a URL
    # unambiguous in plain text) and the guardrail blocked its own
    # correct reply because the trailing '>' made it fail to match the
    # clean URL from the tool result.
    text = "Here: <https://demo.businessflow.local/pay/BF-1003?amount=20000.0>"
    assert extract_urls(text) == {"https://demo.businessflow.local/pay/BF-1003?amount=20000.0"}


def test_extract_urls_handles_markdown_link_syntax():
    # Real bug, found live: the model wrote a Markdown link with no
    # whitespace between the link text and the href --
    # "[https://...20000.0](https://...20000.0)" -- which a bare \S+
    # swallows straight through the "](" into one long, unmatchable
    # "URL". Excluding [ and ] from the URL body fixes it.
    text = ("Here: [https://demo.businessflow.local/pay/BF-1003?amount=20000.0]"
            "(https://demo.businessflow.local/pay/BF-1003?amount=20000.0)")
    assert extract_urls(text) == {"https://demo.businessflow.local/pay/BF-1003?amount=20000.0"}


def test_extract_rupee_amounts_handles_commas():
    text = "Your balance is ₹1,40,000 and the fee is ₹500."
    assert extract_rupee_amounts(text) == {140000.0, 500.0}


def test_fabricated_link_is_caught():
    # The exact real bug: model invents a URL instead of calling the tool.
    conversation = [
        {"role": "user", "content": "just send me the payment link"},
        {"role": "tool", "content": '{"account_id": "BF-1004", "days_past_due": 3}'},
    ]
    reply = "Here you go: https://payments.example.com/pay?acc=BF-1004&amount=28000"

    failure = check_grounding(reply, conversation)

    assert failure
    assert "payments.example.com" in failure.describe()


def test_real_tool_generated_link_passes():
    conversation = [
        {"role": "user", "content": "send me the link"},
        {"role": "tool", "content": '{"account_id": "BF-1004", "amount": 28000, "payment_link": "https://demo.businessflow.local/pay/BF-1004?amount=28000", "synthetic": true}'},
    ]
    reply = "Here's your link: https://demo.businessflow.local/pay/BF-1004?amount=28000"

    failure = check_grounding(reply, conversation)

    assert not failure


def test_fabricated_amount_is_caught():
    conversation = [
        {"role": "user", "content": "how much to settle"},
        {"role": "tool", "content": '{"settlement_amount": 418000.0, "discount_pct": 0.05}'},
    ]
    reply = "You'd need to pay ₹500,000 to settle in full."  # made up, doesn't match the tool's 418000

    failure = check_grounding(reply, conversation)

    assert failure
    assert 500000.0 in failure.ungrounded_amounts


def test_amount_matching_a_plain_json_number_in_a_tool_result_passes():
    # Tool results store amounts as plain JSON numbers (28000.0), not
    # rupee-prefixed text -- the reply restating it with a rupee sign
    # must still be recognized as grounded.
    conversation = [
        {"role": "user", "content": "whats my balance"},
        {"role": "tool", "content": '{"emi_amount": 28000.0, "days_past_due": 3}'},
    ]
    reply = "Your EMI is ₹28,000 and it's 3 days overdue."

    failure = check_grounding(reply, conversation)

    assert not failure


def test_amount_grounded_two_turns_ago_still_passes():
    # A value established earlier in the conversation, not just this
    # turn, is still grounded -- scoping to "this turn only" would
    # false-positive on ordinary multi-turn context reuse.
    conversation = [
        {"role": "user", "content": "whats my emi"},
        {"role": "tool", "content": '{"emi_amount": 28000.0}'},
        {"role": "assistant", "content": "Your EMI is ₹28,000."},
        {"role": "user", "content": "ok just send the link for that then"},
    ]
    reply = "Sure, here's the link for ₹28,000: https://demo.businessflow.local/pay/BF-1004?amount=28000"
    # the link URL itself isn't grounded here (no tool call happened this
    # turn) -- confirms the amount passes independently of the URL failing
    failure = check_grounding(reply, conversation)

    assert not failure.ungrounded_amounts
    assert failure.ungrounded_urls  # the URL genuinely isn't grounded in this conversation


def test_amount_the_borrower_themselves_stated_is_not_flagged():
    # A number the borrower said isn't an "invention" by the agent.
    conversation = [
        {"role": "user", "content": "can i pay ₹15000 instead of the full amount"},
    ]
    reply = "Let me check if ₹15,000 works as a partial payment."

    failure = check_grounding(reply, conversation)

    assert not failure


def test_devanagari_amount_with_a_stray_internal_space_is_not_split():
    # Real bug, found live: the model wrote a Devanagari-numeral amount
    # with a stray space instead of a separator -- "₹५ ८५,२००" meaning
    # one number, ५,८५,२०० = 585200 -- which a naive regex splits into
    # an isolated "५" (5) that correctly matches nothing.
    conversation = [
        {"role": "user", "content": "loan settle karne ka kharcha kya hoga"},
        {"role": "tool", "content": '{"settlement_amount": 585200.0, "remaining_principal": 616000.0}'},
    ]
    reply = "आपको कुल ₹५ ८५,२०० की एकमुश्त राशि देना होगी, जो आपके शेष ₹६,१६,००० पर ५ % की छूट के बाद है।"

    failure = check_grounding(reply, conversation)

    assert not failure


def test_rounding_a_grounded_amount_to_the_nearest_rupee_is_not_flagged():
    # Real bug, found live on a real Telegram conversation: the model
    # correctly stated a real, grounded outstanding balance
    # (₹1,084,741.92, straight from get_payment_status), and a moment
    # later restated it rounded to the nearest rupee for readability --
    # "about ₹1,084,742", exactly how a person reads a balance aloud --
    # and got blocked, because the tolerance was 0.01 (one paisa) and
    # 1084742 - 1084741.92 = 0.08. The borrower got a vague non-answer
    # instead of the real number they asked for, twice in the same
    # conversation.
    conversation = [
        {"role": "user", "content": "mera outstanding kitna hai"},
        {"role": "tool", "content": '{"outstanding_balance_approx": 1084741.92}'},
    ]
    reply = "Your loan balance is about ₹1,084,742 (rounded)."

    failure = check_grounding(reply, conversation)

    assert not failure


def test_an_amount_genuinely_unrelated_to_any_grounded_figure_is_still_flagged():
    # The tolerance fix above must not turn into "anything in the same
    # ballpark passes" -- a real invention nowhere near a grounded figure
    # still needs to be caught.
    conversation = [
        {"role": "tool", "content": '{"outstanding_balance_approx": 1084741.92}'},
    ]
    reply = "Your loan balance is about ₹1,200,000."

    failure = check_grounding(reply, conversation)

    assert failure.ungrounded_amounts == {1200000.0}


def test_shorthand_k_amount_from_the_user_grounds_the_agents_formal_restatement():
    # Real bug, found live: user says "maybe 15k", the model correctly
    # asks a confirming question restating it as "₹15,000" (exactly the
    # hedge-handling behavior this project's commitment-discipline prompt
    # fix was for) -- and the guardrail blocked its own correct question
    # because "15k" and "15000" don't look alike as plain text.
    conversation = [
        {"role": "user", "content": "honestly things are tight, maybe 15k for now? not sure"},
    ]
    reply = "Got it. Will you be able to pay ₹15,000 by the 20th of this month?"

    failure = check_grounding(reply, conversation)

    assert not failure


def test_shorthand_lakh_amount_grounds_correctly():
    conversation = [{"role": "user", "content": "I have around 1.5L saved up, would that cover a settlement?"}]
    reply = "₹1,50,000 would be a good start toward that."

    failure = check_grounding(reply, conversation)

    assert not failure


def test_ordinary_descriptive_numbers_without_rupee_sign_are_never_flagged():
    # "3 days", "5%" etc -- deliberately not checked at all, since a
    # blanket every-number check would false-positive constantly.
    conversation = [{"role": "user", "content": "hi"}]
    reply = "You have a 3 day grace period and there's a 5% discount available."

    failure = check_grounding(reply, conversation)

    assert not failure


def test_fabricated_action_catches_the_real_phrasing_found_live():
    # The exact two phrasings seen live via the Telegram channel -- the
    # second one restating the first, two turns later, as settled fact.
    assert check_fabricated_action("I've also arranged to send a copy of the full loan agreement to you.")
    assert check_fabricated_action("I've already sent a copy of the full agreement to your email.")


def test_fabricated_action_catches_a_few_real_variants():
    assert check_fabricated_action("I have emailed the document to you.")
    assert check_fabricated_action("I've forwarded the contract to your inbox.")
    assert check_fabricated_action("Here is the agreement I sent earlier.")


def test_fabricated_action_does_not_flag_an_ordinary_reply():
    assert not check_fabricated_action("Your EMI is ₹12,500, due on the 18th.")
    assert not check_fabricated_action("I can escalate this to a human who can send it to you.")


def test_unapplied_restructuring_catches_the_real_phrasing_found_live():
    # The exact phrasing seen live: calculate_hypothetical's real numbers,
    # described as already applied.
    assert check_unapplied_restructuring_claim(
        "Great! With a 3-month extension the loan now has 17 months left, "
        "and the EMI will be reduced to about ₹10,294.12 each month."
    )


def test_unapplied_restructuring_catches_a_few_real_variants():
    assert check_unapplied_restructuring_claim("Your tenure now has 17 months remaining.")
    assert check_unapplied_restructuring_claim("Your loan has been extended by 3 months.")
    assert check_unapplied_restructuring_claim("Your EMI has been restructured to a lower amount.")
    assert check_unapplied_restructuring_claim("Your loan was successfully extended.")


def test_unapplied_restructuring_does_not_flag_a_correctly_hedged_reply():
    assert not check_unapplied_restructuring_claim(
        "If this is approved, your EMI would become ₹10,294.12 over 17 months -- "
        "I've sent this for approval and you'll hear back once it's reviewed."
    )
    assert not check_unapplied_restructuring_claim("Your EMI is ₹12,500, due on the 18th.")


# --- currency markers other than "₹" ---------------------------------------
#
# The check used to recognize only "₹". A reply that wrote the same invented
# figure as "Rs. 5,00,000" or "500000 rupees" or in Hindi therefore bypassed
# it completely. These pin down every spelling people and models actually use.

_SETTLEMENT_CONVERSATION = [
    {"role": "user", "content": "how much to settle"},
    {"role": "tool", "content": '{"settlement_amount": 418000.0, "emi_amount": 28000.0}'},
]


def test_extract_rupee_amounts_understands_rs_inr_rupees_and_hindi_markers():
    cases = {
        "Pay Rs. 50,000 today": {50000.0},
        "Pay Rs 50000 today": {50000.0},
        "Pay rs.1,25,000 today": {125000.0},
        "Pay INR 1,25,000 today": {125000.0},
        "Pay 50,000 rupees today": {50000.0},
        "Pay 5 rupees today": {5.0},
        "Pay 500 Rs. today": {500.0},
        "आपको 50,000 रुपये देने हैं": {50000.0},
        "आपको ५०,००० रुपए देने हैं": {50000.0},
        "रु. 500 बकाया है": {500.0},
        "Pay ₹1,40,000 and ₹500.": {140000.0, 500.0},
    }
    for text, expected in cases.items():
        assert extract_rupee_amounts(text) == expected, text


def test_extract_rupee_amounts_ignores_words_that_merely_end_in_rs_or_contain_ru():
    # "rs" at the end of "hours"/"doctors", and "रु" at the start of "रुक",
    # are not currency markers; treating them as such would invent amounts.
    assert extract_rupee_amounts("wait 5 hours or ask doctors 3 times") == set()
    assert extract_rupee_amounts("५ मिनट रुक कर देखें") == set()


def test_fabricated_amount_written_with_rs_is_caught():
    failure = check_grounding("You'd need to pay Rs. 5,00,000 to settle in full.", _SETTLEMENT_CONVERSATION)

    assert failure
    assert 500000.0 in failure.ungrounded_amounts


def test_fabricated_amount_written_as_number_then_rupees_is_caught():
    failure = check_grounding("To settle you need to pay 500000 rupees.", _SETTLEMENT_CONVERSATION)

    assert failure
    assert 500000.0 in failure.ungrounded_amounts


def test_fabricated_amount_written_in_hindi_is_caught():
    failure = check_grounding("पूरा निपटान करने के लिए आपको 5,00,000 रुपये देने होंगे।", _SETTLEMENT_CONVERSATION)

    assert failure
    assert 500000.0 in failure.ungrounded_amounts


def test_grounded_amount_written_with_rs_or_rupees_still_passes():
    # The widened check must not start blocking correct replies.
    for reply in (
        "Your EMI is Rs. 28,000.",
        "Your EMI is INR 28,000.",
        "Your EMI is 28000 rupees.",
        "आपकी EMI 28,000 रुपये है।",
        "Settling costs Rs 4,18,000 in full.",
    ):
        assert not check_grounding(reply, _SETTLEMENT_CONVERSATION), reply


def test_borrowers_own_rs_amount_grounds_the_agents_restatement():
    conversation = [{"role": "user", "content": "can i pay Rs 15000 instead"}]

    assert not check_grounding("Will you be able to pay ₹15,000 by the 20th?", conversation)


def test_no_break_spaces_inside_an_amount_do_not_split_it():
    # The model emits U+202F / U+00A0 between digit groups. Read as just "5",
    # a correct Hindi reply was blocked as an invented amount of Rs 5.
    for sep in (" ", " ", " ", " "):
        assert extract_rupee_amounts(f"₹5{sep}56,738.07") == {556738.07}, repr(sep)


def test_a_correct_hindi_reply_with_no_break_spaces_is_not_blocked():
    conversation = [
        {"role": "user", "content": "poora loan settle karna hai"},
        {"role": "tool", "content": '{"settlement_amount": 556738.07, "outstanding": 586040.07}'},
    ]
    reply = "आपको ₹5 56,738.07 का भुगतान करना होगा, जो ₹5 86,040.07 पर 5 % की छूट के बाद है।"

    assert not check_grounding(reply, conversation)


def test_lakh_and_crore_words_after_an_amount_are_read_as_the_real_figure():
    cases = {
        "₹5.56 लाख": {556000.0},
        "₹5 lakh": {500000.0},
        "Rs. 4.18 lakh": {418000.0},
        "5 लाख रुपये": {500000.0},
        "1.5 crore rupees": {15000000.0},
        "₹2 हज़ार": {2000.0},
    }
    for text, expected in cases.items():
        assert extract_rupee_amounts(text) == expected, text


def test_lakh_without_a_currency_marker_is_not_money():
    assert extract_rupee_amounts("5 लाख लोग और 3 lakh visitors") == set()


def test_a_made_up_amount_written_in_lakh_is_caught_and_a_correct_one_is_not():
    # "5 lakh" (= 500,000) used to be read as nothing at all, so an invented
    # figure phrased this way slipped through; "₹4.18 lakh" for a real 418,000
    # used to be read as 4.18 and blocked a correct reply.
    assert check_grounding("To settle you need to pay 5 lakh rupees.", _SETTLEMENT_CONVERSATION)
    assert not check_grounding("Settling costs ₹4.18 lakh in full.", _SETTLEMENT_CONVERSATION)


def test_descriptive_numbers_are_still_never_treated_as_money():
    for reply in ("It is 3 days past due.", "That is a 5% discount.", "You have 17 months left."):
        assert not check_grounding(reply, _SETTLEMENT_CONVERSATION), reply

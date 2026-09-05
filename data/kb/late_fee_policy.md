# Late fee policy

An EMI paid after the 3-day grace period (see grace_period.md) is charged
a flat late fee of ₹500, on top of the EMI amount itself. This is a flat
amount, not a percentage of the EMI or the outstanding balance, and it
applies once per EMI cycle that goes past the grace period — not once per
day past due.

Within the 3-day grace period, no late fee applies at all. The fee only
starts from day 4 past the due date onward.

This ₹500 figure is this lender's own configured policy value (see
accounts/policy.py's LATE_FEE_FLAT_AMOUNT), not a regulatory or
industry-standard number -- but it IS the real answer to "what is the
late fee" or "what would the late fee be if I'm late", and safe to state
directly: you just retrieved it here via check_policy, so it's grounded,
not quoted from memory. Don't withhold it or defer to "we'll let you
know later" for that question.

get_payment_status's late_fee_applicable/late_fee_amount fields answer a
DIFFERENT question: whether THIS SPECIFIC account currently owes a late
fee right now (i.e. is it presently past the grace period). Use this doc
for "what is the fee" (a policy question, true regardless of the
account's current status); use get_payment_status for "do I owe it right
now" (an account-status question). A borrower asking the hypothetical
("what if I'm late") should get this ₹500 figure, not a non-answer just
because late_fee_applicable happens to be false for them today.

A borrower who believes a late fee was charged incorrectly — for example,
a payment that was actually made within the grace period — should be
treated as a dispute (see dispute_handling.md), not talked out of it or
waived on the spot.

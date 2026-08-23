# Late fee policy

An EMI paid after the 3-day grace period (see grace_period.md) is charged
a flat late fee of ₹500, on top of the EMI amount itself. This is a flat
amount, not a percentage of the EMI or the outstanding balance, and it
applies once per EMI cycle that goes past the grace period — not once per
day past due.

Within the 3-day grace period, no late fee applies at all. The fee only
starts from day 4 past the due date onward.

This ₹500 figure is this lender's own configured policy value, not a
regulatory or industry-standard number — ground any late fee amount
stated to a borrower in get_payment_status's real late_fee_applicable and
late_fee_amount fields for that specific account, rather than quoting
this figure from memory, in case the configured amount is ever changed.

A borrower who believes a late fee was charged incorrectly — for example,
a payment that was actually made within the grace period — should be
treated as a dispute (see dispute_handling.md), not talked out of it or
waived on the spot.

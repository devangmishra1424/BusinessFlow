# Late fee policy

An EMI still unpaid after the 3-day grace period (see grace_period.md) is
charged a flat late fee of ₹500 — a flat amount, not a percentage, charged
once per EMI that goes past the grace period, not once per day. It is never
compounded and never charged interest. From day 4 past the due date it is a
real charge on the account, shown with the EMI it is for until it is paid or
waived. A payment made within the grace period never carries the fee.

A payment goes to the EMI first, then to any late fee still owed; anything
left is an extra payment. Paying the EMI keeps the loan up to date; the fee
stays owed until paid.

This ₹500 figure is current policy (accounts/policy.py LATE_FEE_FLAT_AMOUNT),
safe to state directly. get_payment_status's charges and late_fee_amount say
which fees THIS account owes right now.

A borrower who believes a fee was charged wrongly is raising a dispute
(see dispute_handling.md), not to be talked out of it. Only our team can waive a
fee, with a reason the borrower is told.

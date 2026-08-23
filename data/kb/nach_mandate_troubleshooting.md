# NACH mandate troubleshooting

NACH (National Automated Clearing House) is the bank-debit mandate most
borrowers use to pay their EMI automatically — the bank debits the EMI
amount from the borrower's account on the due date, without the borrower
doing anything manually each cycle.

A NACH debit can fail for a few common reasons:

- Insufficient balance in the borrower's bank account on the debit date.
- The mandate itself has expired or lapsed — banks require it to be
  renewed periodically, and a lapsed mandate blocks every debit attempt
  until it's renewed.
- A bank-side technical failure — the debit request was rejected or
  timed out on the bank's end, unrelated to the borrower's balance or
  mandate status.

get_payment_status's nach_mandate_active field says only whether the
mandate is currently active or not — it does not say WHY a specific past
debit attempt bounced. Never guess a specific reason for a specific
failed payment; state what is actually known (active or not) and, if the
borrower wants the specific cause, say that isn't something this agent
can determine and offer to escalate.

Re-registering or reactivating a NACH mandate is not something this
agent can do directly. It requires a new mandate form, signed and
authorized with the borrower's bank — this agent has no tool that
performs that action. When a mandate is inactive, the agent's role is to
explain this plainly, not to imply it can fix the mandate itself. In the
meantime, a promise-to-pay or a payment link are the interim options
available while the borrower (or a human on this side) sorts out the
mandate with the bank.

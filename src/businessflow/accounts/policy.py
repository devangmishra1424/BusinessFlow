"""Collections policy constants.

Single source of truth: tools enforce these bounds programmatically, and the
KB docs in data/kb/ state the same numbers in natural language for RAG
retrieval. Keep both in sync by hand — the KB text is not generated from here.
"""

GRACE_PERIOD_DAYS = 3

# How many days before an EMI is due the proactive outbound heads-up
# reminder goes out -- outbound/decide.py's own deadline-proximity check.
HEADS_UP_DAYS_BEFORE_DUE = 2

# A promise-to-pay is "kept" if payment lands within this many days of the
# promised date, not only on the exact date.
PROMISE_TOLERANCE_DAYS = 2

# Escalate to a human on the account's Nth broken promise, not the 1st —
# a single slip is normal; a pattern isn't.
BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION = 2

MAX_RESTRUCTURING_EXTENSION_MONTHS = 3

# A restructured/partial payment can't be proposed below this fraction of the
# standard EMI — propose_partial_payment must reject anything lower.
MIN_PARTIAL_PAYMENT_PCT = 0.70

RESTRUCTURING_TYPES = (
    "extend_tenure",       # lower monthly EMI, longer timeline; preserves delinquency history
    "one_time_settlement", # single reduced lump-sum payment closes the loan early
)

# Flat discount off remaining principal offered for an early one-time
# settlement -- a simple, explainable rule rather than a real NPV calc.
SETTLEMENT_DISCOUNT_PCT = 0.05

# An open dispute or a mandatory-escalation-triggering promise history means
# no automated restructuring offer — human only, context already assembled.
DISPUTE_BLOCKS_AUTOMATED_RESTRUCTURING = True

# Flat rupee late fee charged once an EMI is past GRACE_PERIOD_DAYS --
# grace_period.md says no late fee applies WITHIN the grace period, which
# implies one exists after it, but nothing in this codebase actually
# defined an amount before this constant. 500 is an ASSUMED, illustrative
# default in this project's existing style of round, simple constants
# (see SETTLEMENT_DISCOUNT_PCT above) -- it is not a researched real NBFC
# figure, and a real business adopting this should confirm/adjust it.
LATE_FEE_FLAT_AMOUNT = 500

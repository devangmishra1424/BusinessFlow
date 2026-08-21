"""Seeds the 4 demo accounts (and their payment history / promises) into
the real Postgres tables. Idempotent -- clears existing demo rows first,
so it's safe to re-run after editing the seed data below.

Run: python -m scripts.seed_accounts
"""

from businessflow.accounts.db import get_connection
from businessflow.accounts.store import DEMO_TODAY

_ACCOUNT_IDS = ["BF-1001", "BF-1002", "BF-1003", "BF-1004"]

_ACCOUNTS = [
    {
        "account_id": "BF-1001",
        "borrower_name": "Priya Sharma",
        "business_name": "Cotton Threads Boutique",
        "phone_number": "+919812345001",
        "language_preference": "hinglish",
        "loan_type": "Working Capital Loan",
        "principal_amount": 250_000,
        "emi_amount": 12_500,
        "tenure_months": 24,
        "months_remaining": 14,
        "emi_due_date": "2026-08-18",  # 3 days past due as of DEMO_TODAY
        "nach_mandate_active": True,
        "dispute_open": False,
        "risk_tier": "low",
    },
    {
        "account_id": "BF-1002",
        "borrower_name": "Arjun Mehta",
        "business_name": "Mehta Hardware & Electricals",
        "phone_number": "+919812345002",
        "language_preference": "hi",
        "loan_type": "Equipment Loan",
        "principal_amount": 500_000,
        "emi_amount": 22_000,
        "tenure_months": 36,
        "months_remaining": 20,
        "emi_due_date": "2026-08-10",  # 11 days past due
        "nach_mandate_active": False,
        "dispute_open": False,
        "risk_tier": "medium",
    },
    {
        "account_id": "BF-1003",
        "borrower_name": "Fatima Khan",
        "business_name": "Khan Textiles Wholesale",
        "phone_number": "+919812345003",
        "language_preference": "hinglish",
        "loan_type": "Working Capital Loan",
        "principal_amount": 800_000,
        "emi_amount": 35_000,
        "tenure_months": 30,
        "months_remaining": 11,
        "emi_due_date": "2026-08-01",  # 20 days past due
        "nach_mandate_active": False,
        "dispute_open": True,  # disputes an incorrectly applied late fee
        "risk_tier": "high",
    },
    {
        "account_id": "BF-1004",
        "borrower_name": "Ravi Iyer",
        "business_name": "Iyer Auto Spares",
        "phone_number": "+919812345004",
        "language_preference": "en",
        "loan_type": "Business Expansion Loan",
        "principal_amount": 600_000,
        "emi_amount": 28_000,
        "tenure_months": 30,
        "months_remaining": 22,
        "emi_due_date": "2026-08-09",  # 12 days past due
        "nach_mandate_active": True,
        "dispute_open": False,
        "risk_tier": "medium",
    },
]

_PAYMENT_HISTORY = [
    ("BF-1001", "2026-05-18", 12_500, True),
    ("BF-1001", "2026-06-18", 12_500, True),
    ("BF-1001", "2026-07-18", 12_500, True),
    ("BF-1002", "2026-05-10", 22_000, True),
    ("BF-1002", "2026-06-13", 22_000, False),
    ("BF-1002", "2026-07-19", 22_000, False),
    ("BF-1003", "2026-06-01", 35_000, True),
    # July and August EMIs both missed entirely -- no records for them.
    ("BF-1004", "2026-06-09", 28_000, True),
    ("BF-1004", "2026-07-09", 28_000, True),
]

_PROMISES = [
    ("BF-1002", "2026-06-15", "2026-06-20", 22_000, False),
    ("BF-1002", "2026-07-10", "2026-07-14", 22_000, True),
    ("BF-1003", "2026-06-10", "2026-06-15", 35_000, False),
    ("BF-1003", "2026-07-05", "2026-07-10", 35_000, False),
]


def main():
    conn = get_connection()

    conn.execute("delete from events where account_id = any(%s)", (_ACCOUNT_IDS,))
    conn.execute("delete from escalations where account_id = any(%s)", (_ACCOUNT_IDS,))
    conn.execute("delete from disputes where account_id = any(%s)", (_ACCOUNT_IDS,))
    conn.execute("delete from promises where account_id = any(%s)", (_ACCOUNT_IDS,))
    conn.execute("delete from payment_history where account_id = any(%s)", (_ACCOUNT_IDS,))
    conn.execute("delete from accounts where account_id = any(%s)", (_ACCOUNT_IDS,))

    for a in _ACCOUNTS:
        conn.execute(
            """
            insert into accounts (
                account_id, borrower_name, business_name, phone_number, language_preference,
                loan_type, principal_amount, emi_amount, tenure_months, months_remaining,
                emi_due_date, nach_mandate_active, dispute_open, risk_tier
            ) values (
                %(account_id)s, %(borrower_name)s, %(business_name)s, %(phone_number)s, %(language_preference)s,
                %(loan_type)s, %(principal_amount)s, %(emi_amount)s, %(tenure_months)s, %(months_remaining)s,
                %(emi_due_date)s, %(nach_mandate_active)s, %(dispute_open)s, %(risk_tier)s
            )
            """,
            a,
        )

    for account_id, payment_date, amount, on_time in _PAYMENT_HISTORY:
        conn.execute(
            "insert into payment_history (account_id, payment_date, amount, on_time) values (%s, %s, %s, %s)",
            (account_id, payment_date, amount, on_time),
        )

    for account_id, made_on, promised_date, promised_amount, kept in _PROMISES:
        conn.execute(
            "insert into promises (account_id, made_on, promised_date, promised_amount, kept) values (%s, %s, %s, %s, %s)",
            (account_id, made_on, promised_date, promised_amount, kept),
        )

    print(f"seeded {len(_ACCOUNTS)} accounts, {len(_PAYMENT_HISTORY)} payment records, {len(_PROMISES)} promises")
    print(f"DEMO_TODAY anchor: {DEMO_TODAY}")


if __name__ == "__main__":
    main()

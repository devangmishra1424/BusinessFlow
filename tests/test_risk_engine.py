import unittest
from datetime import date
from businessflow.accounts.models import Account, PromiseToPay
from businessflow.outbound.risk_engine import RiskTier, ToneStrategy, calculate_account_risk


class TestRiskEngine(unittest.TestCase):
    def test_calculate_account_risk_low(self):
        acc = Account(
            account_id="ACC-TEST-1",
            borrower_name="Ramesh Kumar",
            business_name="Kumar Traders",
            phone_number="+919876543210",
            language_preference="en",
            loan_type="working_capital",
            principal_amount=500000.0,
            emi_amount=25000.0,
            tenure_months=24,
            months_remaining=12,
            emi_due_date=date(2026, 9, 20),
            nach_mandate_active=True,
            dispute_open=False,
            risk_tier="low",
        )
        profile = calculate_account_risk(acc, as_of=date(2026, 9, 16))
        self.assertEqual(profile.tier, RiskTier.LOW)
        self.assertEqual(profile.recommended_tone, ToneStrategy.EMPATHETIC)

    def test_calculate_account_risk_high_and_broken_promises(self):
        acc = Account(
            account_id="ACC-TEST-2",
            borrower_name="Suresh Patel",
            business_name="Patel Logistics",
            phone_number="+919876543211",
            language_preference="hi",
            loan_type="working_capital",
            principal_amount=800000.0,
            emi_amount=40000.0,
            tenure_months=24,
            months_remaining=6,
            emi_due_date=date(2026, 9, 1),
            nach_mandate_active=False,
            dispute_open=True,
            risk_tier="high",
            promises=[
                PromiseToPay(made_on=date(2026, 9, 2), promised_date=date(2026, 9, 5), promised_amount=40000.0, kept=False),
                PromiseToPay(made_on=date(2026, 9, 6), promised_date=date(2026, 9, 10), promised_amount=40000.0, kept=False),
            ]
        )
        profile = calculate_account_risk(acc, as_of=date(2026, 9, 16))
        self.assertEqual(profile.tier, RiskTier.CRITICAL)
        self.assertEqual(profile.recommended_tone, ToneStrategy.FINAL_NOTICE)
        self.assertTrue(any("broken promise" in f for f in profile.factors))


if __name__ == "__main__":
    unittest.main()


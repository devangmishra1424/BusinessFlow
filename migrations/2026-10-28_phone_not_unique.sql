-- One borrower can hold several loans, each its own account on the same phone number (a second loan, an equipment
-- loan beside a term loan; one Telegram chat links them all). Databases created before multi-loan carry a uniqueness
-- rule on the phone number that refuses the second loan; schema.sql never had it. Found 2026-10-23 building the demo.
-- Idempotent. To undo (fails if two accounts now share a phone): alter table accounts add constraint accounts_phone_number_key unique (phone_number);
alter table accounts drop constraint if exists accounts_phone_number_key;

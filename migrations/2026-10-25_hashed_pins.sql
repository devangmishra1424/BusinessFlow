-- Borrower access keys (6-digit PINs) are stored hashed (accounts/pins.py): a salted scrypt hash in access_key_hash,
-- and the plain-text access_key nulled. The hashing itself needs scrypt, so it is not done in SQL:
--   1. this migration adds the column (additive; the old code keeps working on access_key meanwhile);
--   2. after the new code is live, `python -m scripts.hash_access_keys` hashes every remaining plain key and nulls it.
-- Until step 2, a borrower signing in with a plain key is checked once the old way and upgraded on the spot
-- (accounts/store.py verify_account_key). Re-runnable.

alter table accounts add column if not exists access_key_hash text;

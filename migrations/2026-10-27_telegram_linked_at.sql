-- One Telegram chat can hold several loans (a borrower with two loans uses one Telegram). telegram_linked_at marks when
-- the account was last verified from, or switched to in, its chat; the most recent is the chat's active loan, resumed
-- after a restart. Additive; existing links get their last update time. Re-runnable.

alter table accounts add column if not exists telegram_linked_at timestamptz;
update accounts set telegram_linked_at = updated_at where telegram_chat_id is not null and telegram_linked_at is null;

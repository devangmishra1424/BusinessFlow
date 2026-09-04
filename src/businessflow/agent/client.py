"""The Groq client and system prompt shared by both the bare, tool-less
reply() (used to prove the voice I/O shell works end to end) and the real
tool-calling agent loop in agent/loop.py.
"""

import os
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-20b"

# Advances through as many fallback keys as are actually configured
# (ALTERNATE_GROQ_KEY, then ALTERNATE_GROQ_KEY2, ALTERNATE_GROQ_KEY3, ...),
# and reverts back to the primary on its own after _FALLBACK_COOLDOWN_SECONDS.
# Originally this was a one-way, permanent-for-the-process switch to a
# single hardcoded fallback; for a long-running server that's a real bug,
# not just an inefficiency -- it would stay on the fallback forever even
# after the primary's quota reset, and it had nowhere to go once that one
# fallback also got rate-limited.
#
# _FALLBACK_COOLDOWN_SECONDS was originally 24h, on the assumption that
# Groq's only real limit here is the daily (tokens-per-day) one. Found
# live this was the wrong model entirely: every real rate-limit hit this
# project actually saw was the much smaller, much faster per-MINUTE cap
# (8,000 tokens/min per key, confirmed via real completion calls -- and
# confirmed the 5 configured keys are genuinely independent buckets, not
# one shared org-level limit, by burning real tokens on one and watching
# the others' remaining-token headers stay untouched). That cap clears in
# well under two minutes every time it was checked live. With a 24h
# cooldown, once ANY one bad turn cycled through every configured
# fallback, the process stayed pinned to the LAST one for the rest of the
# day -- concentrating all subsequent traffic onto a single key (with the
# other 4 sitting completely idle) instead of giving the earlier keys the
# ~60-90 seconds they actually needed to recover. 150s comfortably clears
# that real recovery window without waiting anywhere near a full day.
# Deliberately still short is fine even in an actual daily-exhaustion
# case: reverting to a still-exhausted primary just wastes one fast-
# failing retry before cycling through the same fallbacks again, not a
# new failure mode.
_FALLBACK_KEY_ENV_VAR = "ALTERNATE_GROQ_KEY"
_MAX_FALLBACK_KEY_SUFFIX = 20  # ALTERNATE_GROQ_KEY2..20 -- generous headroom; adding one more needs no code change

_current_key_index = 0  # 0 = primary (GROQ_API_KEY); N>0 = the Nth configured fallback
_switched_at: float | None = None
_FALLBACK_COOLDOWN_SECONDS = 150


def _fallback_env_var_names() -> list[str]:
    """ALTERNATE_GROQ_KEY, then ALTERNATE_GROQ_KEY2, ALTERNATE_GROQ_KEY3,
    ... -- as many as are actually set in the environment right now,
    discovered dynamically rather than hardcoded to a fixed count.
    Checked by suffix rather than stopped at the first gap, so removing
    one out of order doesn't silently hide the ones configured after it."""
    names = []
    if os.environ.get(_FALLBACK_KEY_ENV_VAR):
        names.append(_FALLBACK_KEY_ENV_VAR)
    for n in range(2, _MAX_FALLBACK_KEY_SUFFIX + 1):
        var = f"{_FALLBACK_KEY_ENV_VAR}{n}"
        if os.environ.get(var):
            names.append(var)
    return names


def _active_env_var() -> str:
    if _current_key_index == 0:
        return "GROQ_API_KEY"
    return _fallback_env_var_names()[_current_key_index - 1]

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a collections agent for an Indian SMB lender, speaking to a "
    "borrower. {account_context}{language_instruction} Be direct, warm, "
    "and brief -- this is a spoken conversation, not a written one. "
    "{commitment_discipline} {dispute_handling} {read_only_tools} {ground_account_facts} "
    "{untracked_account_data} {no_fabricated_links} {no_fabricated_actions} "
    "{ground_policy_claims} {check_dispute_block_first} {out_of_domain_legal} "
    "{ground_nach_failures} {fraud_or_identity_claim} {due_date_change_requests} "
    "{cibil_credit_score} {discount_firmness} {restructuring_approval_flow} "
    "{payment_history_requests} {closure_certificate_requests} {grievance_redressal} "
    "{no_mental_math}"
)

# Real borrowers hedge ("maybe 15k", "not sure which is better") instead of
# committing outright -- treating a hedge as a firm decision and acting on
# it (logging a promise, sending a payment link) is worse than asking one
# confirming question. Found via eval/realistic_conversation_benchmark.py:
# without this, a "maybe 15k, not sure" got logged as an accepted promise
# with a fabricated due date the borrower never gave.
_COMMITMENT_DISCIPLINE = (
    "Only call log_promise_to_pay, generate_payment_link, or "
    "propose_partial_payment when the borrower has given a specific, firm "
    "amount (and, for a promise, a specific date) -- if they're hedging "
    "('maybe', 'not sure', thinking out loud, or offering several options "
    "at once), ask one short confirming question instead of acting on it. "
    "Never invent a date or amount the borrower did not actually state."
)

# flag_dispute exists specifically to freeze automated collection action
# while a HUMAN verifies -- asking the borrower for "proof" the agent has
# no actual way to check just delays that freeze, it doesn't verify
# anything. Found via eval/realistic_conversation_benchmark.py: the same
# dispute claim, worded with typos, got met with "share your transaction
# reference" instead of an immediate flag -- noisy phrasing shouldn't
# change whether a dispute claim gets flagged.
_DISPUTE_HANDLING = (
    "If a borrower says they already paid, or disputes a charge, call "
    "flag_dispute right away, even if you can't verify it yourself -- a "
    "human will check the details afterward. Don't withhold the flag "
    "while asking for proof you have no way to actually verify."
)

# get_payment_status and calculate_hypothetical look up everything they
# need (balance, EMI, months remaining) from account_id alone -- neither
# needs the borrower to state a figure first. Found via repeated runs of
# eval/tool_calling_benchmark.py: on some runs, a plain "what would a
# settlement cost me" got met with "tell me your remaining balance"
# instead of just calling the tool, which already knows it.
_READ_ONLY_TOOLS = (
    "get_payment_status and calculate_hypothetical need only account_id "
    "-- call them directly to answer a status or 'what if' question, "
    "never ask the borrower to state a balance or figure the tool "
    "already looks up itself."
)

# Found live via the Telegram channel (not an eval scenario): a single,
# natural compound question ("how much is my loan, how many months of
# emi is left, what's my emi and the interest") got answered without
# calling any tool at all -- the EMI figure happened to be right by
# coincidence, but "0 months left" was flatly wrong (the real
# months_remaining was 14, confirmed by the very next turn's
# calculate_hypothetical result implying that same base). grounding.py's
# check_grounding didn't catch it: months-remaining isn't a rupee amount
# it checks, and the one number it does check (the EMI) happened to be
# real. This has to be prompted explicitly, the same reason
# _GROUND_POLICY_CLAIMS exists for policy claims the mechanical check
# can't cover either.
_GROUND_ACCOUNT_FACTS = (
    "Never state a specific fact about THIS account -- balance, EMI "
    "amount, months remaining, original loan amount, due date, dispute "
    "status -- from memory or a guess. Always call get_payment_status "
    "first (or reuse its real result from earlier this conversation) "
    "and answer only from what it actually returned. If one message "
    "asks several things about the account at once, that's still just "
    "one get_payment_status call -- read every field the question "
    "touched from that one result, rather than answering the parts "
    "you're confident about and guessing the rest."
)

# get_payment_status now returns a real interest_rate_pct field, but it
# is null/missing for most accounts -- populated only once that specific
# borrower's signed loan agreement has been uploaded and successfully
# parsed (rag/extraction.py's extract_loan_terms). Found alongside the
# bug this replaces: rather than saying plainly that a figure isn't on
# file, the model folded a vague non-answer about interest into the EMI
# figure instead. The behavior this enforces (never fabricate) is
# unchanged from before this field existed -- only the reason for a
# "no" answer changes, from "never tracked at all" to "not extracted yet
# for this specific account."
_UNTRACKED_ACCOUNT_DATA = (
    "If a borrower asks for the interest rate/APR, check "
    "get_payment_status's real interest_rate_pct field first (or reuse "
    "its result from earlier this conversation) -- most accounts don't "
    "have one extracted yet, so it's often null. If it's null or "
    "missing for this specific account, say plainly that it isn't on "
    "file for this account and offer to check with someone who does, "
    "rather than implying a number or folding it vaguely into another "
    "figure. If interest_rate_pct DOES have a real value, state that "
    "real number -- don't call it untracked when the tool just gave you "
    "one."
)

# Found live via eval/realistic_conversation_benchmark.py's
# many_operations_same_account_en scenario: asked to lower/stretch a
# payment on an account it already knew (this same session) had an open
# dispute, the model asked "how many extra months would you like?"
# instead of checking eligibility first -- a dead-end question, since the
# dispute was always going to block it regardless of the answer. The
# same model correctly went straight to calling propose_partial_payment
# (and got the same block) for a different restructuring ask two turns
# later in that same conversation -- so this is a real, specific
# inconsistency to close, not a hard model limitation.
_CHECK_DISPUTE_BLOCK_FIRST = (
    "If you already know (from get_payment_status or anything said "
    "earlier this session) that an account has an open dispute or "
    "repeated broken promises, don't ask for restructuring specifics "
    "(how many months, how much lower) before checking -- go straight "
    "to calculate_hypothetical or propose_partial_payment (or state the "
    "block directly and offer escalation) instead of gathering details "
    "for an offer you already know won't be approved. But once the "
    "borrower has already given a specific number or month count -- "
    "there's nothing left to gather -- always call calculate_hypothetical "
    "or propose_partial_payment with that real figure before stating any "
    "outcome. 'State the block directly' is for when nothing concrete has "
    "been proposed yet, never a substitute for checking a real number the "
    "borrower already gave you -- even late in a long conversation, after "
    "you've already discussed this account's dispute or broken promises "
    "several times, a NEW concrete number still needs its own real tool "
    "call. Don't rely on what you already know about the account instead "
    "of checking the specific new figure."
)

# Found live, twice in the same real conversation: asked about an
# early-settlement discount on a disputed account, the model narrated
# "you'd normally get a 5% discount, that's about ₹X" using its own
# arithmetic BEFORE explaining the dispute blocks it -- calculate_hypothetical
# returns no settlement_amount at all when blocked (see
# _blocked_from_automated_restructuring), so there is no real figure to
# state in that case, period. The guardrail correctly rejected the
# invented number both times, leaving the borrower with no answer at all
# instead of a clean "here's why I can't give you a number yet."
_NO_MENTAL_MATH = (
    "Never do arithmetic yourself in a reply -- not a percentage, not a "
    "sum, not 'roughly X'. The grounding guardrail only trusts a number "
    "that came from a real tool result; your own mental math gets blocked "
    "even when the inputs were real and the arithmetic was correct, which "
    "means the borrower gets NO answer instead of the real one. If a tool "
    "like calculate_hypothetical already returns the figure you need, use "
    "that exact value. If you need a number derived from other real "
    "figures already established this conversation (a percentage of a "
    "real balance, a difference, a sum) and no other tool returns it "
    "directly, call compute() and state its exact result. If a policy "
    "tool comes back blocked (eligible: False) with no figure attached, "
    "don't estimate what the number WOULD have been -- there is nothing "
    "to state; explain the block and offer escalation instead."
)

# Found live via eval/realistic_conversation_benchmark.py: after backing
# out of a settlement offer and asking for "the regular link", the model
# had already fetched the real EMI via get_payment_status but then wrote
# out its OWN made-up URL instead of calling generate_payment_link --
# presented as a real link, even though it isn't one that actually works.
_NO_FABRICATED_LINKS = (
    "Never write out a payment link yourself, not even as an example -- "
    "a URL you compose is not real and will not work. Always call "
    "generate_payment_link to get the actual link before mentioning one."
)

# Found live via the Telegram channel (not an eval scenario, and the
# single most serious thing found there): asked to send the loan
# agreement, the model replied "I've arranged to send a copy to
# [email]... you should receive it shortly" -- then, two turns later,
# unprompted, restated it as settled fact ("I've already sent a copy").
# There is no email/SMS/document-delivery tool anywhere in this codebase
# at all (confirmed by a full repo audit, same as outbound/send.py's own
# comment) and no email field on the account -- this wasn't a wrong
# number, it was a confident claim of a real action, taken on behalf of
# a real lender, that never happened and never could have. grounding.py's
# check_grounding only checks URLs and rupee amounts, so a claimed
# ACTION slips through it completely, the same reason _NO_FABRICATED_LINKS
# above has to exist in the prompt instead of relying on the mechanical
# check alone.
_NO_FABRICATED_ACTIONS = (
    "Never claim to have sent, emailed, mailed, forwarded, or delivered "
    "anything to the borrower -- there is no email, SMS, or document-"
    "delivery capability in this system at all, and no email address is "
    "even on file for any account. If a borrower asks for a document to "
    "be sent or emailed, say plainly that this chat can't send documents "
    "directly -- offer to quote the relevant clause via check_policy, or "
    "escalate to a human who could actually arrange delivery. This "
    "applies for the rest of the conversation too: if you already said "
    "this once, don't later restate 'I've already sent it' as if it "
    "became true because you said it before."
)

# Found live via eval/realistic_conversation_benchmark.py's
# many_operations_same_account_en scenario: asked how long a dispute
# "usually" takes to resolve, the model stated "5-7 business days" --
# a specific, confident-sounding number that appears nowhere in the real
# policy docs and was never checked (check_policy wasn't called that
# turn). The Guardrail (guardrail/grounding.py) only checks URLs and
# rupee amounts, so a plain-text claim like this slips through it
# entirely -- this has to be caught here, in the prompt, instead.
_GROUND_POLICY_CLAIMS = (
    "Never state a specific policy detail -- a fee, a timeline, a "
    "resolution window, how long something 'usually' takes -- from "
    "memory. Call check_policy first and ground the answer in what it "
    "actually returns; if it doesn't have an answer, say you're not "
    "sure and will find out, rather than giving a plausible-sounding "
    "number."
)

# Found via eval/red_team.py's out_of_domain_legal scenario: asked
# whether the loan agreement was "legally enforceable" and about "legal
# rights," the model didn't give a wrong legal opinion (good), but also
# didn't recognize the question was out of scope -- it said it would
# "check the policy documents and get back to you," a dead end, since
# check_policy's KB is internal collections policy, not legal counsel.
_OUT_OF_DOMAIN_LEGAL = (
    "If a borrower asks a genuinely legal question -- whether the loan "
    "agreement is enforceable, their legal rights, whether they should "
    "sue or be sued -- that is out of scope for check_policy (it only "
    "covers internal collections policy, not legal advice) and out of "
    "scope for you. Say plainly that you can't give legal advice and "
    "offer to escalate to a human, rather than saying you'll research "
    "it and follow up."
)

# faq_general.md frames this entire agent as existing BECAUSE a borrower's
# auto-debit failed or is at risk of failing, and accounts.nach_mandate_active
# is a real, already-seeded boolean column -- but get_payment_status never
# exposed it to a borrower-facing question, and no KB doc existed either way,
# so the model had nothing to check and nothing telling it not to guess a
# specific reason for a specific bounced debit, or that it can't itself fix a
# mandate. Found via a full audit of accounts/policy.py, tools/account_tools.py,
# and data/kb/ against schema.sql's real columns, not a benchmark scenario --
# the same reason _GROUND_ACCOUNT_FACTS and _UNTRACKED_ACCOUNT_DATA above
# exist for other fields the model could otherwise guess at instead of
# checking.
_GROUND_NACH_FAILURES = (
    "If a borrower asks why their auto-debit/NACH payment failed, or asks "
    "you to fix, re-register, or reactivate their mandate, check "
    "get_payment_status's real nach_mandate_active field first (or reuse "
    "its result from earlier this conversation) and ground the rest of "
    "the answer in check_policy's nach_mandate_troubleshooting content -- "
    "the tool can only tell you whether the mandate is CURRENTLY active, "
    "never why one specific debit attempt bounced, so don't guess a "
    "specific cause. Never claim you can re-register, reactivate, or "
    "otherwise fix a NACH mandate yourself -- that requires a new mandate "
    "form authorized with the borrower's bank, which is outside anything "
    "this agent can do; offer a promise-to-pay or a payment link as the "
    "interim option instead."
)

# A borrower saying "this loan isn't mine" / "someone took this loan out in
# my name" is a categorically different claim than an ordinary billing
# dispute ("this late fee is wrong", "I already paid this") -- it disputes
# the loan's basic legitimacy or the borrower's own identity, not one
# specific charge or amount. Both currently could be routed through the same
# flag_dispute/escalate_to_human tools with no distinction at all, so a
# human triaging a queue of free-text reasons had no signal that this one is
# more urgent than a routine "borrower says they already paid." Confirmed
# via a direct audit of flag_dispute/escalate_to_human (both take a plain
# reason string, no urgency field) and this prompt (nothing fraud-specific
# existed anywhere in it) -- an audit finding, not a benchmark one, same
# provenance as _GROUND_NACH_FAILURES above.
_FRAUD_OR_IDENTITY_CLAIM = (
    "If a borrower's claim sounds like they're disputing the loan's basic "
    "legitimacy or their own identity -- 'this loan isn't mine', 'someone "
    "took this loan out in my name', anything suggesting identity theft or "
    "fraud -- that is categorically more urgent than an ordinary billing "
    "dispute, and isn't just a routine flag_dispute case. Call "
    "escalate_to_human directly, with a reason string that starts with "
    "'SUSPECTED FRAUD/IDENTITY CLAIM' so whoever picks it up sees the "
    "urgency immediately. Don't try to reassure the borrower, investigate "
    "it yourself, or ask them for 'proof' -- you have no way to verify "
    "identity or loan origination, and this needs a human faster than a "
    "normal dispute escalation does."
)

# This agent has no mechanism to actually change an EMI due date -- it's
# tied to the borrower's NACH mandate/bank auto-debit setup (see
# _GROUND_NACH_FAILURES above), not a field this system can edit on
# request. This is the same class of problem _NO_FABRICATED_ACTIONS already
# guards against (claiming to do something with no real mechanism behind
# it) -- called out explicitly here, separately, because due-date-change
# requests are common enough on their own to need a direct instruction
# rather than relying on the model to generalize from the email-delivery
# case that motivated that fragment.
_DUE_DATE_CHANGE_REQUESTS = (
    "If a borrower asks to move or change their EMI due date, say plainly "
    "that this channel can't do that directly -- a due date is tied to the "
    "NACH mandate registered with their bank, and changing it needs a new "
    "mandate registration, not something this system can edit on request. "
    "Offer to escalate to a human who handles that instead. Never agree to "
    "'note', 'arrange', or 'put in' a due-date change -- there is no "
    "mechanism behind that promise and it will not actually happen."
)

# The system has no access to any borrower's actual credit report or CIBIL
# score at all -- get_payment_status exposes account fields (balance, EMI,
# NACH status, interest rate where extracted), never a credit score, and no
# credit-bureau integration exists anywhere in this codebase. A general,
# factual answer about late payments and credit bureaus is fine (it's
# common knowledge, not a claim about THIS account), but anything specific
# to this borrower's real score, or personalized advice on managing their
# credit, would be fabricated the same way a specific ungrounded policy
# timeline would be (_GROUND_POLICY_CLAIMS above) -- there is simply
# nothing on file to ground it in.
_CIBIL_CREDIT_SCORE = (
    "If asked whether a missed payment affects their credit score, you may "
    "give a general, factual answer -- late payments are commonly reported "
    "to credit bureaus and can affect credit scores -- but never state "
    "anything specific about THIS borrower's actual score or credit "
    "report; you have no access to either. Don't give personalized "
    "financial advice about how to manage their credit."
)

# SETTLEMENT_DISCOUNT_PCT (accounts/policy.py) is a fixed 5% policy constant
# that calculate_hypothetical always applies to a one-time settlement --
# there is no higher tier and no path to one through this channel.
# eval/red_team.py's social-engineering scenario already checks the model
# doesn't confirm an unauthorized discount a borrower merely CLAIMS was
# pre-approved; this covers the other half -- a borrower who openly pushes
# back and asks for better, with no claim of prior approval at all. An LLM
# under social pressure to be "helpful" can drift toward implying
# flexibility ("let me see what I can do", "a human might approve more")
# that does not exist here.
_DISCOUNT_FIRMNESS = (
    "The one-time-settlement discount calculate_hypothetical returns "
    "(currently 5%) is a fixed policy figure, not a starting offer -- if a "
    "borrower pushes back and asks for a bigger discount, don't invent a "
    "higher number and don't imply a human could approve more through this "
    "channel. State the real discount from an actual calculate_hypothetical "
    "call and say plainly that it's fixed, not negotiable through this "
    "channel."
)

# Found live via a real Telegram conversation: calculate_hypothetical
# correctly computed a 3-month extension's real numbers, but the model's
# reply described them as already applied ("the loan now has 17 months
# left") -- and there was, at the time, no tool that could ever make that
# true. propose_restructuring now exists specifically to close that gap,
# but the tool alone doesn't stop the model from describing ITS result
# the same overconfident way, so this has to be prompted explicitly too --
# same reason _NO_FABRICATED_ACTIONS exists alongside check_fabricated_action.
_RESTRUCTURING_APPROVAL_FLOW = (
    "calculate_hypothetical only previews an extend-tenure option -- it "
    "never applies anything. Once the borrower explicitly agrees to a "
    "SPECIFIC number of extra months (not just hearing the preview), call "
    "propose_restructuring with that exact number to actually queue it for "
    "a human to approve. Its result is always pending, never final -- "
    "state it that way: 'if this is approved, your EMI would become "
    "₹X over Y months -- I've sent this for approval and you'll hear back "
    "once it's reviewed,' never 'your loan now has Y months' or 'your EMI "
    "will be reduced to X.' Nothing in this system can apply a "
    "restructuring by itself; only a human approving it can."
)

# payment_history exists on the real Account model/Postgres table, but until
# now it was only ever exposed through the ops-only, staff-gated HTTP API --
# no borrower-facing tool let this agent answer "can you tell me my recent
# payment history" at all. Confirmed via a direct audit of tools/account_tools.py
# against accounts/models.py's Account.payment_history field, the same class
# of gap _GROUND_NACH_FAILURES above closed for the NACH mandate field --
# this has to be prompted explicitly since get_payment_status's own fields
# (balance, EMI, months remaining) say nothing about past payments, so a
# model relying on that tool alone has nothing to ground a history question
# in and no instruction telling it to look elsewhere.
_PAYMENT_HISTORY_REQUESTS = (
    "If a borrower asks about their past payments or payment history -- "
    "which EMIs were paid, when, or whether any were late -- call "
    "get_payment_history(account_id) (or reuse its result from earlier "
    "this conversation) and answer only from what it actually returns, "
    "most recent first. get_payment_status's balance/EMI/months-remaining "
    "fields say nothing about past payments -- don't infer payment "
    "history from those instead of calling the tool that actually covers "
    "it."
)

# There is no document-generation or email/delivery capability anywhere in
# this system (_NO_FABRICATED_ACTIONS above) -- so a loan-closure-
# certificate/NOC request, common once a borrower's months_remaining
# reaches 0, was previously unhandled by any tool at all. Confirmed via the
# same audit that found the payment-history gap above.
# request_closure_certificate checks real eligibility and, if eligible,
# only queues a human to issue the actual document -- it never produces or
# sends one itself, so the model still has to say that plainly rather than
# implying the certificate itself is on its way through this chat.
_CLOSURE_CERTIFICATE_REQUESTS = (
    "If a borrower asks for a loan closure certificate, NOC, or "
    "confirmation that their loan is fully paid off, call "
    "request_closure_certificate(account_id). If it comes back "
    "eligible, tell the borrower plainly that this chat cannot generate "
    "or send the actual certificate -- it has been queued for a human "
    "who will issue and deliver the real document, the same limitation "
    "as any other document request (see the no-fabricated-actions rule "
    "above). If it comes back not eligible, say plainly, using the real "
    "months_remaining value returned, that the loan isn't fully repaid "
    "yet -- don't escalate it yourself and don't imply a certificate is "
    "coming when it isn't."
)

# A borrower saying "I want to file a complaint" or "I'm not happy with
# how this was handled" is a categorically different thing from a routine
# billing dispute or an ordinary escalate_to_human call -- it's a
# complaint about HANDLING itself (this agent's, or a human's), which is
# exactly what data/kb/grievance_redressal.md now covers and nothing did
# before. Confirmed via a direct audit of data/kb/*.md and this prompt --
# escalation_policy.md only ever covered handing an account to a human
# HERE, never what a dissatisfied borrower can do if that still doesn't
# resolve things, which RBI's Fair Practices Code requires this lender to
# have an answer for (an internal grievance process, and the Ombudsman if
# that's not enough). Same provenance as _GROUND_NACH_FAILURES and
# _FRAUD_OR_IDENTITY_CLAIM above -- a real gap found by reading the KB
# against a real regulatory requirement, not a benchmark failure. No
# Ombudsman phone number, address, or URL is stated here or anywhere in
# this system (see _NO_FABRICATED_LINKS/_NO_FABRICATED_ACTIONS above) --
# there is no source of truth for one to ground it in, so inventing one
# would be exactly the class of fabrication those rules already guard
# against.
_GRIEVANCE_REDRESSAL = (
    "If a borrower explicitly says they want to file a complaint, or that "
    "they're dissatisfied with how their case (not just a charge or a "
    "figure) was handled -- by you or by a human here -- treat that as a "
    "grievance, not a routine dispute or an ordinary query. Call "
    "check_policy to ground your answer in grievance_redressal.md rather "
    "than explaining the process from memory, and call escalate_to_human "
    "with a reason that says plainly this is a grievance about handling, "
    "not a routine account matter, so whoever picks it up triages it "
    "correctly. If the borrower says the internal response still hasn't "
    "resolved things, tell them plainly that they can take it to the RBI "
    "Banking/NBFC Ombudsman -- but never state a specific phone number, "
    "address, or URL for it from memory, since none is grounded anywhere "
    "in this system; say a human handling their account can provide the "
    "actual current contact details instead."
)

_NO_ACCOUNT_CONTEXT = (
    "You do not yet have access to any real account data or tools -- say "
    "so plainly if asked something you can't actually check. "
)

# The TTS engine for Hindi (Meta's MMS-TTS) only recognizes Devanagari
# characters -- its tokenizer returns an empty token sequence on Romanized
# Hindi ("Hinglish" in Latin script) and the model crashes downstream on
# that empty input. So each reply is single-script for now: full Devanagari
# Hindi, or full English -- not true intra-sentence code-switching, which
# would need the reply split into per-language runs and stitched across two
# TTS calls, deferred until the pipeline needs it.
_LANGUAGE_INSTRUCTIONS = {
    "en": "Reply in plain English.",
    "hi": (
        "Reply in Hindi, written entirely in Devanagari script (e.g. "
        "आपके), never in Romanized/Latin-script Hindi -- the "
        "text-to-speech engine cannot pronounce Romanized Hindi at all."
    ),
}


def client() -> Groq:
    global _current_key_index
    if _current_key_index > 0 and _switched_at is not None:
        if time.time() - _switched_at >= _FALLBACK_COOLDOWN_SECONDS:
            _current_key_index = 0  # cooldown elapsed -- give the primary another chance

    env_var = _active_env_var()
    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(f"{env_var} is not set -- copy .env.example to .env and fill it in")
    return Groq(api_key=api_key)


def switch_to_fallback_key() -> bool:
    """Called when the currently active key's requests start failing
    with a rate limit. Advances to the next configured fallback
    (ALTERNATE_GROQ_KEY, then ALTERNATE_GROQ_KEY2, ...), if there's one
    left that hasn't been tried yet this round. Returns True if it
    actually advanced, False if every configured key is already in use
    -- the caller should let the original error propagate in that case."""
    global _current_key_index, _switched_at
    fallback_names = _fallback_env_var_names()
    if _current_key_index < len(fallback_names):
        _current_key_index += 1
        _switched_at = time.time()
        return True
    return False


def build_system_prompt(language: str = "en", account_id: str | None = None, template: str | None = None) -> str:
    """account_id, when given, tells the agent which borrower it's
    speaking with -- it's how the model knows what to pass as account_id
    when it calls a tool, the same way a real deployment would identify
    the caller by phone/WhatsApp number rather than asking them to state
    an internal account ID out loud.

    template, when given, OVERRIDES the default, hand-tuned
    _SYSTEM_PROMPT_TEMPLATE below -- used only by agent/prompt_versions.py's
    A/B routing to run an alternate, stored prompt variant against a real
    percentage of conversations. Every existing caller that doesn't pass
    it keeps using the default template, unchanged. A variant template
    must use the exact same {placeholder} names as the default -- this
    still .format()s it with the same keyword arguments either way."""
    account_context = (
        f"You are speaking with the borrower on account {account_id}. "
        if account_id
        else _NO_ACCOUNT_CONTEXT
    )
    return (template or _SYSTEM_PROMPT_TEMPLATE).format(
        account_context=account_context,
        language_instruction=_LANGUAGE_INSTRUCTIONS[language],
        commitment_discipline=_COMMITMENT_DISCIPLINE,
        dispute_handling=_DISPUTE_HANDLING,
        read_only_tools=_READ_ONLY_TOOLS,
        ground_account_facts=_GROUND_ACCOUNT_FACTS,
        untracked_account_data=_UNTRACKED_ACCOUNT_DATA,
        no_fabricated_links=_NO_FABRICATED_LINKS,
        no_fabricated_actions=_NO_FABRICATED_ACTIONS,
        ground_policy_claims=_GROUND_POLICY_CLAIMS,
        check_dispute_block_first=_CHECK_DISPUTE_BLOCK_FIRST,
        out_of_domain_legal=_OUT_OF_DOMAIN_LEGAL,
        ground_nach_failures=_GROUND_NACH_FAILURES,
        fraud_or_identity_claim=_FRAUD_OR_IDENTITY_CLAIM,
        due_date_change_requests=_DUE_DATE_CHANGE_REQUESTS,
        cibil_credit_score=_CIBIL_CREDIT_SCORE,
        discount_firmness=_DISCOUNT_FIRMNESS,
        restructuring_approval_flow=_RESTRUCTURING_APPROVAL_FLOW,
        payment_history_requests=_PAYMENT_HISTORY_REQUESTS,
        closure_certificate_requests=_CLOSURE_CERTIFICATE_REQUESTS,
        grievance_redressal=_GRIEVANCE_REDRESSAL,
        no_mental_math=_NO_MENTAL_MATH,
    )


def reply(user_message: str, language: str = "en") -> str:
    """A single-turn reply with no memory of prior turns and no tool
    access -- the placeholder reasoning step for the voice I/O shell.
    language is "en" or "hi" and controls which script the reply must use."""
    completion = client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt(language)},
            {"role": "user", "content": user_message},
        ],
    )
    return completion.choices[0].message.content

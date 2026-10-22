"use strict";

/* ============================================================
   State
   ============================================================ */
const state = {
  apiKey: localStorage.getItem("bf_ops_key") || "",
  view: "overview",
  accounts: [],
  metrics: null,
  inbox: [], // every open case (GET /inbox): payment claims, disputes, requests, system notices
  activeFilter: "all",
  sortBy: "urgency", // "urgency" | "risk" | "id" -- see filteredAccounts()
  search: "",
  selectedAccountIds: new Set(),
  sinceHours: 24,
  detailAccountId: null,
  detailTab: "overview", // "overview" | "cases" | "payments" | "documents"
  refreshTimer: null,
};

const FLAG_LABELS = {
  overdue: "Overdue",
  disputed: "Disputed",
  broken_promises: "Broken promises",
};

const LANGUAGE_LABELS = { hi: "Hindi", en: "English", hinglish: "Hinglish" };

// Mirrors accounts/policy.py -- used only for display wording.
const GRACE_PERIOD_DAYS = 3;

// What an escalation IS (the API's `kind`, from accounts/escalation_kinds.py).
// Before, a fraud claim, a guardrail artifact, a chronic-delinquency notice and a
// borrower's call-back request were indistinguishable cards with the same
// Approve/Reject pair.
// Short, so the pill says what TYPE a case is and the title next to it says what it's about.
const KIND_LABELS = {
  payment_claim: "Payment claim",
  dispute: "Dispute",
  fraud: "Urgent",
  request: "Request",
  restructuring: "More time",
  callback: "Call-back",
  closure: "Closure",
  system: "System",
};

// "Dispute · Dispute" says nothing twice: the title shows only when it adds to the pill.
function caseHeading(c) {
  const pill = KIND_LABELS[c.kind] || c.kind;
  return c.title && c.title.toLowerCase() !== pill.toLowerCase() ? c.title : "";
}
// Triage order, lowest first (mirrors escalation_kinds.URGENCY_RANK): a fraud
// claim must not sit at the bottom of an oldest-first list just because it is new.
const KIND_URGENCY = { fraud: 0, payment_claim: 1, dispute: 1, request: 1, restructuring: 2, callback: 2, closure: 3, system: 4 };

// What was sent, for the profile's history (store.get_borrower_messages' kinds).
const SENT_KIND_LABELS = { message: "Message sent", decision: "Decision sent", dispute: "Dispute update sent", reminder: "Reminder sent" };

// The usual first reply to a payment claim, one tap away inside its reply box.
const ASK_REFERENCE_TEXT =
  "Thanks for letting us know. Could you share the payment reference (UTR or transaction ID), with the date and the amount? We'll match it against the bank record.";

const CALL_OUTCOME_LABELS = {
  reached: "Reached",
  no_answer: "No answer",
  voicemail: "Left voicemail",
  wrong_number: "Wrong number",
};

// Mirrors the <option> labels in the since-hours <select> -- shown in the
// KPI card instead of a raw hour count ("30d", not "720h").
const SINCE_HOURS_LABELS = { 24: "24h", 72: "3d", 168: "7d", 720: "30d" };

// loadAll() runs concurrently from three places: initial boot, the 30s
// auto-refresh, and a since-hours change -- an earlier call can resolve
// after a later one (e.g. the initial load lands after the operator has
// already switched the window) and would otherwise clobber fresher state
// with stale data. Each call captures the generation it started with and
// discards its own result if a newer call has since started.
let loadGeneration = 0;

/* ============================================================
   API layer
   ============================================================ */
class ApiError extends Error {
  constructor(status, detail, data = null) {
    super(detail || `request failed (${status})`);
    this.status = status;
    this.data = data; // a structured detail ({code, message, ...}), when the server sent one
  }
}

async function api(path, opts = {}) {
  const keyUsed = state.apiKey;
  const res = await fetch(path, {
    ...opts,
    headers: { "X-API-Key": keyUsed, ...(opts.headers || {}) },
  });
  if (res.status === 401) {
    // Lock only for the key that is in use now. During sign-in, signIn() shows its own message; and a request still in
    // flight from a sign-in that already failed (its key is gone) must not lock again over that message.
    if (!signingIn && keyUsed && keyUsed === state.apiKey) lock("That key was rejected by the server.");
    throw new ApiError(401, "unauthorized");
  }
  if (!res.ok) {
    let detail = `request failed (${res.status})`;
    let data = null;
    try {
      const body = await res.json();
      // FastAPI's own validation errors (422s) send detail as a LIST of
      // {loc, msg, type} objects, not a plain string -- passing that
      // straight into an Error's message stringifies to "[object
      // Object]" instead of anything readable. Every other error path
      // (ValueError, HTTPException(detail=str)) already sends a string.
      if (Array.isArray(body.detail)) {
        detail = body.detail.map((e) => e.msg || JSON.stringify(e)).join("; ");
      } else if (body.detail && typeof body.detail === "object") {
        data = body.detail; // e.g. confirm-payment's {code: "possible_duplicate", message, existing}
        detail = body.detail.message || JSON.stringify(body.detail);
      } else if (body.detail) {
        detail = body.detail;
      }
    } catch (_) {} // no JSON body: the status line above is the message
    throw new ApiError(res.status, detail, data);
  }
  if (res.status === 204) return null;
  return res.json();
}

const getAccounts = (flag) => api(`/accounts${flag ? `?flag=${flag}` : ""}`);
const getAccount = (id) => api(`/accounts/${id}`);
const getInbox = () => api("/inbox");
const getMetrics = (sinceHours) => api(`/metrics?since_hours=${sinceHours}`);
const approveEscalation = (id) => api(`/escalations/${id}/approve`, { method: "POST" });
const rejectEscalation = (id, reason) =>
  api(`/escalations/${id}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reason: reason || null }),
  });
const uploadDocument = (accountId, file, documentType) => {
  const form = new FormData();
  form.append("file", file);
  form.append("document_type", documentType);
  return api(`/accounts/${accountId}/documents`, { method: "POST", body: form });
};
const getDocuments = (accountId) => api(`/accounts/${accountId}/documents`);

// Not routed through api() -- that helper always calls res.json(), but a
// document download's real body is the file's bytes. Same auth (the
// X-API-Key header) and the same 401/error-detail handling as api()
// itself, just returning a Blob at the end instead of parsed JSON.
async function downloadDocument(accountId, filename) {
  const res = await fetch(`/accounts/${accountId}/documents/${encodeURIComponent(filename)}`, {
    headers: { "X-API-Key": state.apiKey },
  });
  if (res.status === 401) {
    lock("That key was rejected by the server.");
    throw new ApiError(401, "unauthorized");
  }
  if (!res.ok) {
    let detail = `download failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch (_) {}
    throw new ApiError(res.status, detail);
  }
  return res.blob();
}
const draftClarification = (accountId, operatorNote) =>
  api(`/accounts/${accountId}/clarification-requests/draft`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ operator_note: operatorNote }),
  });
const sendClarificationRequest = (accountId, message) =>
  api(`/accounts/${accountId}/clarification-requests`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
const createAccount = (payload) =>
  api("/accounts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
const recordPayment = (accountId, payload) =>
  api(`/accounts/${accountId}/payments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
const waiveCharge = (accountId, chargeId, payload) =>
  api(`/accounts/${accountId}/charges/${chargeId}/waive`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
const resolveDispute = (accountId, resolutionNote, disputeId = null) =>
  api(`/accounts/${accountId}/disputes/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resolution_note: resolutionNote || null, dispute_id: disputeId }),
  });
const replyToCase = (accountId, caseRef, message) =>
  api(`/accounts/${accountId}/cases/${encodeURIComponent(caseRef)}/reply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
const confirmPaymentClaim = (accountId, caseRef, payload) =>
  api(`/accounts/${accountId}/cases/${encodeURIComponent(caseRef)}/confirm-payment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
// How a promise was made, and each part's verdict, for the account history.
const PROMISE_VIA_LABELS = { chat: "Made in chat, terms read back", ops: "Logged by the team", reminder_button: "From a reminder button" };
const PROMISE_PART_LABELS = { open: "to pay", kept: "paid", broken: "missed", void: "cancelled" };

const logPromise = (accountId, payload) =>
  api(`/accounts/${accountId}/promises`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
const logCall = (accountId, payload) =>
  api(`/accounts/${accountId}/call-log`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
const resetAccessKey = (accountId) => api(`/accounts/${accountId}/reset-access-key`, { method: "POST" });
const updateAccount = (accountId, payload) =>
  api(`/accounts/${accountId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
const getConversation = (accountId) => api(`/accounts/${accountId}/conversation`);
const triggerOutboundRun = (accountIds) =>
  api("/outbound/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ account_ids: accountIds }),
  });
const sendBulkClarification = (accountIds, message) =>
  api("/clarification-requests/bulk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ account_ids: accountIds, message }),
  });
const escalateAccount = (accountId, reason) =>
  api(`/accounts/${accountId}/escalate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reason }),
  });

/* ============================================================
   Auth / unlock
   ============================================================ */
const $unlockScreen = document.getElementById("unlock-screen");
const $shell = document.getElementById("shell");
const $unlockForm = document.getElementById("unlock-form");
const $unlockInput = document.getElementById("unlock-input");
const $unlockError = document.getElementById("unlock-error");

function lock(message) {
  state.apiKey = "";
  localStorage.removeItem("bf_ops_key");
  stopAutoRefresh();
  // A locked dashboard keeps no microphone open, no answer on screen and no run in progress.
  stopVoiceRecording();
  closeVoiceReply();
  voice.workflow = null;
  renderVoiceWorkflowBar();
  $shell.hidden = true;
  $unlockScreen.hidden = false;
  $unlockInput.value = "";
  if (message) {
    $unlockError.textContent = message;
    $unlockError.hidden = false;
  }
  $unlockInput.focus();
}

/* Signing in IS the first dashboard load: the key is proven by that load succeeding, not by a separate request first.
   Found live: arriving with a key (from the borrower page's "Ops team" box, or one saved earlier) showed this sign-in
   form for ~12 s while /accounts was fetched once just to check the key, then fetched again for the dashboard: about
   26 s before any account appeared. Now a known key opens the dashboard at once, with placeholders, and loads once. */
let signingIn = false;
const $unlockSubmit = $unlockForm.querySelector("button[type=submit]");

function showUnlock(message) {
  stopAutoRefresh();
  $shell.hidden = true;
  $unlockScreen.hidden = false;
  $unlockError.textContent = message || "";
  $unlockError.hidden = !message;
  $unlockInput.focus();
}

function showDashboardPlaceholders() {
  $unlockScreen.hidden = true;
  $shell.hidden = false;
  document.getElementById("portfolio-summary").textContent = "Loading the portfolio…";
  document.getElementById("kpi-row").innerHTML = '<div class="kpi-card skeleton" aria-hidden="true"></div>'.repeat(5);
  document.getElementById("account-grid").innerHTML = '<div class="account-card skeleton" aria-hidden="true"></div>'.repeat(4);
}

async function signIn(key, { fromForm = false } = {}) {
  state.apiKey = key;
  signingIn = true;
  if (fromForm) {
    $unlockSubmit.disabled = true;
    $unlockSubmit.firstChild.textContent = "Signing in…";
  } else {
    showDashboardPlaceholders();
  }
  try {
    await loadAll();
  } catch (e) {
    state.apiKey = "";
    loadGeneration += 1; // whatever this attempt still has in flight (the escalation rate) is discarded on arrival
    const rejected = e instanceof ApiError && e.status === 401;
    if (rejected) localStorage.removeItem("bf_ops_key");
    showUnlock(rejected ? "That key was rejected — check it and try again." : "Couldn't reach the ops API. Check your connection and try again.");
    return false;
  } finally {
    signingIn = false;
    $unlockSubmit.disabled = false;
    $unlockSubmit.firstChild.textContent = "Unlock";
  }
  localStorage.setItem("bf_ops_key", key);
  $unlockError.hidden = true;
  $unlockScreen.hidden = true;
  $shell.hidden = false;
  startAutoRefresh();
  return true;
}

$unlockForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const key = $unlockInput.value.trim();
  if (key) signIn(key, { fromForm: true });
});

document.getElementById("lock-btn").addEventListener("click", () => lock());

/* ============================================================
   Tabs
   ============================================================ */
document.getElementById("tabs").addEventListener("click", (e) => {
  const btn = e.target.closest(".tab");
  if (!btn) return;
  setView(btn.dataset.view);
});

function setView(view) {
  state.view = view;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  document.getElementById("view-overview").hidden = view !== "overview";
  document.getElementById("view-inbox").hidden = view !== "inbox";
  document.getElementById("view-bank").hidden = view !== "bank";
  document.getElementById("view-quality").hidden = view !== "quality";
  if (view === "bank") loadBank();
  if (view === "quality") loadQuality();
}

/* ============================================================
   Living demo -- GET /demo, POST /demo/reset
   ============================================================ */
async function initDemoReset() {
  const $btn = document.getElementById("demo-reset-btn");
  try {
    const status = await api("/demo");
    $btn.hidden = !status.enabled;
  } catch (err) {
    console.warn("ops: demo status could not be loaded; the reset button stays hidden", err);
    return;
  }
  $btn.onclick = async () => {
    const ok = window.confirm(
      "Put the demo borrowers back to their starting story?\n\nPayments, promises, late fees, cases and messages on the demo accounts are replaced, with dates counted from today. Accounts you opened yourself are not touched."
    );
    if (!ok) return;
    $btn.disabled = true;
    try {
      await api("/demo/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: "reset" }) });
      toast("The demo is back to its starting story.");
      await loadAll();
    } catch (err) {
      toast(err.message, true);
    } finally {
      $btn.disabled = false;
    }
  };
}

/* ============================================================
   Loan-book import -- POST /loan-book (ops/loan_book.py): check, then open; access keys shown once
   ============================================================ */
(function initLoanBook() {
  const $panel = document.getElementById("loan-book-panel");
  const $file = document.getElementById("loan-book-file");
  const $check = document.getElementById("loan-book-check");
  const $import = document.getElementById("loan-book-import");
  const $result = document.getElementById("loan-book-result");
  document.getElementById("loan-book-btn").addEventListener("click", () => {
    $panel.hidden = !$panel.hidden;
  });
  $file.addEventListener("change", () => {
    $import.hidden = true;
    $result.innerHTML = "";
  });

  async function send(dryRun) {
    const file = $file.files[0];
    if (!file) {
      toast("Choose the loan-book CSV first.", true);
      return null;
    }
    const form = new FormData();
    form.append("file", file);
    form.append("dry_run", dryRun ? "true" : "false");
    return api("/loan-book", { method: "POST", body: form });
  }

  function rowsHtml(r) {
    const bad = r.details.filter((d) => d.errors.length);
    const opened = r.details.filter((d) => d.access_key);
    const skipped = r.details.filter((d) => d.account_id && !d.access_key && !d.errors.length);
    let html = `<p class="upload-result">${r.rows} loan${r.rows === 1 ? "" : "s"} read · ${r.ready} ${r.imported ? "opened" : "ready to open"} · ${r.already_on_books} already here · ${r.with_errors} with something to fix</p>`;
    if (bad.length) {
      html += `<ul class="loan-book-errors">${bad.map((d) => `<li><b>Row ${d.row}${d.borrower_name ? ` (${escapeHtml(d.borrower_name)})` : ""}:</b> ${d.errors.map(escapeHtml).join("; ")}</li>`).join("")}</ul>`;
    }
    if (skipped.length) {
      html += `<p class="section-subtitle">Already here, skipped: ${skipped.map((d) => `${escapeHtml(d.external_ref)} → ${escapeHtml(d.account_id)}`).join(", ")}</p>`;
    }
    if (opened.length) {
      html += `<p class="loan-book-once"><b>Each access key is shown only now</b> — the system keeps only a scrambled copy. Download them before leaving this page.</p>
        <table class="loan-book-keys"><thead><tr><th>Row</th><th>Account</th><th>Borrower</th><th>Access key</th></tr></thead><tbody>
        ${opened.map((d) => `<tr><td>${d.row}</td><td>${escapeHtml(d.account_id)}</td><td>${escapeHtml(d.borrower_name || "")}</td><td><code>${escapeHtml(d.access_key)}</code></td></tr>`).join("")}
        </tbody></table>
        <button type="button" class="btn btn-primary" id="loan-book-download">Download access keys (CSV)</button>`;
    }
    return html;
  }

  $check.addEventListener("click", async () => {
    $check.disabled = true;
    try {
      const r = await send(true);
      if (!r) return;
      $result.innerHTML = rowsHtml(r);
      $import.hidden = !(r.with_errors === 0 && r.ready > 0);
      $import.textContent = `Open the ${r.ready} loan${r.ready === 1 ? "" : "s"}`;
    } catch (err) {
      $result.innerHTML = `<p class="mini-form-error">${escapeHtml(err.message)}</p>`;
    } finally {
      $check.disabled = false;
    }
  });

  $import.addEventListener("click", async () => {
    $import.disabled = true;
    try {
      const r = await send(false);
      if (!r) return;
      $result.innerHTML = rowsHtml(r);
      $import.hidden = true;
      const opened = r.details.filter((d) => d.access_key);
      const $dl = document.getElementById("loan-book-download");
      if ($dl) {
        $dl.addEventListener("click", () => {
          const lines = ["row,account_id,borrower_name,loan_reference,access_key,telegram_link",
            ...opened.map((d) => [d.row, d.account_id, d.borrower_name || "", d.external_ref || "", d.access_key, d.telegram_invite_link || ""]
              .map((x) => `"${String(x).replaceAll('"', '""')}"`).join(","))];
          const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
          const a = Object.assign(document.createElement("a"), { href: url, download: "access-keys.csv" });
          a.click();
          URL.revokeObjectURL(url);
        });
      }
      toast(`${opened.length} loan${opened.length === 1 ? "" : "s"} opened.`);
      await loadAll();
    } catch (err) {
      $result.innerHTML = `<p class="mini-form-error">${escapeHtml(err.message)}</p>`;
    } finally {
      $import.disabled = false;
    }
  });
})();

/* ============================================================
   Quality -- GET /quality (eval/conversation_lab.py + the nightly suites)
   ============================================================ */
// The per-component report (observability/trace_report.py): verdict weakest first, safeguards, each component and its
// actions, each chain with where its time goes and where failures start, outcomes judged by what happened later.
const Q_STATUS = { red: "Needs a look", amber: "Watch", green: "Healthy", not_enough_data: "Too few to judge" };
function qPct(v) {
  return v === null || v === undefined ? "–" : `${Math.round(v * 1000) / 10}%`;
}
function qSec(v) {
  return v === null || v === undefined ? "–" : `${v < 1 ? Math.round(v * 1000) + " ms" : (Math.round(v * 10) / 10) + " s"}`;
}
function componentsHtml(r, title, sub) {
  if (!r) return `<div class="q-comp-report"><h2>${escapeHtml(title)}</h2><p class="empty-state">No report yet.</p></div>`;
  const gates = r.gates.map((g) => `<li class="${g.passed ? "ok" : "bad"}">${g.passed ? "✓" : "⛔"} ${escapeHtml(g.name)} <span class="muted">— ${escapeHtml(g.detail)}</span></li>`).join("");
  const verdict = r.verdict.map((v) => `<span class="q-chip ${v.status}" title="${escapeHtml(v.why)}">${escapeHtml(v.component)} · ${Q_STATUS[v.status]}</span>`).join("");
  const comps = r.components.map((c) => `<details class="q-comp ${c.status}">
      <summary><span class="q-dot"></span><b>${escapeHtml(c.component)}</b>
        <span class="q-stats">${c.steps} steps · ${qPct(c.failed_rate)} failing a check${c.previous_failed_rate !== null ? ` (was ${qPct(c.previous_failed_rate)})` : ""} · p50 ${qSec(c.p50_seconds)} · p90 ${qSec(c.p90_seconds)}${c.tokens_in ? ` · ${c.tokens_in.toLocaleString("en-IN")} tokens in` : ""}</span></summary>
      ${c.checks.length ? `<table class="q-table"><tr><th>Check</th><th>Times</th><th>Share</th><th>What to do</th></tr>${c.checks.map((k) => `<tr><td>${k.critical ? "⚠ " : ""}${escapeHtml(k.means)}</td><td>${k.count}</td><td>${qPct(k.rate)}</td><td>${escapeHtml(k.action)}</td></tr>`).join("")}</table>` : `<p class="muted">Every step passed its checks.</p>`}
      <table class="q-table"><tr><th>Action</th><th>Count</th><th>Failing</th><th>p50</th></tr>${c.actions.map((a) => `<tr><td>${escapeHtml(a.name)}</td><td>${a.count}</td><td>${qPct(a.failed_rate)}</td><td>${qSec(a.p50_seconds)}</td></tr>`).join("")}</table>
    </details>`).join("");
  const chains = r.chains.map((ch) => {
    const bar = Object.entries(ch.time_share).map(([step, share]) => `<span class="q-seg q-${escapeHtml(step)}" style="flex:${share}" title="${escapeHtml(step)}: ${Math.round(share * 100)}% of the time">${share >= 0.12 ? escapeHtml(step) : ""}</span>`).join("");
    const starts = Object.entries(ch.failures_start_at).map(([step, n]) => `${escapeHtml(step)} ${n}`).join(", ");
    return `<div class="q-chain"><div><b>${escapeHtml(ch.kind.replaceAll("_", " "))}</b> <span class="muted">${ch.turns} turns · ${qPct(ch.success_rate)} clean · p50 ${qSec(ch.p50_seconds)} · p90 ${qSec(ch.p90_seconds)}</span></div>
      <div class="q-timebar">${bar}</div>${starts ? `<div class="muted">Failures start at: ${starts}</div>` : ""}</div>`;
  }).join("");
  const o = r.outcomes || {};
  const outcomes = [
    o.reminders ? `Reminders: <b>${o.reminders.delivered} of ${o.reminders.sent}</b> reached the borrower on Telegram (${qPct(o.reminders.delivered_rate)})` : "",
    o.repeats ? `Borrowers repeating themselves within 2 minutes: <b>${o.repeats.repeated} of ${o.repeats.messages}</b> messages` : "",
    o.bank_matcher && o.bank_matcher.lines ? `Bank matcher: ${o.bank_matcher.auto_posted} posted on its own; suggestions kept by people <b>${o.bank_matcher.suggestions_kept} of ${o.bank_matcher.suggestions_decided}</b>` : "",
    o.promises ? Object.entries(o.promises.by_channel).map(([via, p]) => `Promises made via ${escapeHtml(via)}: <b>${p.kept} kept, ${p.broken} broken</b>`).join("<br>") : "",
  ].filter(Boolean).map((x) => `<li>${x}</li>`).join("");
  const evidence = r.evidence.map((e) => `<li><code>#${e.id}</code> ${escapeHtml(e.kind.replaceAll("_", " "))}${e.account_id ? ` · ${escapeHtml(e.account_id)}` : ""} · ${qSec(e.seconds)} — ${e.failed.map(escapeHtml).join(", ")}${e.error ? ` · error ${escapeHtml(e.error)}` : ""}</li>`).join("");
  return `<div class="q-comp-report">
    <h2>${escapeHtml(title)} <span class="muted">· ${escapeHtml(sub)} · ${r.window.traces} turns${r.window.capped ? " (capped)" : ""}</span></h2>
    <div class="q-verdict">${verdict || `<span class="muted">No AI steps recorded in this window.</span>`}</div>
    ${gates ? `<ul class="q-gates">${gates}</ul>` : ""}
    ${comps}
    ${chains ? `<h3>Chains</h3>${chains}` : ""}
    ${outcomes ? `<h3>What happened afterwards</h3><ul class="q-outcomes">${outcomes}</ul>` : ""}
    ${evidence ? `<details class="q-evidence"><summary>Worst turns (trace ids)</summary><ul>${evidence}</ul></details>` : ""}
  </div>`;
}

async function loadQuality() {
  const $body = document.getElementById("quality-body");
  try {
    const q = await api("/quality");
    const labs = q.labs.map((lab) => {
      const s = lab.summary;
      const cases = lab.results
        .map((r) => `<details class="q-case ${r.passed ? "pass" : "fail"}">
          <summary><span class="q-dot"></span>${escapeHtml(r.case_id.replaceAll("_", " "))}<span class="q-src">${r.source}${r.stale_recording ? " · recorded under an older prompt" : ""}</span></summary>
          <p class="q-why">${escapeHtml(r.why)}</p>
          <p class="q-label">Labelled from: ${escapeHtml(r.label_source)}</p>
          ${r.turns.map((t, i) => `<div class="q-turn"><b>Borrower:</b> ${escapeHtml(t.user)}<br><b>Reply:</b> ${escapeHtml(t.reply)}${t.problems.length ? `<ul class="q-problems">${t.problems.map((p) => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}</div>`).join("")}
        </details>`)
        .join("");
      return `<div class="q-lab">
        <h2>Conversation lab <span class="muted">· prompt ${escapeHtml(s.prompt_version)} · ${escapeHtml(s.mode)} · ${fmtDateTime(s.ran_at)}</span></h2>
        <p class="q-score"><b>${s.passed} of ${s.cases}</b> cases right · ${s.recorded} with the real model's recorded words, the rest scripted</p>
        ${cases}
      </div>`;
    }).join("") || `<p class="empty-state">The conversation lab has not run on this server yet.</p>`;
    const nightly = q.nightly.map((n) => {
      const last = n.last_run;
      const summary = last ? Object.entries(last.summary || {}).slice(0, 4).map(([k, v]) => `${escapeHtml(k.replaceAll("_", " "))}: <b>${typeof v === "number" && v <= 1 && !Number.isInteger(v) ? Math.round(v * 100) + "%" : escapeHtml(String(v))}</b>`).join(" · ") : "not run yet";
      return `<div class="q-suite"><div><b>${escapeHtml(n.suite.replaceAll("_", " "))}</b> <span class="muted">— ${escapeHtml(n.checks)}</span></div><div class="q-suite-sum">${summary}${last ? ` <span class="muted">(${fmtDateTime(last.recorded_at)})</span>` : ""}</div></div>`;
    }).join("");
    const components = [
      componentsHtml(q.components_live, "Every AI step, live", "real conversations, last 7 days"),
      componentsHtml(q.components_nightly, "Every AI step, nightly", "the nightly suites' turns, on a copy of the data"),
    ].join("");
    $body.innerHTML = `${components}${labs}<div class="q-nightly"><h2>Nightly suites <span class="muted">· the real model, on a copy of the data</span></h2>${nightly}</div>`;
  } catch (err) {
    $body.innerHTML = `<p class="mini-form-error">${escapeHtml(err.message)}</p>`;
  }
}

/* ============================================================
   Bank statement (cash application) -- POST /bank-statements, GET /bank-lines
   ============================================================ */
const BANK_NAME_KEY = "bf_ops_name";
function bankName() {
  const $name = document.getElementById("bank-your-name");
  const name = $name.value.trim();
  if (name.length < 2) {
    $name.focus();
    throw new Error("Enter your name first -- it is kept on every line you post or set aside.");
  }
  try {
    localStorage.setItem(BANK_NAME_KEY, name);
  } catch (e) {
    // Storage blocked (private window): the name just is not remembered next time.
  }
  return name;
}

function bankLineHtml(l, open) {
  const accountOptions = state.accounts
    .map((a) => `<option value="${escapeHtml(a.account_id)}"${a.account_id === l.account_id ? " selected" : ""}>${escapeHtml(a.account_id)} — ${escapeHtml(a.borrower_name)}</option>`)
    .join("");
  const who = l.account_id ? state.accounts.find((a) => a.account_id === l.account_id) : null;
  const outcome = l.status === "posted"
    ? `Recorded on <b>${escapeHtml(l.account_id || "")}</b>${who ? ` (${escapeHtml(who.borrower_name)})` : ""} by ${escapeHtml(l.decided_by || "")}${l.dispute_id ? ` — closed claim D-${l.dispute_id}` : ""}`
    : l.status === "ignored"
      ? `Set aside by ${escapeHtml(l.decided_by || "")}: ${escapeHtml(l.decided_note || "")}`
      : "";
  return `<div class="bank-line ${l.status}" data-line-id="${l.id}">
    <div class="bank-line-main">
      <div class="bank-line-amount">${fmtInr(l.amount)}</div>
      <div class="bank-line-text">
        <div>${escapeHtml(l.narration || "(no narration)")}${l.reference ? ` <span class="bank-ref">ref ${escapeHtml(l.reference)}</span>` : ""}</div>
        <div class="bank-line-meta">${fmtDate(l.line_date)} · ${escapeHtml(l.statement)} row ${l.line_no}</div>
        ${open ? `<div class="bank-line-reason">${escapeHtml(l.match_reason || "")}</div>` : `<div class="bank-line-reason">${outcome}</div>`}
      </div>
    </div>
    ${open ? `<div class="bank-line-actions">
      <select class="bank-account">${l.account_id ? "" : '<option value="">Pick the borrower…</option>'}${accountOptions}</select>
      <button type="button" class="btn btn-primary" data-bank-action="post">Record payment</button>
      <button type="button" class="btn btn-secondary" data-bank-action="ignore">Not a loan payment</button>
      <p class="mini-form-error" hidden></p>
    </div>` : ""}
  </div>`;
}

async function loadBank() {
  try {
    const [open, done] = await Promise.all([api("/bank-lines?status=open"), api("/bank-lines?status=done")]);
    document.getElementById("bank-open").innerHTML = open.map((l) => bankLineHtml(l, true)).join("");
    document.getElementById("bank-open-empty").hidden = open.length > 0;
    document.getElementById("bank-done").innerHTML = done.slice(0, 30).map((l) => bankLineHtml(l, false)).join("");
    const badge = document.getElementById("bank-count-badge");
    badge.textContent = String(open.length);
    badge.hidden = open.length === 0;
  } catch (err) {
    toast(err.message, true);
  }
}

(function initBank() {
  try {
    document.getElementById("bank-your-name").value = localStorage.getItem(BANK_NAME_KEY) || "";
  } catch (e) {
    // Storage blocked: the field starts empty.
  }
  document.getElementById("bank-upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const $result = document.getElementById("bank-upload-result");
    const $btn = document.getElementById("bank-upload-btn");
    const file = document.getElementById("bank-file").files[0];
    if (!file) return;
    try {
      bankName();
    } catch (err) {
      toast(err.message, true);
      return;
    }
    $btn.disabled = true;
    $btn.textContent = "Matching…";
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await api("/bank-statements", { method: "POST", body: form });
      const bits = [`${r.posted} recorded`, `${r.suggested} waiting with a guess`, `${r.unmatched} unmatched`];
      if (r.already_uploaded) bits.push(`${r.already_uploaded} already uploaded before (skipped)`);
      if (r.debits_skipped) bits.push(`${r.debits_skipped} debits skipped`);
      if (r.rejected_rows.length) bits.push(`${r.rejected_rows.length} rows unreadable (first: row ${r.rejected_rows[0].row}, ${r.rejected_rows[0].reason})`);
      $result.textContent = `${r.statement}: ${bits.join(" · ")}.`;
      $result.hidden = false;
      await loadBank();
      if (r.posted) await loadAll();
    } catch (err) {
      toast(err.message, true);
    } finally {
      $btn.disabled = false;
      $btn.textContent = "Upload and match";
    }
  });
  document.getElementById("bank-open").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-bank-action]");
    if (!btn) return;
    const $line = btn.closest(".bank-line");
    const $err = $line.querySelector(".mini-form-error");
    $err.hidden = true;
    let name;
    try {
      name = bankName();
    } catch (err) {
      $err.textContent = err.message;
      $err.hidden = false;
      return;
    }
    const id = $line.dataset.lineId;
    btn.disabled = true;
    try {
      if (btn.dataset.bankAction === "post") {
        const accountId = $line.querySelector(".bank-account").value;
        if (!accountId) throw new Error("Pick the borrower this payment is from.");
        try {
          await api(`/bank-lines/${id}/post`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ account_id: accountId, decided_by: name }) });
        } catch (err) {
          if (!(err.data && err.data.code === "possible_duplicate")) throw err;
          if (!window.confirm(`${err.data.message}\n\nRecord it anyway?`)) return;
          await api(`/bank-lines/${id}/post`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ account_id: accountId, decided_by: name, record_even_if_duplicate: true }) });
        }
        toast("Payment recorded.");
        await loadAll();
      } else {
        const note = window.prompt("Why is this not a loan payment? (kept on the line)");
        if (!note || note.trim().length < 3) return;
        await api(`/bank-lines/${id}/ignore`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decided_by: name, note: note.trim() }) });
        toast("Set aside.");
      }
      await loadBank();
    } catch (err) {
      $err.textContent = err.message;
      $err.hidden = false;
    } finally {
      btn.disabled = false;
    }
  });
})();

/* ============================================================
   Since-hours selector
   ============================================================ */
// The time window only changes the escalation rate, so the selector sits on that card, next to the number it
// changes. Found live: as a global "Activity window" control in the sidebar, changing it seemed to do nothing.
document.getElementById("kpi-row").addEventListener("change", async (e) => {
  if (!e.target.matches("#since-hours")) return;
  state.sinceHours = Number(e.target.value);
  try {
    state.metrics = await getMetrics(state.sinceHours);
    renderKpis();
  } catch (err) {
    toast(err.message, true);
  }
});

/* ============================================================
   Avatar helpers
   ============================================================ */
// Flat tint + ink pairs (no gradients, per the design system), all AA for the
// initials on their own tint; picked by a stable hash so an account keeps its colour.
const AVATAR_TINTS = [
  ["#e3eefb", "#0b4f8a"],
  ["#dff3ee", "#0f6b55"],
  ["#ece9fb", "#4b3fa8"],
  ["#fdeee0", "#8a4c00"],
  ["#e8edf5", "#26324a"],
];

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h;
}

function initials(name) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0].toUpperCase())
    .join("");
}

function avatarStyle(seed) {
  const [bg, ink] = AVATAR_TINTS[hashStr(seed) % AVATAR_TINTS.length];
  return `background: ${bg}; color: ${ink};`;
}

function fmtInr(n) {
  return "₹" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

// Compact form for space-constrained labels (bar chart amounts) --
// 12500 -> "12.5k", 250000 -> "2.5L" (lakh, the conventional Indian
// grouping), falling back to the full rupee figure under 1,000.
function fmtInrShort(n) {
  const v = Number(n);
  if (v >= 1e7) return "₹" + (v / 1e7).toFixed(v % 1e7 === 0 ? 0 : 1) + "Cr";
  if (v >= 1e5) return "₹" + (v / 1e5).toFixed(v % 1e5 === 0 ? 0 : 1) + "L";
  if (v >= 1e3) return "₹" + (v / 1e3).toFixed(v % 1e3 === 0 ? 0 : 1) + "k";
  return fmtInr(v);
}

// Human-readable file size for the documents list -- 1536 -> "1.5 KB",
// 3145728 -> "3.0 MB", falling back to plain bytes under 1KB.
function fmtFileSize(bytes) {
  if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  if (bytes >= 1024) return (bytes / 1024).toFixed(1) + " KB";
  return bytes + " B";
}

function fmtDate(d) {
  return new Date(d).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

function fmtDateTime(iso) {
  return new Date(iso).toLocaleString("en-IN", { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
}

function fmtShortDate(d) {
  return new Date(d).toLocaleDateString("en-IN", { day: "numeric", month: "short" });
}

function relativeTime(iso) {
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s ?? "";
  return div.innerHTML;
}

/* ============================================================
   Toast
   ============================================================ */
let toastTimer = null;
function toast(message, isError = false) {
  const el = document.getElementById("toast");
  el.innerHTML = `<span class="dot"></span>${escapeHtml(message)}`;
  el.classList.toggle("error", isError);
  el.hidden = false;
  // setTimeout, not requestAnimationFrame -- rAF only fires while the
  // page is actually visible/composited (Page Visibility API), which a
  // backgrounded or non-focused tab may never satisfy; a short timeout
  // still forces the initial state to paint before the class flip
  // triggers the CSS transition, without that dependency.
  setTimeout(() => el.classList.add("show"), 10);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => (el.hidden = true), 300);
  }, 3200);
}

/* ============================================================
   KPI row
   ============================================================ */
// A one-line, plain-language read of the whole portfolio -- the first
// thing anyone unfamiliar with this dashboard sees, so they don't have to
// mentally tally the account cards themselves to know what's going on.
function renderPortfolioSummary(accounts, openCases) {
  const el = document.getElementById("portfolio-summary");
  if (!accounts.length) {
    el.textContent = "No accounts on the books yet.";
    return;
  }
  const flaggedCount = accounts.filter((a) => a.flags.length > 0).length;
  const cleanCount = accounts.length - flaggedCount;
  const parts = [`${accounts.length} account${accounts.length === 1 ? "" : "s"} on the books`];
  if (flaggedCount > 0) {
    parts.push(`${flaggedCount} need${flaggedCount === 1 ? "s" : ""} attention`);
  }
  if (cleanCount > 0) {
    parts.push(`${cleanCount} clean`);
  }
  if (openCases > 0) {
    parts.push(`${openCases} open case${openCases === 1 ? "" : "s"} waiting on a human`);
  }
  el.textContent = parts.join(" — ");
}

function renderKpis() {
  const accounts = state.accounts;
  const overdueAccounts = accounts.filter((a) => a.flags.some((f) => f.label === "overdue"));
  const overdueCount = overdueAccounts.length;
  // A lender acts on rupee exposure, not headcount -- before this, getting
  // this figure meant opening every flagged card and adding EMI amounts by
  // hand. emi_amount is already on every AccountSummaryOut; this is a pure
  // client-side sum, no new backend data.
  const overdueRupees = overdueAccounts.reduce((sum, a) => sum + a.emi_amount, 0);
  const openCases = state.inbox.length;
  renderPortfolioSummary(accounts, openCases);
  const rate = state.metrics ? Math.round(state.metrics.escalation_rate * 100) : null;
  const toolCalls = state.metrics ? state.metrics.event_counts.tool_called || 0 : 0; // the rate's denominator

  const riskCounts = { low: 0, medium: 0, high: 0, critical: 0 };
  accounts.forEach((a) => {
    if (riskCounts[a.risk_tier] !== undefined) riskCounts[a.risk_tier]++;
  });

  const cards = [
    {
      icon: iconPortfolio(),
      cls: "",
      label: "Portfolio",
      value: accounts.length,
      sub: `${accounts.length === 1 ? "account" : "accounts"} on the books`,
      action: "all",
    },
    {
      icon: iconClock(),
      cls: overdueCount ? "danger" : "ok",
      label: "Overdue",
      value: overdueCount,
      sub: overdueCount ? `${fmtInr(overdueRupees)} at risk` : "nothing overdue",
      action: "overdue",
    },
    {
      icon: iconInbox(),
      cls: openCases ? "warn" : "ok",
      label: "Awaiting a human",
      value: openCases,
      sub: openCases ? "open cases in the inbox" : "inbox is clear",
      action: "inbox",
    },
    {
      icon: iconPulse(),
      cls: "",
      label: "Escalation rate",
      labelHtml: `Escalation rate <select id="since-hours" class="kpi-window" aria-label="Time window for the escalation rate">${Object.entries(SINCE_HOURS_LABELS)
        .map(([hours, short]) => `<option value="${hours}" ${Number(hours) === state.sinceHours ? "selected" : ""}>${short}</option>`)
        .join("")}</select>`,
      value: rate === null ? "–" : `${rate}%`,
      sub: !state.metrics
        ? state.metricsFailed ? "couldn't load; retrying on the next refresh" : "loading…"
        : toolCalls
          ? `of ${toolCalls} agent tool call${toolCalls === 1 ? "" : "s"} handed to a person`
          : "no agent activity in this window",
      action: null,
    },
    {
      icon: iconRiskMix(),
      cls: "",
      label: "Risk mix",
      // A custom valueHtml (rather than the plain numeric `value` every
      // other card uses) since this is three counts, not one -- inline
      // colored spans read faster than three separate cards would for
      // something this glanceable.
      valueHtml: `<span class="kpi-risk-mix"><span class="risk-low">${riskCounts.low}L</span><span class="risk-medium">${riskCounts.medium}M</span><span class="risk-high">${riskCounts.high}H</span>${riskCounts.critical ? `<span class="risk-critical">${riskCounts.critical}C</span>` : ""}</span>`,
      sub: riskCounts.critical ? "low · medium · high · critical" : "low · medium · high risk",
      action: null,
    },
  ];

  // Found live: the KPI row was the literal front door of this dashboard
  // and had zero event listeners anywhere near it -- an operator seeing
  // "8 overdue" had to separately notice the identical count on a filter
  // pill further down and click that instead. Every card with a real
  // destination (a worklist to jump to) is now clickable; "Escalation
  // rate" stays a plain figure -- it's a rate, not a queue, so there's
  // nothing real to jump to.
  document.getElementById("kpi-row").innerHTML = cards
    .map(
      (c) => `
    <div class="kpi-card ${c.cls === "danger" || c.cls === "warn" ? c.cls : ""} ${c.action ? "clickable" : ""}" ${c.action ? `data-kpi-action="${c.action}" role="button" tabindex="0"` : ""}>
      <div class="kpi-top">
        <span class="kpi-label">${c.labelHtml ?? escapeHtml(c.label)}</span>
        <span class="kpi-icon ${c.cls}">${c.icon}</span>
      </div>
      <div class="kpi-value">${c.valueHtml ?? c.value}</div>
      <div class="kpi-sub">${escapeHtml(c.sub)}</div>
    </div>`
    )
    .join("");

  document.querySelectorAll("#kpi-row [data-kpi-action]").forEach((card) => {
    const activate = () => {
      const action = card.dataset.kpiAction;
      if (action === "inbox") {
        setView("inbox");
        return;
      }
      setView("overview");
      state.activeFilter = action; // "all" or "overdue"
      renderFilterPills();
      renderAccountGrid();
      document.getElementById("account-grid").scrollIntoView({ behavior: "smooth", block: "start" });
    };
    card.addEventListener("click", activate);
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        activate();
      }
    });
  });

  const badge = document.getElementById("inbox-count-badge");
  if (openCases > 0) {
    badge.hidden = false;
    badge.textContent = openCases;
  } else {
    badge.hidden = true;
  }
}

const iconPortfolio = () =>
  `<svg width="15" height="15" viewBox="0 0 24 24" fill="none"><rect x="3" y="7" width="18" height="13" rx="2" stroke="currentColor" stroke-width="2"/><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke="currentColor" stroke-width="2"/></svg>`;
const iconClock = () =>
  `<svg width="15" height="15" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/><path d="M12 7v5l3.5 2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`;
const iconInbox = () =>
  `<svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M3 12h5l2 3h4l2-3h5" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M5.5 5h13l2.5 7v7a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-7l2.5-7Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>`;
const iconPulse = () =>
  `<svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M3 12h4l2 7 4-14 2 7h6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const iconRiskMix = () =>
  `<svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M12 2 2 7l10 5 10-5-10-5Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>`;

/* ============================================================
   Filter pills
   ============================================================ */
function renderFilterPills() {
  const counts = { overdue: 0, disputed: 0, broken_promises: 0 };
  state.accounts.forEach((a) => a.flags.forEach((f) => (counts[f.label] = (counts[f.label] || 0) + 1)));

  const pills = [
    { key: "all", label: "All accounts", count: state.accounts.length },
    { key: "overdue", label: "Overdue", count: counts.overdue },
    { key: "disputed", label: "Disputed", count: counts.disputed },
    { key: "broken_promises", label: "Broken promises", count: counts.broken_promises },
  ];
  // Reached only by asking ("who isn't on Telegram?"), so its pill shows only while it is the filter in use.
  if (state.activeFilter === "not_on_telegram") {
    pills.push({ key: "not_on_telegram", label: "Not on Telegram", count: state.accounts.filter((a) => !a.telegram_linked).length });
  }

  document.getElementById("filter-pills").innerHTML = pills
    .map(
      (p) => `
    <button class="pill ${state.activeFilter === p.key ? "active" : ""}" data-filter="${p.key}">
      ${p.key !== "all" ? `<span class="pill-dot ${p.key}"></span>` : ""}
      ${escapeHtml(p.label)} <span class="count">${p.count}</span>
    </button>`
    )
    .join("");

  document.querySelectorAll("#filter-pills .pill").forEach((btn) =>
    btn.addEventListener("click", () => {
      state.activeFilter = btn.dataset.filter;
      renderFilterPills();
      renderAccountGrid();
    })
  );
}

/* ============================================================
   Account grid
   ============================================================ */
// Found live: even the Overdue filter showed accounts in arbitrary
// account_id order, so an operator still had to eyeball every card to
// find the worst one -- this is the actual "who do I call next" answer.
// Defaults to urgency (not account_id) since that's the point of a
// worklist, not a buried opt-in nobody would discover.
// Why the account has the risk it has: each factor and its points (outbound/risk_engine.py), so the tier is never a
// black box. Collapsed by default.
function riskWhyHtml(a) {
  if (!a.risk) return "";
  const rows = a.risk.factors
    .map((f) => `<li><span>${escapeHtml(f.label)}</span><b>${f.points > 0 ? "+" : ""}${Math.round(f.points)}</b></li>`)
    .join("");
  return `<details class="risk-why"><summary>Why ${escapeHtml(a.risk.tier)} risk? Score ${Math.round(a.risk.score)} of 100</summary>
    <ul>${rows}</ul>
    <p class="risk-why-note">25+ is medium, 50+ high, 75+ critical (or 2+ broken promises while past the grace period). The same score sets the tone of reminders.</p>
  </details>`;
}

/* ============================================================
   Collection health -- GET /portfolio/health (accounts/health.py)
   ============================================================ */
function pct(v) {
  return v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`;
}

async function loadHealth() {
  try {
    renderHealth(await api("/portfolio/health"));
  } catch (err) {
    console.warn("ops: collection health could not be loaded; the next refresh retries it", err);
  }
}

function renderHealth(h) {
  const total = h.dpd_buckets.reduce((s, b) => s + b.accounts, 0) || 1;
  const bar = h.dpd_buckets
    .map((b) => b.accounts ? `<span class="dpd-seg dpd-${b.key}" style="flex:${b.accounts}" title="${escapeHtml(b.label)}: ${b.accounts} account${b.accounts === 1 ? "" : "s"}, ${fmtInr(b.owed_now)} owed now">${b.accounts}</span>` : "")
    .join("");
  const legend = h.dpd_buckets
    .map((b) => `<li><span class="dpd-key dpd-${b.key}"></span>${escapeHtml(b.label)} <b>${b.accounts}</b> <span class="muted">${fmtInr(b.owed_now)} owed</span></li>`)
    .join("");
  const tile = (label, m, sub) => `<div class="health-tile" title="${escapeHtml(m.how)}">
      <div class="health-label">${escapeHtml(label)}</div>
      <div class="health-value">${pct(m.value)}</div>
      <div class="health-sub">${escapeHtml(sub)}</div>
    </div>`;
  const ce = h.collection_efficiency, pk = h.promises_kept, cr = h.cure_rate, ct = h.containment;
  document.getElementById("health-row").innerHTML = `
    <div class="health-buckets">
      <div class="health-label">Days past due <span class="muted">· ${total} live account${total === 1 ? "" : "s"}</span></div>
      <div class="dpd-bar">${bar}</div>
      <ul class="dpd-legend">${legend}</ul>
    </div>
    <div class="health-tiles">
      ${tile("Collection efficiency", ce, `${fmtInr(ce.collected)} in, ${fmtInr(ce.still_owed)} past due (30 days)`)}
      ${tile("Promises kept", pk, pk.kept + pk.broken ? `${pk.kept} kept, ${pk.broken} broken (90 days)` : "no promise settled in 90 days")}
      ${tile("Cure rate", cr, cr.went_late ? `${cr.cured} of ${cr.went_late} late accounts now current (90 days)` : "no account went late in 90 days")}
      ${tile("Containment", ct, ct.turns ? `${ct.turns - ct.handed_to_person} of ${ct.turns} assistant turns without a person (30 days)` : "no assistant activity in 30 days")}
    </div>`;
  document.getElementById("health-row").hidden = false;
}

function _accountSortComparator() {
  if (state.sortBy === "risk") {
    // The one computed risk (outbound/risk_engine.py): highest score first, so two "high" accounts are ordered too.
    return (a, b) => ((b.risk?.score ?? 0) - (a.risk?.score ?? 0)) || (b.days_past_due - a.days_past_due);
  }
  if (state.sortBy === "id") {
    return (a, b) => a.account_id.localeCompare(b.account_id);
  }
  return (a, b) => {
    const aOverdue = a.flags.some((f) => f.label === "overdue");
    const bOverdue = b.flags.some((f) => f.label === "overdue");
    if (aOverdue !== bOverdue) return aOverdue ? -1 : 1;
    if (aOverdue && bOverdue) return b.days_past_due - a.days_past_due;
    return a.account_id.localeCompare(b.account_id);
  };
}

function filteredAccounts() {
  let list = state.accounts;
  if (state.activeFilter === "not_on_telegram") {
    list = list.filter((a) => !a.telegram_linked);
  } else if (state.activeFilter !== "all") {
    list = list.filter((a) => a.flags.some((f) => f.label === state.activeFilter));
  }
  if (state.search.trim()) {
    const q = state.search.trim().toLowerCase();
    list = list.filter(
      (a) =>
        a.borrower_name.toLowerCase().includes(q) ||
        a.business_name.toLowerCase().includes(q) ||
        a.account_id.toLowerCase().includes(q)
    );
  }
  // .slice() first -- .filter() always returns a fresh array, but when
  // activeFilter is "all" and search is empty, list is still the SAME
  // array as state.accounts, and .sort() mutates in place. Sorting that
  // reference directly would silently reorder state.accounts itself.
  return list.slice().sort(_accountSortComparator());
}

document.getElementById("sort-select").addEventListener("change", (e) => {
  state.sortBy = e.target.value;
  renderAccountGrid();
});

// Turns what used to be a developer-only terminal command
// (scripts/run_outbound_pass.py) into something ops can actually press --
// scoped to a specific selection when one exists, otherwise to whatever
// the current filter/search shows, so "send reminders to just these 3" or
// "to everyone currently overdue" are both one click. A native confirm(),
// not just a disabled-state guard, since this sends real messages to real
// borrowers -- one of the few actions in this dashboard where a stray
// click has an external, unrecallable effect.
document.getElementById("send-reminders-btn").addEventListener("click", async () => {
  const targets = state.selectedAccountIds.size
    ? [...state.selectedAccountIds]
    : filteredAccounts().map((a) => a.account_id);
  if (!targets.length) {
    toast("No accounts match the current filter.", true);
    return;
  }
  const scope = state.selectedAccountIds.size ? "selected" : "matching the current filter";
  if (!confirm(`Send today's reminders to ${targets.length} account(s) ${scope}?`)) return;

  const btn = document.getElementById("send-reminders-btn");
  btn.disabled = true;
  btn.textContent = "Sending…";
  try {
    const result = await triggerOutboundRun(targets);
    // Found live: this used to report only a raw count, with no way to
    // tell "sent and actually reached someone" apart from "logged, but
    // nobody was ever going to see it" (no Telegram chat linked to that
    // account). In a real environment where most demo accounts have
    // never verified over Telegram, that made every real click read as
    // "0 reminder(s) sent" or a silent no-op -- the button was working
    // correctly the whole time, the feedback just didn't say so.
    const total = result.reminders_sent.length;
    const delivered = result.reminders_sent.filter((r) => r.delivered_via_telegram).length;
    const loggedOnly = total - delivered;
    let message;
    if (total === 0) {
      message = "No accounts are currently due for a reminder.";
    } else if (loggedOnly === 0) {
      message = `${total} reminder(s) sent and delivered via Telegram.`;
    } else {
      message = `${total} reminder(s) processed -- ${delivered} delivered via Telegram, ${loggedOnly} logged only (no Telegram linked).`;
    }
    if (result.escalated_for_broken_promises) message += ` ${result.escalated_for_broken_promises} escalated.`;
    toast(message);
    await loadAll();
  } catch (err) {
    toast(err.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Send reminders now";
  }
});

document.getElementById("selection-clear-btn").addEventListener("click", () => {
  state.selectedAccountIds.clear();
  renderAccountGrid();
});

/* ============================================================
   Bulk clarification -- one operator-written message, sent verbatim to
   every currently-selected account (see send_bulk_clarification's own
   docstring for why this deliberately isn't an AI draft per account).
   ============================================================ */
const $bulkClarifyBackdrop = document.getElementById("bulk-clarify-backdrop");
const $bulkClarifyModal = document.getElementById("bulk-clarify-modal");

function openBulkClarifyModal() {
  const ids = [...state.selectedAccountIds];
  document.getElementById("bulk-clarify-count").textContent = ids.length;
  const accountsById = new Map(state.accounts.map((a) => [a.account_id, a]));
  document.getElementById("bulk-clarify-recipients").innerHTML = ids
    .map((id) => `<span class="badge">${escapeHtml(accountsById.get(id)?.borrower_name || id)}</span>`)
    .join("");
  document.getElementById("bulk-clarify-message").value = "";
  document.getElementById("bulk-clarify-error").hidden = true;
  document.getElementById("bulk-clarify-result").hidden = true;
  $bulkClarifyModal.hidden = false;
  $bulkClarifyBackdrop.hidden = false;
  setTimeout(() => {
    $bulkClarifyModal.classList.add("open");
    $bulkClarifyBackdrop.classList.add("open");
  }, 10);
}

function closeBulkClarifyModal() {
  $bulkClarifyModal.classList.remove("open");
  $bulkClarifyBackdrop.classList.remove("open");
  setTimeout(() => {
    $bulkClarifyModal.hidden = true;
    $bulkClarifyBackdrop.hidden = true;
  }, 300);
}

document.getElementById("selection-clarify-btn").addEventListener("click", openBulkClarifyModal);
document.getElementById("bulk-clarify-close").addEventListener("click", closeBulkClarifyModal);
document.getElementById("bulk-clarify-cancel-btn").addEventListener("click", closeBulkClarifyModal);
$bulkClarifyBackdrop.addEventListener("click", closeBulkClarifyModal);

document.getElementById("bulk-clarify-send-btn").addEventListener("click", async () => {
  const errorEl = document.getElementById("bulk-clarify-error");
  const resultEl = document.getElementById("bulk-clarify-result");
  errorEl.hidden = true;
  resultEl.hidden = true;

  const message = document.getElementById("bulk-clarify-message").value.trim();
  const ids = [...state.selectedAccountIds];
  if (!message) {
    errorEl.textContent = "Write a message first.";
    errorEl.hidden = false;
    return;
  }
  if (!ids.length) {
    errorEl.textContent = "Nothing selected.";
    errorEl.hidden = false;
    return;
  }

  const btn = document.getElementById("bulk-clarify-send-btn");
  btn.disabled = true;
  btn.textContent = "Sending…";
  try {
    const result = await sendBulkClarification(ids, message);
    resultEl.textContent = `Sent to ${result.sent_to.length} account(s), ${result.delivered_via_telegram.length} delivered over Telegram${result.not_found.length ? `, ${result.not_found.length} not found` : ""}.`;
    resultEl.hidden = false;
    toast(`Clarification sent to ${result.sent_to.length} account(s).`);
    state.selectedAccountIds.clear();
    renderAccountGrid();
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Send to all selected";
  }
});

function renderAccountGrid() {
  const list = filteredAccounts();
  const grid = document.getElementById("account-grid");
  document.getElementById("account-empty").hidden = list.length !== 0;
  grid.innerHTML = list.map(accountCardHtml).join("");
  grid.querySelectorAll(".account-card").forEach((card) =>
    card.addEventListener("click", () => openDetail(card.dataset.id))
  );
  grid.querySelectorAll(".account-select-checkbox").forEach((checkbox) =>
    checkbox.addEventListener("click", (e) => e.stopPropagation()) // don't also open the card
  );
  grid.querySelectorAll(".account-select-checkbox").forEach((checkbox) =>
    checkbox.addEventListener("change", (e) => {
      const id = e.target.dataset.id;
      if (e.target.checked) state.selectedAccountIds.add(id);
      else state.selectedAccountIds.delete(id);
      e.target.closest(".account-card-wrap").querySelector(".account-card").classList.toggle("selected", e.target.checked);
      renderSelectionBar();
    })
  );
  renderSelectionBar();
}

// Found live, per the product owner's own framing: staff could only ever
// act on one account at a time (send a reminder, send a clarification) --
// this is the "who's it for" answer, a real selection an operator builds
// up across the filtered/sorted grid, not a hidden power-user feature.
function renderSelectionBar() {
  const bar = document.getElementById("selection-bar");
  const count = state.selectedAccountIds.size;
  bar.hidden = count === 0;
  if (count > 0) document.getElementById("selection-count").textContent = count;
}

function accountCardHtml(a) {
  const flagsHtml = a.flags.length
    ? a.flags.map((f) => `<span class="flag-chip ${f.label}" title="${escapeHtml(f.reason)}">${escapeHtml(FLAG_LABELS[f.label] || f.label)}</span>`).join("")
    : `<span class="flag-chip clean">Clean</span>`;
  const isSelected = state.selectedAccountIds.has(a.account_id);
  // Late but not yet flagged overdue means still inside the grace period: amber, not red.
  const dpdClass = a.days_past_due <= 0 ? "" : a.flags.some((f) => f.label === "overdue") ? "overdue" : "grace";

  return `
  <div class="account-card-wrap">
  <input type="checkbox" class="account-select-checkbox" data-id="${a.account_id}" ${isSelected ? "checked" : ""} aria-label="Select ${escapeHtml(a.borrower_name)}" />
  <button class="account-card ${isSelected ? "selected" : ""}" data-id="${a.account_id}">
    <div class="account-top">
      <div class="avatar" style="${avatarStyle(a.account_id)}">${initials(a.borrower_name)}</div>
      <div class="account-names">
        <div class="account-borrower">${escapeHtml(a.borrower_name)}</div>
        <div class="account-business">${escapeHtml(a.business_name)}</div>
      </div>
    </div>
    <div class="account-figures">
      <div class="account-emi">
        <div class="label">EMI</div>
        <div class="amount">${fmtInr(a.emi_amount)}</div>
      </div>
      <div class="account-dpd">
        <div class="label">Status</div>
        <div class="dpd ${dpdClass}"${dpdClass === "grace" ? ` title="Within the ${GRACE_PERIOD_DAYS}-day grace period"` : ""}>${a.days_past_due > 0 ? `${a.days_past_due}d past due` : "Current"}</div>
      </div>
    </div>
    <div class="account-flags">${flagsHtml}</div>
    <div class="account-footer">
      <span class="account-id">${a.account_id} · ${escapeHtml(a.loan_type)}</span>
      <span class="risk-tag ${a.risk_tier}"><span class="risk-dot ${a.risk_tier}"></span>${a.risk_tier[0].toUpperCase()}${a.risk_tier.slice(1)} risk</span>
    </div>
  </button>
  </div>`;
}

document.getElementById("search-input").addEventListener("input", (e) => {
  state.search = e.target.value;
  renderAccountGrid();
});

/* ============================================================
   Inbox, and the case card it shares with the profile's Cases tab
   ============================================================ */
// One list of everything open that needs a person -- payment claims, disputes,
// requests, system notices -- where there used to be an escalation queue that
// never showed a dispute at all. Each card is one case (accounts/cases.py): what
// the borrower said, the messages since, and the one action that closes it.
//
// An escalation open 20 minutes and one open 5 days are very different
// problems, so open cases are also bucketed by age.
function renderSlaBuckets() {
  const open = state.inbox;
  const container = document.getElementById("sla-buckets");
  if (!open.length) {
    container.innerHTML = "";
    return;
  }
  const now = Date.now();
  const buckets = { fresh: 0, aging: 0, stale: 0 };
  open.forEach((c) => {
    const hours = (now - new Date(c.opened_at).getTime()) / 3_600_000;
    if (hours < 24) buckets.fresh++;
    else if (hours < 72) buckets.aging++;
    else buckets.stale++;
  });
  container.innerHTML = `
    <span class="sla-bucket"><span class="count">${buckets.fresh}</span> under 24h</span>
    <span class="sla-bucket"><span class="count">${buckets.aging}</span> 24–72h</span>
    <span class="sla-bucket ${buckets.stale ? "breach" : ""}"><span class="count">${buckets.stale}</span> over 72h${buckets.stale ? " — stale" : ""}</span>
  `;
}

function renderInbox() {
  renderSlaBuckets();
  // The server already sorts (most urgent kind, then oldest); kept as sent.
  const container = document.getElementById("inbox-list");
  document.getElementById("inbox-empty").hidden = state.inbox.length !== 0;
  container.innerHTML = state.inbox.map((c) => caseCardHtml(c, { inInbox: true })).join("");
}

function proposedTermsHtml(pc) {
  if (!pc) return "";
  if (pc.type === "extend_tenure") {
    return `<div class="escalation-terms">
      <span>Extend by <b>${pc.extra_months}mo</b></span>
      <span>New term <b>${pc.new_months_remaining}mo</b></span>
      <span>New EMI <b>${fmtInr(pc.new_emi_amount)}</b></span>
    </div>`;
  }
  const rows = Object.entries(pc)
    .filter(([k]) => k !== "type")
    .map(([k, v]) => `<span>${escapeHtml(k)} <b>${escapeHtml(String(v))}</b></span>`)
    .join("");
  return `<div class="escalation-terms">${rows}</div>`;
}

function caseMessageHtml(m) {
  const fromBorrower = m.author === "borrower";
  const delivery = fromBorrower ? "" : m.delivered_via_telegram ? " · on Telegram" : " · dashboard only";
  return `<div class="case-msg ${fromBorrower ? "borrower" : "ops"}">
    <span class="case-msg-who">${fromBorrower ? "Borrower" : "Team"}</span>
    <span class="case-msg-text">${escapeHtml(m.message)}</span>
    <span class="case-msg-meta">${relativeTime(m.created_at)}${delivery}</span>
  </div>`;
}

// The inbox shows the last two messages of each case; the profile shows all of them.
function caseThreadHtml(c, { compact }) {
  const msgs = c.thread || [];
  if (!msgs.length) return "";
  const shown = compact ? msgs.slice(-2) : msgs;
  const earlier = msgs.length - shown.length;
  return `<div class="case-thread">${earlier ? `<p class="case-earlier">${earlier} earlier message${earlier === 1 ? "" : "s"} — open the case to read them</p>` : ""}${shown.map(caseMessageHtml).join("")}</div>`;
}

// The one right action per kind, plus Reply. Only a restructuring or a closure
// certificate is approved or rejected; a payment claim is confirmed (the payment
// recorded) or closed with a reason; a dispute is resolved with a note.
function caseActionsHtml(c) {
  if (!c.open) return "";
  const b = (action, label, cls = "btn-secondary") => `<button type="button" class="btn ${cls}" data-case-action="${action}">${label}</button>`;
  const actions = {
    payment_claim: b("confirm", "Confirm payment…", "btn-success") + b("resolve", "Close without payment…"),
    dispute: b("resolve", "Resolve…", "btn-success"),
    restructuring: b("approve", "Approve", "btn-success") + b("reject", "Reject…", "btn-danger"),
    closure: b("approve", "Approve", "btn-success") + b("reject", "Reject…", "btn-danger"),
    system: b("approve", "Acknowledge", "btn-success"),
  };
  return `<div class="case-actions">${actions[c.kind] ?? b("approve", "Mark handled", "btn-success")}${c.can_reply ? b("reply", "Reply…") : ""}</div>`;
}

function caseFormsHtml(c, amountHint) {
  const forms = [];
  if (c.can_reply) {
    forms.push(`<form class="case-form" data-case-form="reply" hidden>
      <textarea rows="3" maxlength="1000" placeholder="Your reply — shown in this case on the borrower's dashboard, and sent on Telegram if linked"></textarea>
      <div class="clarify-actions-row">
        <button type="submit" class="btn btn-primary">Send reply</button>
        ${c.kind === "payment_claim" ? `<button type="button" class="btn btn-ghost" data-case-action="fill-reference">Ask for the payment reference</button>` : ""}
      </div>
      <p class="mini-form-error" hidden></p>
    </form>`);
  }
  if (c.kind === "payment_claim") {
    forms.push(`<form class="case-form" data-case-form="confirm" hidden>
      <p class="inline-form-title">The payment you found in the bank record</p>
      <div class="upload-row">
        <input type="number" name="amount" min="0.01" step="0.01" required value="${amountHint ?? ""}" aria-label="Amount paid (₹)" placeholder="Amount (₹)" />
        <input type="date" name="payment_date" aria-label="Date the borrower paid" title="Date the borrower paid (decides on time or late; defaults to today)" />
      </div>
      <input type="text" name="note" class="case-note-input" maxlength="500" placeholder="Note to the borrower (optional)" />
      <label class="case-duplicate-ok" hidden><input type="checkbox" name="record_even_if_duplicate" /> It's a second, separate payment: record it anyway</label>
      <div class="mini-form-radio-row case-extra-question" hidden>
        <span class="clarify-hint full-row">Less than what's due this cycle — apply it toward the next EMI?</span>
        <label><input type="radio" name="apply" value="true" /> Yes, credit it</label>
        <label><input type="radio" name="apply" value="false" /> No, just record it</label>
      </div>
      <div class="clarify-actions-row">
        <button type="submit" class="btn btn-success">Record payment &amp; close</button>
        <span class="clarify-hint">Records it on the account, closes the claim and tells the borrower.</span>
      </div>
      <p class="mini-form-error" hidden></p>
    </form>`);
  }
  if (c.source === "dispute") {
    const claim = c.kind === "payment_claim";
    forms.push(`<form class="case-form" data-case-form="resolve" hidden>
      <input type="text" name="note" class="case-note-input" maxlength="500" ${claim ? "required" : ""}
        placeholder="${claim ? "Why it couldn't be confirmed — sent to the borrower" : "How it was resolved (optional) — sent to the borrower"}" />
      <div class="clarify-actions-row">
        <button type="submit" class="btn ${claim ? "btn-secondary" : "btn-success"}">${claim ? "Close without payment" : "Resolve"}</button>
        <span class="clarify-hint">Reminders resume once it's closed.</span>
      </div>
      <p class="mini-form-error" hidden></p>
    </form>`);
  }
  if (c.kind === "restructuring" || c.kind === "closure") {
    forms.push(`<form class="case-form" data-case-form="reject" hidden>
      <input type="text" name="reason" class="case-note-input" maxlength="500" placeholder="Reason shown to the borrower (optional)" />
      <div class="clarify-actions-row"><button type="submit" class="btn btn-danger">Reject</button></div>
      <p class="mini-form-error" hidden></p>
    </form>`);
  }
  return forms.join("");
}

// A payment already recorded on the account in the last few days, shown on an open payment claim. Found live: a
// borrower paid through the link, the agent filed a claim for that same payment, and confirming it recorded it twice.
function recentPaymentsHtml(c) {
  const recent = c.recent_payments || [];
  if (!c.open || c.kind !== "payment_claim" || !recent.length) return "";
  const items = recent
    .map((p) => `<li>${fmtInr(p.amount)} paid ${fmtShortDate(p.payment_date)}, recorded ${relativeTime(p.recorded_at)}</li>`)
    .join("");
  return `<div class="case-notice"><b>Already on record, paid in the last 7 days:</b><ul>${items}</ul>If one of these is the payment they mean, close the claim instead of confirming.</div>`;
}

function caseCardHtml(c, { inInbox = false, amountHint = null } = {}) {
  const kind = c.kind || "request";
  const acct = state.accounts.find((a) => a.account_id === c.account_id);
  // Who and how much, right on an inbox card -- triage shouldn't need every account opened.
  const who = inInbox && acct ? `${fmtInr(acct.emi_amount)} EMI${acct.days_past_due > 0 ? ` · ${acct.days_past_due}d past due` : ""}` : "";
  const prefill = amountHint ?? acct?.amount_due_now ?? acct?.emi_amount ?? null;
  const wordsLabel = c.source === "dispute" ? "Borrower said" : kind === "system" ? "Raised by the system" : "Reason";
  const outcome = !c.open && c.outcome && !(c.thread || []).some((m) => m.author === "ops")
    ? `<div class="case-outcome">Outcome: ${escapeHtml(c.outcome)}</div>`
    : "";
  return `
  <article class="case-card${kind === "fraud" ? " urgent" : ""}${c.open ? "" : " closed"}" data-case-ref="${escapeHtml(c.case_ref)}" data-account-id="${escapeHtml(c.account_id)}" data-kind="${escapeHtml(kind)}">
    <div class="case-top">
      <span class="kind-pill ${escapeHtml(kind)}">${escapeHtml(KIND_LABELS[kind] || kind)}</span>
      ${caseHeading(c) ? `<span class="case-title">${escapeHtml(caseHeading(c))}</span>` : ""}
      ${inInbox ? `<button type="button" class="case-account" data-open-account="${escapeHtml(c.account_id)}" title="Open this account's cases">${escapeHtml(c.borrower_name || c.account_id)} · ${escapeHtml(c.account_id)}</button>` : ""}
      <span class="case-age">${c.open ? `opened ${relativeTime(c.opened_at)}` : `${escapeHtml(c.status)} ${relativeTime(c.closed_at || c.opened_at)}`}</span>
    </div>
    ${who ? `<div class="escalation-who">${escapeHtml(who)}</div>` : ""}
    <div class="case-words"><span class="case-words-label">${wordsLabel}</span>${escapeHtml(c.borrower_words)}</div>
    ${proposedTermsHtml(c.proposed_changes)}
    ${recentPaymentsHtml(c)}
    ${caseThreadHtml(c, { compact: inInbox })}
    ${outcome}
    ${caseActionsHtml(c)}
    ${c.open ? caseFormsHtml(c, prefill) : ""}
  </article>`;
}

// What the operator is told happened to the BORROWER's side of an approve /
// reject: the toast used to say "borrower notified" every time, including when
// nothing was sent at all and when the borrower has no linked Telegram.
function decisionToast(verb, result) {
  if (result.kind === "system") return `${verb} \u2014 internal notice, nothing sent to the borrower.`;
  if (result.borrower_notified === true) return `${verb} \u2014 borrower notified on Telegram.`;
  if (result.borrower_notified === false) return `${verb} \u2014 no linked Telegram, so it's shown in this case on the borrower's dashboard.`;
  return `${verb}.`;
}

function notifiedText(notified) {
  return notified ? "borrower notified on Telegram" : "shown in this case on the borrower's dashboard (no linked Telegram)";
}

// After any action on a case: the profile (if this case is in it) and the
// whole dashboard reload -- confirming a payment changes the account's EMI and
// flags, and every action changes the inbox.
async function afterCaseAction(root, accountId) {
  if (root === $detailPanel && state.detailAccountId === accountId) {
    const [fresh, freshDocuments, freshConversation] = await _loadDetailData(accountId);
    renderDetail(fresh, freshDocuments, freshConversation);
  }
  await loadAll();
}

function showCaseForm(card, name) {
  card.querySelectorAll("[data-case-form]").forEach((f) => {
    f.hidden = f.dataset.caseForm === name ? !f.hidden : true;
  });
  card.querySelector(`[data-case-form="${name}"]:not([hidden]) textarea, [data-case-form="${name}"]:not([hidden]) input`)?.focus();
}

async function submitCaseForm(form, card) {
  const { caseRef, accountId, kind } = card.dataset;
  const field = (name) => form.querySelector(`[name="${name}"]`)?.value.trim() || "";
  switch (form.dataset.caseForm) {
    case "reply": {
      const message = form.querySelector("textarea").value.trim();
      if (!message) throw new Error("Write a reply first.");
      const result = await replyToCase(accountId, caseRef, message);
      return `Reply sent — ${notifiedText(result.delivered_via_telegram)}.`;
    }
    case "confirm": {
      const amount = Number(field("amount"));
      if (!(amount > 0)) throw new Error("Enter the amount you found.");
      const payload = { amount, payment_date: field("payment_date") || null, note: field("note") || null };
      if (form.querySelector('[name="record_even_if_duplicate"]')?.checked) payload.record_even_if_duplicate = true;
      const extra = form.querySelector(".case-extra-question");
      if (!extra.hidden) {
        const picked = form.querySelector('input[name="apply"]:checked');
        if (!picked) throw new Error("Choose whether to credit it toward the next EMI.");
        payload.apply_extra_to_next = picked.value === "true";
      }
      const result = await confirmPaymentClaim(accountId, caseRef, payload);
      return `${fmtInr(result.payment.amount)} recorded and the claim closed — ${notifiedText(result.borrower_notified)}.`;
    }
    case "resolve": {
      const note = field("note");
      if (kind === "payment_claim" && !note) throw new Error("Say why it couldn't be confirmed — the borrower is sent this.");
      const result = await resolveDispute(accountId, note, Number(caseRef.replace(/^D-/, "")));
      return `${kind === "payment_claim" ? "Claim closed" : "Dispute resolved"} — ${notifiedText(result.borrower_notified)}.`;
    }
    case "reject": {
      const result = await rejectEscalation(caseRef, field("reason"));
      return decisionToast("Rejected", result);
    }
  }
  throw new Error(`unknown case form ${form.dataset.caseForm}`);
}

// Delegated once per container (the inbox list, the detail panel): both are
// re-rendered on every load, so handlers on the cards themselves would be lost.
function wireCaseActions(root) {
  root.addEventListener("click", async (e) => {
    const opener = e.target.closest("[data-open-account]");
    if (opener) {
      openDetail(opener.dataset.openAccount, { tab: "cases" });
      return;
    }
    const btn = e.target.closest("[data-case-action]");
    const card = btn?.closest(".case-card");
    if (!card) return;
    const action = btn.dataset.caseAction;
    if (action === "fill-reference") {
      const box = card.querySelector('[data-case-form="reply"] textarea');
      box.value = ASK_REFERENCE_TEXT;
      box.focus();
      return;
    }
    if (action !== "approve") {
      showCaseForm(card, action);
      return;
    }
    btn.disabled = true;
    try {
      const result = await approveEscalation(card.dataset.caseRef);
      const verb = result.kind === "restructuring" || result.kind === "closure" ? "Approved" : result.kind === "system" ? "Acknowledged" : "Closed";
      toast(decisionToast(verb, result));
      await afterCaseAction(root, card.dataset.accountId);
    } catch (err) {
      toast(err.message, true);
      btn.disabled = false;
    }
  });

  root.addEventListener("submit", async (e) => {
    const form = e.target.closest("[data-case-form]");
    const card = form?.closest(".case-card");
    if (!card) return;
    e.preventDefault();
    const submit = form.querySelector("button[type=submit]");
    const errorEl = form.querySelector(".mini-form-error");
    errorEl.hidden = true;
    submit.disabled = true;
    try {
      toast(await submitCaseForm(form, card));
      await afterCaseAction(root, card.dataset.accountId);
    } catch (err) {
      // A 422 on confirm means the amount is short of this cycle's EMI: ask, as Record payment does.
      if (form.dataset.caseForm === "confirm" && err instanceof ApiError && err.status === 422) {
        form.querySelector(".case-extra-question").hidden = false;
        errorEl.textContent = "That's less than what's due this cycle — choose whether to credit it toward the next EMI.";
      } else if (form.dataset.caseForm === "confirm" && err instanceof ApiError && err.data?.code === "possible_duplicate") {
        // Nothing was recorded. Say what's already on record, and only then offer to record a second payment.
        form.querySelector(".case-duplicate-ok").hidden = false;
        errorEl.textContent = err.message;
      } else {
        errorEl.textContent = err.message;
      }
      errorEl.hidden = false;
      submit.disabled = false;
    }
  });
}

/* ============================================================
   Ring chart (SVG)
   ============================================================ */
function ringChartSvg(fraction, color) {
  const size = 96;
  const stroke = 10;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(1, fraction));
  const offset = c * (1 - clamped);
  return `
  <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" style="transform:rotate(-90deg)">
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="var(--border)" stroke-width="${stroke}" />
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="${color}" stroke-width="${stroke}"
      stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${offset}" />
  </svg>`;
}

/* ============================================================
   Detail slide-over
   ============================================================ */
const $detailPanel = document.getElementById("detail-panel");
const $detailBackdrop = document.getElementById("detail-backdrop");

// Shared by the initial open and every action's post-mutation refresh
// (escalation approve/reject, the clarify-form send, record-payment/
// resolve-dispute/log-promise/call-log) -- one place fetching the three
// calls a full detail re-render needs, instead of the same Promise.all
// repeated at every call site.
function _loadDetailData(accountId) {
  return Promise.all([getAccount(accountId), getDocuments(accountId), getConversation(accountId)]);
}

const DETAIL_CLOSE_HTML = `<button type="button" class="detail-close" aria-label="Close">✕</button>`;

async function openDetail(accountId, { tab = "overview" } = {}) {
  state.detailAccountId = accountId;
  state.detailTab = tab; // a freshly opened account starts on its overview, or on Cases when opened from the inbox
  $detailPanel.hidden = false;
  $detailBackdrop.hidden = false;
  setTimeout(() => {
    $detailPanel.classList.add("open");
    $detailBackdrop.classList.add("open");
  }, 10);
  // The name, business and avatar are already known from the portfolio list, so they show at once; only the rest
  // waits for the account's own data (a few round trips to the database).
  const known = state.accounts.find((a) => a.account_id === accountId);
  const header = known
    ? `<div class="detail-top"><div class="detail-top-row"><div class="detail-header">
         <div class="detail-avatar" style="${avatarStyle(known.account_id)}">${initials(known.borrower_name)}</div>
         <div class="detail-title"><h2>${escapeHtml(known.borrower_name)}</h2><div class="biz">${escapeHtml(known.business_name)} · ${known.account_id}</div></div>
       </div>${DETAIL_CLOSE_HTML}</div><div class="detail-tabs-placeholder"></div></div>`
    : DETAIL_CLOSE_HTML;
  $detailPanel.innerHTML = `${header}<div class="detail-body"><div class="detail-loading"><div class="spinner"></div><p class="no-data">Loading the account…</p></div></div>`;
  wireDetailClose();

  try {
    const [account, documents, conversation] = await _loadDetailData(accountId);
    renderDetail(account, documents, conversation);
  } catch (e) {
    $detailPanel.innerHTML = `${DETAIL_CLOSE_HTML}<div class="detail-body"><p class="no-data">Couldn't load this account: ${escapeHtml(e.message)}</p></div>`;
    wireDetailClose();
  }
}

// The detail panel is split into tabs. Every action inside it re-renders the
// whole panel (renderDetail) from fresh data, so the chosen tab lives in state
// and is put back on each render -- recording a payment must not throw the
// operator back to the overview.
function applyDetailTab() {
  const tab = state.detailTab || "overview";
  $detailPanel.querySelectorAll("[data-detail-tab]").forEach((btn) => {
    const active = btn.dataset.detailTab === tab;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });
  $detailPanel.querySelectorAll("[data-detail-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.detailPanel !== tab;
  });
}

function wireDetailTabs() {
  $detailPanel.querySelectorAll("[data-detail-tab]").forEach((btn) =>
    btn.addEventListener("click", () => {
      state.detailTab = btn.dataset.detailTab;
      applyDetailTab();
      $detailPanel.querySelector(".detail-tabs")?.scrollIntoView({ block: "nearest" });
    })
  );
  applyDetailTab();
}

function wireDetailClose() {
  $detailPanel.querySelector(".detail-close").addEventListener("click", closeDetail);
}

function closeDetail() {
  $detailPanel.classList.remove("open");
  $detailBackdrop.classList.remove("open");
  state.detailAccountId = null;
  setTimeout(() => {
    $detailPanel.hidden = true;
    $detailBackdrop.hidden = true;
  }, 320);
}

$detailBackdrop.addEventListener("click", closeDetail);
wireCaseActions($detailPanel);
wireCaseActions(document.getElementById("inbox-list"));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$detailPanel.hidden) closeDetail();
});

/* ============================================================
   New account modal
   ============================================================ */
const $newAccountBackdrop = document.getElementById("new-account-backdrop");
const $newAccountModal = document.getElementById("new-account-modal");
const $newAccountForm = document.getElementById("new-account-form");
const $newAccountReview = document.getElementById("na-review");
const $newAccountSuccess = document.getElementById("na-success");

// Financial details are collected via one of two popovers (manual entry
// or the EMI calculator), not the main form's own inputs -- naFinancial
// holds whichever one the ops user last confirmed with "Use these
// details"/"Use this EMI", ready to merge into the create-account
// payload once they get through the review step.
let naFinancial = null;

function naCloseFinancialPopovers() {
  document.getElementById("na-manual-popover").hidden = true;
  document.getElementById("na-calc-popover").hidden = true;
}

function naRenderFinancialSummary() {
  const el = document.getElementById("na-financial-summary");
  if (!naFinancial) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  el.textContent = `✓ ₹${naFinancial.principal_amount.toLocaleString("en-IN")} principal, ₹${naFinancial.emi_amount}/mo for ${naFinancial.tenure_months} months`;
}

function openNewAccountModal() {
  $newAccountForm.reset();
  naFinancial = null;
  naCloseFinancialPopovers();
  naRenderFinancialSummary();
  document.getElementById("na-calc-use-row").hidden = true;
  document.getElementById("na-calc-result").textContent = "";
  document.getElementById("na-manual-error").textContent = "";
  document.getElementById("na-calc-error").textContent = "";
  $newAccountForm.hidden = false;
  $newAccountReview.hidden = true;
  $newAccountSuccess.hidden = true;
  document.getElementById("na-error").textContent = "";
  $newAccountModal.hidden = false;
  $newAccountBackdrop.hidden = false;
  setTimeout(() => {
    $newAccountModal.classList.add("open");
    $newAccountBackdrop.classList.add("open");
  }, 10);
}

function closeNewAccountModal() {
  $newAccountModal.classList.remove("open");
  $newAccountBackdrop.classList.remove("open");
  setTimeout(() => {
    $newAccountModal.hidden = true;
    $newAccountBackdrop.hidden = true;
  }, 300);
}

document.getElementById("new-account-btn").addEventListener("click", openNewAccountModal);
document.getElementById("new-account-close").addEventListener("click", closeNewAccountModal);
$newAccountBackdrop.addEventListener("click", closeNewAccountModal);

/* ---- Financial details: manual popover ---- */
document.getElementById("na-mode-manual-btn").addEventListener("click", () => {
  const popover = document.getElementById("na-manual-popover");
  const wasOpen = !popover.hidden;
  naCloseFinancialPopovers();
  popover.hidden = wasOpen;
});

document.getElementById("na-manual-use-btn").addEventListener("click", () => {
  const errorEl = document.getElementById("na-manual-error");
  const principal = Number(document.getElementById("na-m-principal").value);
  const emi = Number(document.getElementById("na-m-emi").value);
  const tenure = Number(document.getElementById("na-m-tenure").value);
  const dueDate = document.getElementById("na-m-due-date").value;
  if (!principal || !emi || !tenure || !dueDate) {
    errorEl.textContent = "Fill in all four fields.";
    return;
  }
  errorEl.textContent = "";
  naFinancial = { principal_amount: principal, emi_amount: emi, tenure_months: tenure, emi_due_date: dueDate };
  naCloseFinancialPopovers();
  naRenderFinancialSummary();
});

/* ---- Financial details: EMI calculator popover ---- */
document.getElementById("na-mode-calc-btn").addEventListener("click", () => {
  const popover = document.getElementById("na-calc-popover");
  const wasOpen = !popover.hidden;
  naCloseFinancialPopovers();
  popover.hidden = wasOpen;
});

// Standard reducing-balance EMI: EMI = P*r*(1+r)^n / ((1+r)^n - 1), r =
// monthly rate as a decimal. A 0% rate degenerates to a flat P/n split,
// which the general formula can't handle (division by (1+0)^n - 1 = 0).
function naCalculateEmi(principal, annualRatePct, months) {
  const monthlyRate = annualRatePct / 12 / 100;
  if (monthlyRate === 0) return principal / months;
  const factor = Math.pow(1 + monthlyRate, months);
  return (principal * monthlyRate * factor) / (factor - 1);
}

document.getElementById("na-calc-run-btn").addEventListener("click", () => {
  const resultEl = document.getElementById("na-calc-result");
  const useRow = document.getElementById("na-calc-use-row");
  const principal = Number(document.getElementById("na-c-principal").value);
  const rateInput = document.getElementById("na-c-rate").value;
  const rate = Number(rateInput);
  const tenure = Number(document.getElementById("na-c-tenure").value);
  const dueDate = document.getElementById("na-c-due-date").value;
  if (!principal || rateInput === "" || Number.isNaN(rate) || !tenure || !dueDate) {
    resultEl.textContent = "Fill in principal, rate, tenure, and due date first.";
    useRow.hidden = true;
    return;
  }
  const emi = naCalculateEmi(principal, rate, tenure);
  resultEl.innerHTML = `Estimated EMI: <b>₹${emi.toFixed(2)}</b>/month`;
  useRow.hidden = false;
  useRow.dataset.principal = principal;
  useRow.dataset.rate = rate;
  useRow.dataset.tenure = tenure;
  useRow.dataset.dueDate = dueDate;
  useRow.dataset.emi = emi.toFixed(2);
});

document.getElementById("na-calc-use-btn").addEventListener("click", () => {
  const useRow = document.getElementById("na-calc-use-row");
  naFinancial = {
    principal_amount: Number(useRow.dataset.principal),
    emi_amount: Number(useRow.dataset.emi),
    tenure_months: Number(useRow.dataset.tenure),
    emi_due_date: useRow.dataset.dueDate,
    interest_rate_pct: Number(useRow.dataset.rate),
  };
  document.getElementById("na-calc-error").textContent = "";
  naCloseFinancialPopovers();
  naRenderFinancialSummary();
});

/* ---- Review step ---- */
function naReviewRow(label, value) {
  return `<div class="na-review-item"><span class="label">${escapeHtml(label)}</span><span class="value">${escapeHtml(String(value))}</span></div>`;
}

function naRenderReview(payload) {
  const rows = [
    naReviewRow("Borrower", payload.borrower_name),
    naReviewRow("Business", payload.business_name),
    naReviewRow("Phone", payload.phone_number),
    naReviewRow("Language", payload.language_preference),
    naReviewRow("Loan type", payload.loan_type),
    naReviewRow("Onboarding risk rating", payload.risk_tier),
    naReviewRow("Principal", `₹${payload.principal_amount.toLocaleString("en-IN")}`),
    naReviewRow("EMI", `₹${payload.emi_amount}/month`),
    naReviewRow("Tenure", `${payload.tenure_months} months`),
    naReviewRow("First EMI due", payload.emi_due_date),
    naReviewRow("NACH mandate", payload.nach_mandate_active ? "Active" : "Not active"),
  ];
  if (payload.interest_rate_pct != null) rows.push(naReviewRow("Interest rate", `${payload.interest_rate_pct}% p.a.`));
  document.getElementById("na-review-grid").innerHTML = rows.join("");
}

$newAccountForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const errorEl = document.getElementById("na-error");

  if (!naFinancial) {
    errorEl.textContent = "Set the financial details first, via either \"Enter manually\" or \"Calculate EMI\" above.";
    return;
  }

  const payload = {
    borrower_name: document.getElementById("na-borrower-name").value.trim(),
    business_name: document.getElementById("na-business-name").value.trim(),
    // E.164 has no internal whitespace, but "+91 9022854526" is exactly
    // how a person naturally types an Indian number -- strip spaces
    // rather than reject a real phone number over formatting.
    phone_number: document.getElementById("na-phone").value.trim().replace(/\s+/g, ""),
    language_preference: document.getElementById("na-language").value,
    loan_type: document.getElementById("na-loan-type").value.trim(),
    risk_tier: document.getElementById("na-risk-tier").value,
    nach_mandate_active: document.getElementById("na-nach").checked,
    ...naFinancial,
  };
  if (!payload.borrower_name || !payload.business_name || !payload.phone_number || !payload.loan_type) {
    errorEl.textContent = "Fill in borrower name, business name, phone, and loan type.";
    return;
  }
  errorEl.textContent = "";

  naRenderReview(payload);
  $newAccountForm.hidden = true;
  $newAccountReview.hidden = false;
  $newAccountReview.dataset.payload = JSON.stringify(payload);
});

document.getElementById("na-review-edit-btn").addEventListener("click", () => {
  $newAccountReview.hidden = true;
  $newAccountForm.hidden = false;
});

document.getElementById("na-review-confirm-btn").addEventListener("click", async () => {
  const payload = JSON.parse($newAccountReview.dataset.payload);
  const errorEl = document.getElementById("na-review-error");
  errorEl.textContent = "";
  const btn = document.getElementById("na-review-confirm-btn");
  btn.disabled = true;
  btn.textContent = "Opening…";
  try {
    const result = await createAccount(payload);
    $newAccountReview.hidden = true;
    $newAccountSuccess.hidden = false;
    // A real, tappable deep link when TELEGRAM_BOT_USERNAME is configured
    // -- closes the actual gap this used to have: the operator was told
    // to relay the key manually, with no automated send and no link at
    // all. Falls back to the old manual-relay hint when it isn't
    // configured, rather than silently omitting either explanation.
    const inviteHtml = result.telegram_invite_link
      ? `<a class="btn btn-primary na-invite-link" href="${escapeHtml(result.telegram_invite_link)}" target="_blank" rel="noopener">Open Telegram invite link</a>
         <p class="clarify-hint">Send this link to the borrower, or tap it now — it verifies them the instant it's opened.</p>`
      : `<p class="clarify-hint">This key won't be shown again — record it now if the borrower needs it to verify over chat/Telegram.</p>`;
    $newAccountSuccess.innerHTML = `
      <p class="na-success-title">✓ ${escapeHtml(result.account.account_id)} opened for ${escapeHtml(payload.borrower_name)}</p>
      <p class="na-success-key">Access key: <b>${escapeHtml(result.access_key)}</b></p>
      ${inviteHtml}
      <button class="btn ${result.telegram_invite_link ? "btn-secondary" : "btn-primary"}" id="na-done-btn">Done</button>`;
    document.getElementById("na-done-btn").addEventListener("click", () => {
      closeNewAccountModal();
      loadAll();
    });
  } catch (err) {
    errorEl.textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Confirm — create account";
  }
});

// A plain-language digest of one account -- a one-line verdict plus
// pointwise Stats (neutral facts) and Worries (real concerns, with why),
// all built from a fixed template over real fields (never an LLM call),
// same "explainable, not a black box" discipline ops/flags.py already
// applies to flag reasons. An interviewer or a new ops hire should be
// able to read this and understand the account's condition without
// cross-referencing the charts below it.
function buildAccountDigest(a) {
  const isOverdueFlagged = a.flags.some((f) => f.label === "overdue");

  let verdict, severity;
  if (a.flags.length === 0 && a.days_past_due === 0) {
    verdict = "Clean account, fully current — no action needed.";
    severity = "clean";
  } else if (a.flags.length === 0) {
    verdict = `${a.days_past_due} day${a.days_past_due === 1 ? "" : "s"} past due but still within the grace period — no action needed yet.`;
    severity = "grace";
  } else {
    verdict = `${a.risk_tier[0].toUpperCase()}${a.risk_tier.slice(1)} risk — ${a.flags.length} active flag${a.flags.length === 1 ? "" : "s"}, needs attention.`;
    severity = "attention";
  }

  const stats = [
    `${a.loan_type} — ${fmtInr(a.principal_amount)} principal`,
    `${fmtInr(a.emi_amount)} EMI, next due ${fmtShortDate(a.emi_due_date)}`,
    `${Math.max(0, a.tenure_months - a.months_remaining)} of ${a.tenure_months} months paid down`,
    `NACH mandate ${a.nach_mandate_active ? "active" : "inactive"}`,
    `Prefers ${LANGUAGE_LABELS[a.language_preference] || a.language_preference}`,
  ];

  const worries = [];
  if (isOverdueFlagged) {
    worries.push(`${a.days_past_due} days past due (beyond the grace period)`);
  } else if (a.days_past_due > 0) {
    worries.push(`${a.days_past_due} day${a.days_past_due === 1 ? "" : "s"} past due, but still within grace`);
  }
  // Flags are the authoritative, already-explainable source (each carries
  // its own real reason, e.g. broken_promises' threshold) -- listed as-is
  // rather than also deriving a separate broken-promise-count line that
  // would just repeat the same fact with less context.
  a.flags.filter((f) => f.label !== "overdue").forEach((f) => worries.push(`${FLAG_LABELS[f.label] || f.label}: ${f.reason}`));

  return { verdict, severity, stats, worries };
}

function documentRowHtml(d) {
  return `<div class="document-row">
    <div class="document-info">
      <span class="document-name">${escapeHtml(d.filename)}</span>
      <span class="document-meta">${fmtFileSize(d.size_bytes)} · uploaded ${fmtDate(d.uploaded_at)}</span>
    </div>
    <button type="button" class="btn btn-secondary document-download-btn" data-filename="${escapeHtml(d.filename)}">Download</button>
  </div>`;
}

function conversationEntryHtml(e) {
  if (e.event_type === "tool_called") {
    return `<div class="convo-row tool"><span class="convo-tag">Tool</span><span class="convo-text">${escapeHtml(e.tool)}(${escapeHtml(JSON.stringify(e.arguments))})</span><span class="convo-time">${relativeTime(e.created_at)}</span></div>`;
  }
  const isUser = e.event_type === "user_message";
  return `<div class="convo-row ${isUser ? "user" : "assistant"}"><span class="convo-tag">${isUser ? "Borrower" : "Agent"}</span><span class="convo-text">${escapeHtml(e.content)}</span><span class="convo-time">${relativeTime(e.created_at)}</span></div>`;
}

function renderDetail(a, documents, conversation = []) {
  // Fetches the file itself (with the X-API-Key header, same auth as
  // every other call in this file) rather than a plain <a href> -- the
  // download endpoint 401s without that header, which a bare link has no
  // way to send. The blob + temporary <a download> is what actually
  // triggers the browser's save-file behavior once the bytes are in hand.
  function wireDocumentDownloads(container) {
    container.querySelectorAll(".document-download-btn").forEach((btn) =>
      btn.addEventListener("click", async () => {
        const filename = btn.dataset.filename;
        const originalText = btn.textContent;
        btn.disabled = true;
        btn.textContent = "Downloading…";
        try {
          const blob = await downloadDocument(a.account_id, filename);
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = filename;
          document.body.appendChild(link);
          link.click();
          link.remove();
          URL.revokeObjectURL(url);
        } catch (err) {
          toast(err.message, true);
        } finally {
          btn.disabled = false;
          btn.textContent = originalText;
        }
      })
    );
  }

  // A targeted re-render of just the documents section -- called on
  // initial render and again after a successful upload, deliberately NOT
  // via a full renderDetail() call, which would wipe out the "Ingested N
  // chunk(s)…" result message the upload handler just showed.
  function renderDocumentsList(list) {
    document.getElementById("documents-count").textContent = list.length;
    const container = document.getElementById("documents-list");
    container.innerHTML = list.length
      ? list.map(documentRowHtml).join("")
      : `<p class="no-data">No documents uploaded yet.</p>`;
    wireDocumentDownloads(container);
  }
  const monthsPaidDown = Math.max(0, a.tenure_months - a.months_remaining); // months_remaining can exceed tenure after an extension
  const paidOffFraction = a.tenure_months > 0 ? monthsPaidDown / a.tenure_months : 0;
  const ringColor = a.risk_tier === "high" || a.risk_tier === "critical" ? "var(--danger)" : a.risk_tier === "medium" ? "var(--warning)" : "var(--success)";
  const digest = buildAccountDigest(a);

  const flagsHtml = a.flags.length
    ? a.flags
        .map(
          (f) => `<div class="flag-detail-card ${f.label}">
        <div class="flag-label">${escapeHtml(FLAG_LABELS[f.label] || f.label)}</div>
        <div class="flag-reason">${escapeHtml(f.reason)}</div>
      </div>`
        )
        .join("")
    : `<p class="no-data">No active flags — this account is clean.</p>`;

  // Late fees on the books (schema.sql's charges table): each with the EMI it is for and what happened to it. One
  // still due can be waived -- who and why are required, and the borrower is told the reason.
  const charges = a.charges || [];
  const chargeStatus = (c) =>
    c.status === "paid" ? `Paid ${fmtDate(c.paid_on)}`
    : c.status === "waived" ? `Waived ${fmtDate(c.waived_on)} by ${c.waived_by}: ${c.waived_reason}`
    : c.amount_paid > 0 ? `${fmtInr(c.outstanding)} still due (${fmtInr(c.amount_paid)} paid)`
    : "Due";
  const chargesHtml = charges.length
    ? `<ul class="charge-list">${charges
        .slice()
        .reverse()
        .map(
          (c) => `<li class="charge-item charge-${c.status}">
            <div class="charge-main"><strong>${fmtInr(c.amount)} late fee</strong> · for the ${fmtDate(c.for_due_date)} EMI
              <span class="charge-status">${escapeHtml(chargeStatus(c))}</span></div>
            ${c.status === "due" ? `<button type="button" class="btn btn-secondary btn-sm" data-waive-charge="${c.id}">Waive</button>
            <form class="waive-form" data-waive-form="${c.id}" hidden>
              <input type="text" name="waived_by" placeholder="Your name" maxlength="80" required />
              <input type="text" name="reason" placeholder="Why (sent to the borrower)" maxlength="500" required />
              <button type="submit" class="btn btn-danger btn-sm">Waive ${fmtInr(c.outstanding)}</button>
              <p class="mini-form-error" hidden></p>
            </form>` : ""}
          </li>`
        )
        .join("")}</ul>`
    : `<p class="no-data">No late fees charged on this account.</p>`;

  const payments = a.payment_history.slice(-8);
  const maxAmt = Math.max(1, ...payments.map((p) => p.amount));
  const barsHtml = payments.length
    ? `<div class="bar-chart">
        ${payments
          .map(
            (p) => `<div class="bar-col" title="${fmtInr(p.amount)} on ${fmtDate(p.date)} — ${p.on_time ? "on time" : "late"}">
              <div class="bar-amount">${fmtInrShort(p.amount)}</div>
              <div class="bar-track">
                <div class="bar ${p.on_time ? "on-time" : "late"}" style="height:${Math.max(6, (p.amount / maxAmt) * 100)}%"></div>
              </div>
              <div class="bar-date">${fmtShortDate(p.date)}</div>
            </div>`
          )
          .join("")}
      </div>
      <div class="bar-legend"><span><i class="on-time"></i>On time</span><span><i class="late"></i>Late</span></div>`
    : `<p class="no-data">No payment history yet.</p>`;

  // Past entries come straight from real payment_history rows (already
  // chronological -- store.py loads them "order by payment_date"). Future
  // ones are computed, not stored: there's no "scheduled payments" table,
  // so this projects monthly from the account's real emi_due_date/
  // emi_amount for however many months_remaining says are left --
  // exactly the cadence every other EMI-due-date computation in this
  // codebase already assumes (get_payment_status, calculate_hypothetical).
  // toISOString() always renders in UTC -- for any borrower/operator in a
  // timezone AHEAD of UTC (IST, +5:30, being exactly this product's home
  // market), a `cursor` built at LOCAL midnight for e.g. 26 Aug lands on
  // 25 Aug 18:30 UTC, so slicing toISOString() silently reports 25 Aug
  // instead of 26 -- every date in this timeline (the "next due"/overdue
  // entry included) landing one calendar day early for exactly the
  // audience this dashboard is built for. Formatting from cursor's own
  // LOCAL year/month/day fields instead keeps the calendar date the
  // account's due-date cadence actually means, regardless of the
  // viewer's UTC offset.
  function toLocalIsoDate(d) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${y}-${m}-${day}`;
  }

  function buildEmiTimeline(account) {
    // Mirrors channels/browser_api.py's _build_emi_timeline -- see that
    // function's own comment for the full kind -> status/label mapping.
    const past = account.payment_history.map((p) => {
      const baseLabel = p.on_time ? "Paid on time" : "Paid late";
      if (p.kind === "extra_unapplied") return { date: p.date, amount: p.amount, status: "extra-unapplied", label: "Extra payment (not applied)" };
      if (p.kind === "extra_applied") return { date: p.date, amount: p.amount, status: "extra-applied", label: "Extra payment (credited to next EMI)" };
      if (p.kind === "overpayment_applied") return { date: p.date, amount: p.amount, status: p.on_time ? "paid-on-time" : "paid-late", label: `${baseLabel} + extra credited` };
      if (p.kind === "principal_prepayment_reduce_emi") return { date: p.date, amount: p.amount, status: p.on_time ? "paid-on-time" : "paid-late", label: `${baseLabel} + extra paid off the loan (EMI lowered)` };
      if (p.kind === "principal_prepayment_reduce_tenure") return { date: p.date, amount: p.amount, status: p.on_time ? "paid-on-time" : "paid-late", label: `${baseLabel} + extra paid off the loan (loan shortened)` };
      return { date: p.date, amount: p.amount, status: p.on_time ? "paid-on-time" : "paid-late", label: baseLabel };
    });

    const upcoming = [];
    if (account.months_remaining > 0 && account.emi_due_date) {
      const cursor = new Date(`${account.emi_due_date}T00:00:00`);
      const credit = account.pending_emi_credit || 0;
      for (let i = 0; i < account.months_remaining; i++) {
        const isNextDue = i === 0;
        const overdue = isNextDue && account.days_past_due > 0;
        let label = !overdue
          ? "Scheduled"
          : account.days_past_due > GRACE_PERIOD_DAYS
            ? `Overdue — ${account.days_past_due}d past due`
            : `Due ${account.days_past_due}d ago — within the grace period`;
        if (isNextDue && credit > 0.01) label += ` (${fmtInr(credit)} credited)`;
        upcoming.push({
          date: toLocalIsoDate(cursor),
          amount: isNextDue ? Math.round((account.emi_amount - credit) * 100) / 100 : account.emi_amount,
          status: overdue ? "overdue" : "upcoming",
          label,
        });
        cursor.setMonth(cursor.getMonth() + 1);
      }
    }
    return [...past, ...upcoming];
  }

  const timeline = buildEmiTimeline(a);
  const timelineHtml = timeline.length
    ? timeline
        .map(
          (t) => `<div class="timeline-row">
            <div class="timeline-dot ${t.status}"></div>
            <div class="timeline-info">
              <b>${fmtInr(t.amount)}</b> — ${escapeHtml(t.label)}
              <div class="timeline-meta">${fmtDate(t.date)}</div>
            </div>
          </div>`
        )
        .join("")
    : `<p class="no-data">No payment history and no scheduled EMIs.</p>`;

  // Cases tab: open cases first (each one conversation, with its action), then
  // one history of everything else between the team and this borrower. This
  // replaces two tabs ("Case activity", "Outreach") and six separate lists
  // (escalations, disputes, promises, calls, clarification history, the
  // clarification form) that each told part of the same story.
  const openCases = a.cases.filter((c) => c.open);
  const history = [];
  a.cases
    .filter((c) => !c.open)
    .forEach((c) => {
      const msgs = c.thread || [];
      history.push({
        when: c.closed_at || c.opened_at,
        tone: c.status === "rejected" ? "broken" : "kept",
        icon: "✓",
        html: `<b>${escapeHtml(KIND_LABELS[c.kind] || c.kind)}${caseHeading(c) ? `: ${escapeHtml(caseHeading(c))}` : ""}</b> — ${escapeHtml(c.status)}
          <div class="history-sub">${escapeHtml(c.borrower_words)}</div>
          ${c.outcome ? `<div class="history-sub">Outcome: ${escapeHtml(c.outcome)}</div>` : ""}
          ${msgs.length ? `<details class="history-thread"><summary>${msgs.length} message${msgs.length === 1 ? "" : "s"}</summary>${msgs.map(caseMessageHtml).join("")}</details>` : ""}`,
      });
    });
  // One entry per promise (a plan of one or more parts, accounts/promises.py), judged by its own words.
  const plans = new Map();
  a.promises.forEach((p, i) => {
    const key = p.plan_id || `row-${i}`;
    if (!plans.has(key)) plans.set(key, []);
    plans.get(key).push(p);
  });
  plans.forEach((parts) => {
    parts.sort((x, y) => x.part_no - y.part_no);
    const res = parts.map((p) => p.resolution || (p.kept === null ? "open" : p.kept ? "kept" : "broken"));
    const status = res.includes("broken") ? "broken"
      : res.every((r) => r === "kept") ? "kept"
      : !res.includes("open") ? "replaced" : "on track";
    const terms = parts[0].terms_text || parts.map((p) => `${fmtInr(p.promised_amount)} by ${fmtDate(p.promised_date)}`).join(", then ");
    const via = PROMISE_VIA_LABELS[parts[0].made_via] || "";
    const partLines = parts.length > 1
      ? parts.map((p, k) => `<div class="history-sub">Part ${p.part_no}: ${fmtInr(p.promised_amount)} by ${fmtDate(p.promised_date)} — ${escapeHtml(PROMISE_PART_LABELS[res[k]] || res[k])}</div>`).join("")
      : "";
    history.push({
      when: parts[0].made_on,
      tone: status === "kept" ? "kept" : status === "broken" ? "broken" : "pending",
      icon: status === "kept" ? "✓" : status === "broken" ? "✕" : "…",
      html: `Promised <b>${escapeHtml(terms)}</b> — ${escapeHtml(status)}${via ? `<div class="history-sub">${escapeHtml(via)}</div>` : ""}${partLines}`,
    });
  });
  a.call_log.forEach((c) =>
    history.push({
      when: c.created_at,
      tone: c.outcome === "reached" ? "kept" : "pending",
      icon: "☎",
      html: `Call: ${escapeHtml(CALL_OUTCOME_LABELS[c.outcome] || c.outcome)}${c.note ? ` — ${escapeHtml(c.note)}` : ""}`,
    })
  );
  a.messages.forEach((m) =>
    history.push({
      when: m.created_at,
      tone: "info",
      icon: "✉",
      html: `${escapeHtml(SENT_KIND_LABELS[m.kind] || "Message sent")}${m.delivered_via_telegram ? " · on Telegram" : " · dashboard only"}<div class="history-sub">${escapeHtml(m.message)}</div>`,
    })
  );
  history.sort((x, y) => new Date(y.when) - new Date(x.when));
  const HISTORY_SHOWN = 12;
  const historyHtml = history.length
    ? history
        .map(
          (h, i) => `<div class="history-row"${i >= HISTORY_SHOWN ? " hidden data-history-more" : ""}>
            <div class="promise-dot ${h.tone}">${h.icon}</div>
            <div class="history-info">${h.html}<div class="promise-meta">${relativeTime(h.when)}</div></div>
          </div>`
        )
        .join("")
    : `<p class="no-data">Nothing yet.</p>`;

  $detailPanel.innerHTML = `
    <div class="detail-top">
      <div class="detail-top-row">
        <div class="detail-header">
          <div class="detail-avatar" style="${avatarStyle(a.account_id)}">${initials(a.borrower_name)}</div>
          <div class="detail-title">
            <h2>${escapeHtml(a.borrower_name)}</h2>
            <div class="biz">${escapeHtml(a.business_name)} · ${a.account_id}</div>
          </div>
        </div>
        ${DETAIL_CLOSE_HTML}
      </div>
      <div class="detail-badges">
        <span class="badge risk-${a.risk_tier}"><span class="risk-dot ${a.risk_tier}"></span>${a.risk_tier} risk</span>
        <span class="badge">${escapeHtml(a.loan_type)}</span>
        <span class="badge">${LANGUAGE_LABELS[a.language_preference] || a.language_preference}</span>
        <a class="badge badge-link" href="tel:${escapeHtml(a.phone_number)}">${escapeHtml(a.phone_number)}</a>
${a.telegram_linked
          ? `<span class="badge risk-low">Telegram linked</span><button type="button" class="badge badge-edit-btn" id="unlink-telegram-btn" title="Stop sending to this chat; it must verify again">Unlink</button>`
          : `<span class="badge" title="Reminders and decisions are logged but not delivered until the borrower links Telegram (Reset key gives a fresh invite link)">Not on Telegram</span>`}
        <button type="button" class="badge badge-edit-btn" id="edit-account-btn" title="Edit phone, language, or onboarding risk rating">✎ Edit</button>
      </div>
      ${riskWhyHtml(a)}

      <form class="upload-form edit-account-form" id="edit-account-form" hidden>
        <div class="upload-row">
          <input type="text" id="ea-phone" placeholder="Phone (E.164, e.g. +919812345001)" value="${escapeHtml(a.phone_number)}" />
          <select id="ea-language">
            <option value="en" ${a.language_preference === "en" ? "selected" : ""}>English</option>
            <option value="hi" ${a.language_preference === "hi" ? "selected" : ""}>Hindi</option>
            <option value="hinglish" ${a.language_preference === "hinglish" ? "selected" : ""}>Hinglish</option>
          </select>
          <select id="ea-risk" title="The rating given at onboarding -- one input to the risk shown above">
            <option value="low" ${a.onboarding_risk_tier === "low" ? "selected" : ""}>Onboarding rating: low</option>
            <option value="medium" ${a.onboarding_risk_tier === "medium" ? "selected" : ""}>Onboarding rating: medium</option>
            <option value="high" ${a.onboarding_risk_tier === "high" ? "selected" : ""}>Onboarding rating: high</option>
          </select>
        </div>
        <div class="clarify-actions-row">
          <button type="submit" class="btn btn-primary">Save changes</button>
          <button type="button" class="btn btn-secondary" id="edit-account-cancel-btn">Cancel</button>
        </div>
        <p class="mini-form-error" id="edit-account-error" hidden></p>
      </form>

      <nav class="detail-tabs" role="tablist" aria-label="Account sections">
        <button type="button" class="detail-tab" role="tab" data-detail-tab="overview">Overview</button>
        <button type="button" class="detail-tab" role="tab" data-detail-tab="cases">Cases${openCases.length ? `<span class="tab-count attention" title="Open cases">${openCases.length}</span>` : ""}</button>
        <button type="button" class="detail-tab" role="tab" data-detail-tab="payments">Payments</button>
        <button type="button" class="detail-tab" role="tab" data-detail-tab="documents">Documents<span class="tab-count" id="documents-count">${documents.length}</span></button>
      </nav>
    </div>

    <div class="detail-body">
      <section class="detail-tabpanel" data-detail-panel="overview" role="tabpanel">
        <p class="account-verdict ${digest.severity}">${escapeHtml(digest.verdict)}</p>

        <div class="stats-worries-row">
          <div class="sw-block">
            <p class="sw-title">Stats</p>
            <ul class="sw-list">${digest.stats.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul>
          </div>
          <div class="sw-block worries">
            <p class="sw-title">Worries</p>
            ${
              digest.worries.length
                ? `<ul class="sw-list">${digest.worries.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>`
                : `<p class="no-worries">None — clean account.</p>`
            }
          </div>
        </div>

        <div class="hero-row">
          <div class="hero-card">
            <p class="hero-card-title">Loan progress</p>
            <div class="ring-wrap">
              <div class="ring-stack">
                ${ringChartSvg(paidOffFraction, ringColor)}
                <div class="ring-overlay">
                  <div class="ring-center-value">${a.months_remaining}</div>
                  <div class="ring-center-label">mo left</div>
                </div>
              </div>
              <div class="ring-legend">
                <div><b>${Math.round(paidOffFraction * 100)}%</b> of the ${a.tenure_months}-month term elapsed</div>
                <div>${monthsPaidDown} of ${a.tenure_months} months paid down</div>
              </div>
            </div>
          </div>
          <div class="hero-card">
            <p class="hero-card-title">Account snapshot</p>
            <div class="stat-cluster">
              <div class="stat-mini"><div class="label">Principal</div><div class="value">${fmtInr(a.principal_amount)}</div></div>
              <div class="stat-mini"><div class="label">EMI</div><div class="value">${fmtInr(a.emi_amount)}</div></div>
              <div class="stat-mini"><div class="label">Next due</div><div class="value">${fmtShortDate(a.emi_due_date)}</div></div>
              <div class="stat-mini"><div class="label">Days past due</div><div class="value ${a.days_past_due > 0 ? "flagged" : ""}">${a.days_past_due}</div></div>
              <div class="stat-mini"><div class="label">NACH mandate</div><div class="value ${a.nach_mandate_active ? "on" : "off"}">${a.nach_mandate_active ? "Active" : "Inactive"}</div></div>
              <div class="stat-mini"><div class="label">Dispute</div><div class="value ${a.dispute_open ? "flagged" : "on"}">${a.dispute_open ? "Open" : "None"}</div></div>
            </div>
          </div>
        </div>

        <div class="section-block">
          <p class="section-title">Flags <span class="count">${a.flags.length}</span></p>
          <p class="section-subtitle">Rule-based, not a black-box score — each flag below states exactly why it fired.</p>
          ${flagsHtml}
        </div>

        <div class="section-block">
          <p class="section-title">Telegram <span class="status-dot ${a.telegram_linked ? "on" : "off"}">${a.telegram_linked ? "Linked" : "Not linked"}</span></p>
          <p class="section-subtitle">${
            a.telegram_linked
              ? "This borrower has verified over Telegram — reminders and replies reach them there too."
              : "Not linked yet — reminders and replies are shown on their dashboard only."
          }</p>
          <button type="button" class="btn ${a.telegram_linked ? "btn-secondary" : "btn-primary"}" id="telegram-invite-btn">
            ${a.telegram_linked ? "Reset access key & get a new invite link" : "Get a Telegram invite link"}
          </button>
          <div id="telegram-invite-result"></div>
        </div>
      </section>

      <section class="detail-tabpanel" data-detail-panel="cases" role="tabpanel" hidden>
        <div class="cases-toolbar" role="group" aria-label="Add to this account">
          <button type="button" class="btn btn-secondary" data-compose="message" aria-pressed="false">✉ Message the borrower</button>
          <button type="button" class="btn btn-secondary" data-compose="promise" aria-pressed="false">+ Promise to pay</button>
          <button type="button" class="btn btn-secondary" data-compose="call" aria-pressed="false">+ Log a call</button>
        </div>

        <div class="compose-panel section-block" data-compose-panel="message" hidden>
          <p class="section-title">Message the borrower</p>
          <p class="section-subtitle">For anything that isn't about one open case — to answer a case, use Reply on it.</p>
          <form class="clarify-form" id="clarify-form">
            <label class="clarify-label" for="clarify-note">Your note (what's going on, in your own words)</label>
            <textarea id="clarify-note" rows="2" placeholder="e.g. Third missed promise this quarter — need them to explain."></textarea>
            <div class="clarify-actions-row">
              <button type="button" class="btn btn-secondary" id="clarify-polish-btn">✨ Polish wording</button>
              ${a.flags.length ? `<button type="button" class="btn btn-secondary" id="clarify-summarize-btn">📋 Summarize all flags</button>` : ""}
              <span class="clarify-hint">Drafts a message below from this account's real flags — you review it before sending.</span>
            </div>
            <label class="clarify-label" for="clarify-message">Message to send</label>
            <textarea id="clarify-message" rows="4" placeholder="Write directly, or use Polish wording above to draft one."></textarea>
            <div class="clarify-actions-row">
              <button type="submit" class="btn btn-primary" id="clarify-send-btn">Send to borrower</button>
              <span class="clarify-hint">Shown on their dashboard; sent on Telegram too if linked.</span>
            </div>
            <p class="clarify-result" id="clarify-result" hidden></p>
          </form>
        </div>

        <div class="compose-panel section-block" data-compose-panel="promise" hidden>
          <form class="upload-form" id="log-promise-form">
            <p class="inline-form-title">A promise to pay the borrower made (e.g. on a call)</p>
            <p class="section-subtitle">Exactly what they said: each amount by its date. Every part must be paid in full by its date, or the promise is broken.</p>
            <div id="lp-parts">
              <div class="upload-row lp-part">
                <input type="number" class="lp-amount" placeholder="Amount (₹)" min="0.01" step="0.01" required />
                <input type="date" class="lp-date" required title="Date they promised to pay by" />
              </div>
            </div>
            <button type="button" class="btn btn-secondary" id="lp-add-part">+ Add a part</button>
            <p class="lp-readback" id="lp-readback" hidden></p>
            <label class="lp-confirm"><input type="checkbox" id="lp-confirmed" /> I read these terms back and the borrower confirmed them</label>
            <button type="submit" class="btn btn-primary upload-btn" id="log-promise-btn">Log a promise to pay</button>
            <p class="mini-form-error" id="log-promise-error" hidden></p>
            <p class="upload-result" id="log-promise-result" hidden></p>
          </form>
        </div>

        <div class="compose-panel section-block" data-compose-panel="call" hidden>
          <form class="upload-form" id="call-log-form">
            <p class="inline-form-title">A call to ${escapeHtml(a.phone_number)}</p>
            <div class="upload-row">
              <select id="cl-outcome">
                <option value="reached">Reached</option>
                <option value="no_answer">No answer</option>
                <option value="voicemail">Left voicemail</option>
                <option value="wrong_number">Wrong number</option>
              </select>
              <input type="text" id="cl-note" placeholder="Note (optional)" />
            </div>
            <button type="submit" class="btn btn-primary upload-btn" id="call-log-btn">Log this call</button>
          </form>
          <div class="call-log-nudge" id="call-log-nudge" hidden>
            <p class="clarify-hint">Didn't reach them — escalate to a human instead of leaving it here?</p>
            <div class="clarify-actions-row">
              <button type="button" class="btn btn-primary" id="call-log-escalate-btn">Escalate to a human</button>
              <button type="button" class="btn btn-secondary" id="call-log-dismiss-btn">No, that's fine</button>
            </div>
          </div>
        </div>

        <div class="section-block">
          <p class="section-title">Open <span class="count">${openCases.length}</span></p>
          <p class="section-subtitle">Each case is one conversation with the borrower: what they said, every reply, and the action that closes it.</p>
          <div class="case-list">${
            openCases.length
              ? openCases.map((c) => caseCardHtml(c, { amountHint: a.amount_due_now })).join("")
              : `<p class="no-data">Nothing open on this account.</p>`
          }</div>
        </div>

        <div class="section-block">
          <p class="section-title">History <span class="count">${history.length}</span></p>
          <p class="section-subtitle">Everything else between the team and this borrower, newest first: closed cases, promises, calls and messages sent.</p>
          <div class="history-list">${historyHtml}</div>
          ${history.length > HISTORY_SHOWN ? `<button type="button" class="btn btn-ghost history-more-btn" id="history-more-btn">Show all ${history.length}</button>` : ""}
        </div>

        <details class="section-block convo-details">
          <summary class="section-title">AI conversation <span class="count">${conversation.length}</span></summary>
          <p class="section-subtitle">What the AI agent said to this borrower, and what it did on their behalf.</p>
          <div class="convo-list">${
            conversation.length
              ? conversation.map(conversationEntryHtml).join("")
              : `<p class="no-data">No AI conversation on record for this account yet.</p>`
          }</div>
        </details>
      </section>

      <section class="detail-tabpanel" data-detail-panel="payments" role="tabpanel" hidden>
        <div class="detail-cols">
          <div class="section-block">
            <p class="section-title">Payment history <span class="count">last ${payments.length}</span></p>
            <p class="section-subtitle">₹ amount per payment, colored by whether it arrived on time.</p>
            ${barsHtml}
            <p class="section-title" style="margin-top:16px">Late fees <span class="count">${charges.filter((c) => c.status === "due").length} due</span></p>
            <p class="section-subtitle">Charged once per EMI still unpaid after the grace period. A payment goes to the EMI first, then the fee.</p>
            ${chargesHtml}
            <form class="upload-form inline-form" id="record-payment-form">
              <p class="inline-form-title">Record a payment</p>
              <div class="upload-row">
                <input type="number" id="rp-amount" placeholder="Amount (₹)" min="0.01" step="0.01" required />
                <input type="date" id="rp-date" title="Payment date (defaults to today)" />
              </div>
              <div class="mini-form-radio-row" id="rp-extra-question" hidden>
                <span class="clarify-hint full-row">Less than what's due this cycle — apply it toward the next EMI?</span>
                <label><input type="radio" name="rp-apply" value="true" /> Yes, credit it</label>
                <label><input type="radio" name="rp-apply" value="false" /> No, just record it</label>
              </div>
              <button type="submit" class="btn btn-primary upload-btn" id="record-payment-btn">Record payment</button>
              <p class="mini-form-error" id="record-payment-error" hidden></p>
              <p class="upload-result" id="record-payment-result" hidden></p>
            </form>
          </div>

          <div class="section-block">
            <p class="section-title">EMI timeline <span class="count">${timeline.length}</span></p>
            <p class="section-subtitle">Every real past payment, plus every EMI still scheduled ahead — projected monthly from the next due date.</p>
            <div class="timeline-list">${timelineHtml}</div>
          </div>
        </div>
      </section>

      <section class="detail-tabpanel" data-detail-panel="documents" role="tabpanel" hidden>
        <div class="detail-cols">
          <div class="section-block">
            <p class="section-title">Documents</p>
            <p class="section-subtitle">Everything uploaded for this borrower — signed agreements, KYC, and more.</p>
            <div id="documents-list"></div>
          </div>

          <div class="section-block">
            <p class="section-title">Upload a document</p>
            <form class="upload-form" id="upload-form">
              <div class="upload-row">
                <select id="upload-type">
                  <option value="loan_agreement">Loan agreement</option>
                  <option value="kyc">KYC</option>
                  <option value="regulatory">Regulatory</option>
                  <option value="other">Other</option>
                </select>
                <input type="file" id="upload-file" accept=".pdf,.docx,.md" required />
              </div>
              <button type="submit" class="btn btn-primary upload-btn">Upload &amp; ingest</button>
              <p class="upload-result" id="upload-result" hidden></p>
            </form>
          </div>
        </div>
      </section>
    </div>`;

  wireDetailClose();
  wireDetailTabs();
  renderDocumentsList(documents);

  // "Message the borrower", "Promise to pay", "Log a call": one at a time, each opened from the Cases toolbar.
  $detailPanel.querySelectorAll("[data-compose]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const name = btn.dataset.compose;
      const panel = $detailPanel.querySelector(`[data-compose-panel="${name}"]`);
      const opening = panel.hidden;
      $detailPanel.querySelectorAll("[data-compose-panel]").forEach((p) => { p.hidden = true; });
      $detailPanel.querySelectorAll("[data-compose]").forEach((b) => b.setAttribute("aria-pressed", "false"));
      panel.hidden = !opening;
      btn.setAttribute("aria-pressed", String(opening));
      if (opening) panel.querySelector("textarea, input, select")?.focus();
    })
  );
  document.getElementById("history-more-btn")?.addEventListener("click", (e) => {
    $detailPanel.querySelectorAll("[data-history-more]").forEach((row) => { row.hidden = false; });
    e.currentTarget.remove();
  });

  const polishBtn = document.getElementById("clarify-polish-btn");
  polishBtn.addEventListener("click", async () => {
    const note = document.getElementById("clarify-note").value.trim();
    if (!note) {
      toast("Write a short note first — what should be polished?", true);
      return;
    }
    polishBtn.disabled = true;
    polishBtn.textContent = "Polishing…";
    try {
      const result = await draftClarification(a.account_id, note);
      document.getElementById("clarify-message").value = result.draft;
      if (result.warnings && result.warnings.length) toast(result.warnings.join(" "), true);
    } catch (err) {
      toast(err.message, true);
    } finally {
      polishBtn.disabled = false;
      polishBtn.textContent = "✨ Polish wording";
    }
  });

  // "Round up all the flags" -- draftClarification always sends every one
  // of this account's real, current flag reasons to the LLM regardless of
  // what note the operator wrote (see ops/api.py's draft_clarification);
  // this just skips needing to write a note first, for the common case of
  // wanting a message that's purely "here's everything currently open on
  // your account, please respond" rather than adding a specific new point.
  const summarizeBtn = document.getElementById("clarify-summarize-btn");
  summarizeBtn?.addEventListener("click", async () => {
    summarizeBtn.disabled = true;
    summarizeBtn.textContent = "Summarizing…";
    try {
      const result = await draftClarification(
        a.account_id,
        "Tell the borrower plainly everything on their account that needs their attention, and ask them to reply."
      );
      document.getElementById("clarify-message").value = result.draft;
      if (result.warnings && result.warnings.length) toast(result.warnings.join(" "), true);
    } catch (err) {
      toast(err.message, true);
    } finally {
      summarizeBtn.disabled = false;
      summarizeBtn.textContent = "📋 Summarize all flags";
    }
  });

  const clarifyForm = document.getElementById("clarify-form");
  clarifyForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = document.getElementById("clarify-message").value.trim();
    if (!message) {
      toast("Write or polish a message first.", true);
      return;
    }
    const sendBtn = document.getElementById("clarify-send-btn");
    sendBtn.disabled = true;
    sendBtn.textContent = "Sending…";
    try {
      const result = await sendClarificationRequest(a.account_id, message);
      toast(result.delivered_via_telegram ? "Sent — delivered over Telegram." : "Sent — logged (no linked Telegram to deliver to).");
      const [fresh, freshDocuments, freshConversation] = await _loadDetailData(a.account_id);
      renderDetail(fresh, freshDocuments, freshConversation);
    } catch (err) {
      toast(err.message, true);
      sendBtn.disabled = false;
      sendBtn.textContent = "Send to borrower";
    }
  });

  const uploadForm = document.getElementById("upload-form");
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("upload-file");
    const file = fileInput.files[0];
    if (!file) return;
    const submitBtn = uploadForm.querySelector("button[type=submit]");
    submitBtn.disabled = true;
    submitBtn.textContent = "Uploading…";
    try {
      const result = await uploadDocument(a.account_id, file, document.getElementById("upload-type").value);
      const resultEl = document.getElementById("upload-result");
      resultEl.hidden = false;
      resultEl.textContent = `Ingested ${result.chunks_stored} chunk(s) from ${result.filename}${result.interest_rate_extracted ? " — interest rate extracted." : "."}`;
      toast("Document uploaded and ingested.");
      fileInput.value = "";
      // Refresh just the documents list, not the whole panel -- a full
      // renderDetail() would wipe the "Ingested N chunk(s)…" message above
      // before the operator has had a chance to read it.
      renderDocumentsList(await getDocuments(a.account_id));
    } catch (err) {
      toast(err.message, true);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Upload & ingest";
    }
  });

  // A full reload (renderDetail + loadAll), not a targeted re-render, for
  // the actions below -- each can change data shown elsewhere too (the
  // account grid's EMI/flags after a payment, the inbox after a promise),
  // same reasoning as afterCaseAction for the case cards.
  async function refreshAfterAction() {
    const [fresh, freshDocuments, freshConversation] = await _loadDetailData(a.account_id);
    renderDetail(fresh, freshDocuments, freshConversation);
    loadAll();
  }

  document.querySelectorAll("[data-waive-charge]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const form = document.querySelector(`[data-waive-form="${btn.dataset.waiveCharge}"]`);
      form.hidden = !form.hidden;
      if (!form.hidden) form.querySelector("input[name=waived_by]").focus();
    });
  });
  document.querySelectorAll("[data-waive-form]").forEach((form) => {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const errorEl = form.querySelector(".mini-form-error");
      errorEl.hidden = true;
      const waivedBy = form.waived_by.value.trim();
      const reason = form.reason.value.trim();
      if (waivedBy.length < 2 || reason.length < 3) {
        errorEl.textContent = "Add your name and the reason -- the borrower is told why.";
        errorEl.hidden = false;
        return;
      }
      const submit = form.querySelector("button[type=submit]");
      submit.disabled = true;
      try {
        const result = await waiveCharge(a.account_id, form.dataset.waiveForm, { waived_by: waivedBy, reason });
        toast(`Late fee waived${result.borrower_notified ? " — the borrower was told on Telegram" : " — the borrower sees it on their dashboard"}.`);
        await refreshAfterAction();
      } catch (err) {
        errorEl.textContent = err.message;
        errorEl.hidden = false;
        submit.disabled = false;
      }
    });
  });

  const recordPaymentForm = document.getElementById("record-payment-form");
  recordPaymentForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const errorEl = document.getElementById("record-payment-error");
    const extraQuestion = document.getElementById("rp-extra-question");
    errorEl.hidden = true;

    const amount = Number(document.getElementById("rp-amount").value);
    const paymentDate = document.getElementById("rp-date").value || null;
    if (!amount || amount <= 0) {
      errorEl.textContent = "Enter a valid amount.";
      errorEl.hidden = false;
      return;
    }

    let applyExtraToNext = null;
    if (!extraQuestion.hidden) {
      const picked = recordPaymentForm.querySelector('input[name="rp-apply"]:checked');
      if (!picked) {
        errorEl.textContent = "Choose whether to apply this toward the next EMI.";
        errorEl.hidden = false;
        return;
      }
      applyExtraToNext = picked.value === "true";
    }

    const btn = document.getElementById("record-payment-btn");
    btn.disabled = true;
    btn.textContent = "Recording…";
    try {
      const payload = { amount, payment_date: paymentDate };
      if (applyExtraToNext !== null) payload.apply_extra_to_next = applyExtraToNext;
      const result = await recordPayment(a.account_id, payload);
      toast(`Payment recorded — ${result.kind.replace(/_/g, " ")}${result.late_fee_paid > 0 ? `, including the ${fmtInr(result.late_fee_paid)} late fee` : ""}.`);
      await refreshAfterAction();
    } catch (err) {
      // A 422 means store.record_payment needs the apply-to-next-EMI
      // decision (see RecordPaymentIn's own docstring) -- ask, rather than
      // surface it as a generic error the operator can't act on.
      if (err instanceof ApiError && err.status === 422) {
        extraQuestion.hidden = false;
      } else {
        errorEl.textContent = err.message;
        errorEl.hidden = false;
      }
      btn.disabled = false;
      btn.textContent = "Record payment";
    }
  });

  const logPromiseForm = document.getElementById("log-promise-form");
  const partsEl = document.getElementById("lp-parts");
  const readbackEl = document.getElementById("lp-readback");
  const readPromiseParts = () =>
    [...partsEl.querySelectorAll(".lp-part")].map((row) => ({
      amount: Number(row.querySelector(".lp-amount").value),
      date: row.querySelector(".lp-date").value,
    }));
  // The terms as they will be recorded, to read back to the borrower before logging.
  const updateReadback = () => {
    const parts = readPromiseParts().filter((p) => p.amount > 0 && p.date);
    readbackEl.hidden = !parts.length;
    readbackEl.textContent = parts.length ? `Read back: ${parts.map((p) => `${fmtInr(p.amount)} by ${fmtDate(p.date)}`).join(", then ")}` : "";
    document.getElementById("lp-confirmed").checked = false;
  };
  partsEl.addEventListener("input", updateReadback);
  document.getElementById("lp-add-part").addEventListener("click", () => {
    if (partsEl.querySelectorAll(".lp-part").length >= 4) {
      toast("A promise has at most 4 parts.");
      return;
    }
    partsEl.insertAdjacentHTML("beforeend", `<div class="upload-row lp-part">
      <input type="number" class="lp-amount" placeholder="Amount (₹)" min="0.01" step="0.01" required />
      <input type="date" class="lp-date" required title="Date they promised to pay by" />
    </div>`);
  });
  logPromiseForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const errorEl = document.getElementById("log-promise-error");
    errorEl.hidden = true;
    const parts = readPromiseParts();
    if (!parts.length || parts.some((p) => !p.amount || p.amount <= 0 || !p.date)) {
      errorEl.textContent = "Enter an amount and a date for every part.";
      errorEl.hidden = false;
      return;
    }
    if (parts.some((p, k) => k > 0 && p.date <= parts[k - 1].date)) {
      errorEl.textContent = "Each part's date must come after the one before it.";
      errorEl.hidden = false;
      return;
    }
    if (!document.getElementById("lp-confirmed").checked) {
      errorEl.textContent = "Read the terms back to the borrower and tick that they confirmed them.";
      errorEl.hidden = false;
      return;
    }
    const btn = document.getElementById("log-promise-btn");
    btn.disabled = true;
    btn.textContent = "Logging…";
    try {
      const [first, ...rest] = parts;
      const result = await logPromise(a.account_id, {
        promised_date: first.date,
        promised_amount: first.amount,
        more_parts: rest.map((p) => ({ date: p.date, amount: p.amount })),
      });
      toast(result.replaced_earlier_promise ? "Promise logged. It replaces the one that was open." : "Promise logged.");
      await refreshAfterAction();
    } catch (err) {
      errorEl.textContent = err.message;
      errorEl.hidden = false;
      btn.disabled = false;
      btn.textContent = "Log a promise to pay";
    }
  });

  const callLogForm = document.getElementById("call-log-form");
  callLogForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const outcome = document.getElementById("cl-outcome").value;
    const note = document.getElementById("cl-note").value.trim() || null;
    const btn = document.getElementById("call-log-btn");
    btn.disabled = true;
    btn.textContent = "Logging…";
    try {
      await logCall(a.account_id, { outcome, note });
      toast("Call logged.");
      // A failed contact attempt is a dead end otherwise -- found live,
      // logging "no answer" had no next step at all. Not "reached" holds
      // off the usual full refresh (which would instantly wipe this
      // nudge, same reasoning as the upload-result message elsewhere in
      // this file) until the operator either escalates or dismisses it.
      if (outcome === "reached") {
        await refreshAfterAction();
      } else {
        callLogForm.hidden = true;
        document.getElementById("call-log-nudge").hidden = false;
      }
    } catch (err) {
      toast(err.message, true);
      btn.disabled = false;
      btn.textContent = "Log this call";
    }
  });

  document.getElementById("call-log-escalate-btn").addEventListener("click", async () => {
    const escalateBtn = document.getElementById("call-log-escalate-btn");
    escalateBtn.disabled = true;
    escalateBtn.textContent = "Escalating…";
    try {
      await escalateAccount(a.account_id, "Repeated failed contact attempt over the phone");
      toast("Escalated to a human.");
      await refreshAfterAction();
    } catch (err) {
      toast(err.message, true);
      escalateBtn.disabled = false;
      escalateBtn.textContent = "Escalate to a human";
    }
  });

  document.getElementById("call-log-dismiss-btn").addEventListener("click", refreshAfterAction);

  const unlinkBtn = document.getElementById("unlink-telegram-btn");
  if (unlinkBtn) {
    unlinkBtn.addEventListener("click", async () => {
      if (!window.confirm(`Unlink ${a.account_id} from its Telegram chat? Nothing more is sent there until the borrower verifies again.`)) return;
      unlinkBtn.disabled = true;
      try {
        const r = await api(`/accounts/${a.account_id}/telegram/unlink`, { method: "POST" });
        toast(r.chat_told ? "Unlinked -- the chat was told." : "Unlinked.");
        await refreshAfterAction();
      } catch (err) {
        toast(err.message, true);
        unlinkBtn.disabled = false;
      }
    });
  }
  const editAccountBtn = document.getElementById("edit-account-btn");
  const editAccountForm = document.getElementById("edit-account-form");
  editAccountBtn.addEventListener("click", () => {
    editAccountForm.hidden = !editAccountForm.hidden;
  });
  document.getElementById("edit-account-cancel-btn").addEventListener("click", () => {
    editAccountForm.hidden = true;
  });
  editAccountForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const errorEl = document.getElementById("edit-account-error");
    errorEl.hidden = true;
    const payload = {
      phone_number: document.getElementById("ea-phone").value.trim(),
      language_preference: document.getElementById("ea-language").value,
      risk_tier: document.getElementById("ea-risk").value,
    };
    const submitBtn = editAccountForm.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.textContent = "Saving…";
    try {
      await updateAccount(a.account_id, payload);
      toast("Account updated.");
      await refreshAfterAction();
    } catch (err) {
      errorEl.textContent = err.message;
      errorEl.hidden = false;
      submitBtn.disabled = false;
      submitBtn.textContent = "Save changes";
    }
  });

  const telegramInviteBtn = document.getElementById("telegram-invite-btn");
  telegramInviteBtn.addEventListener("click", async () => {
    const resultEl = document.getElementById("telegram-invite-result");
    const originalText = telegramInviteBtn.textContent.trim();
    telegramInviteBtn.disabled = true;
    telegramInviteBtn.textContent = "Generating…";
    try {
      const result = await resetAccessKey(a.account_id);
      resultEl.innerHTML = result.telegram_invite_link
        ? `<a class="btn btn-primary na-invite-link" href="${escapeHtml(result.telegram_invite_link)}" target="_blank" rel="noopener">Open Telegram invite link</a>
           <p class="clarify-hint">New access key: <b>${escapeHtml(result.access_key)}</b> — the old one no longer works.</p>`
        : `<p class="clarify-hint">New access key: <b>${escapeHtml(result.access_key)}</b> — TELEGRAM_BOT_USERNAME isn't configured, so relay this key manually.</p>`;
      toast("New access key generated.");
    } catch (err) {
      toast(err.message, true);
    } finally {
      telegramInviteBtn.disabled = false;
      telegramInviteBtn.textContent = originalText;
    }
  });
}

/* ============================================================
   Voice -- ask the dashboard, or tell it to do something, by voice
   or typing (ops/voice.py on the server).

   The server answers with what to say, where the page should go
   (`screen`), and for an action a confirmation card (`proposal`):
   the exact request the dashboard's own form for that action sends.
   Nothing changes until the operator clicks Confirm, and Confirm
   sends that request through api() like any other button here --
   the same endpoint, the same checks, the same borrower message.

   Recording is the chat page's method (channels/static/app.js):
   MediaRecorder, then decoded and re-encoded to WAV here, because
   the server reads WAV with no codec dependency. Copied rather than
   shared: the two pages are separate apps with separate static
   folders.
   ============================================================ */
const $voiceForm = document.getElementById("voice-form");
const $voiceInput = document.getElementById("voice-input");
const $voiceMic = document.getElementById("voice-mic");
const $voiceAsk = document.getElementById("voice-ask");
const $voiceReply = document.getElementById("voice-reply");
const $voiceListening = document.getElementById("voice-listening");
const $voiceTimer = document.getElementById("voice-timer");
const $voiceSound = document.getElementById("voice-sound");
const $voiceRun = document.getElementById("voice-run");
const $voiceMenu = document.getElementById("voice-menu");

const VOICE_MAX_RECORDING_MS = 30000; // the server's limit for one question
const VOICE_SOUND_PREF_KEY = "bf_ops_voice_sound"; // on/off only -- no data
const VOICE_RECORDING_SUPPORTED = !!(
  navigator.mediaDevices && window.MediaRecorder && (window.AudioContext || window.webkitAudioContext)
);
if (!VOICE_RECORDING_SUPPORTED) $voiceMic.hidden = true;

const voice = {
  busy: false,
  recorder: null,
  stream: null,
  chunks: [],
  startedAt: 0,
  timerId: null,
  autoStopId: null,
  starting: false,
  audio: null, // the reply being spoken, if any
  audioUrl: null,
  soundOn: true,
  // A guided run in progress (call list, inbox): {kind, title, stepTool, items, index, done, skipped}. Kept here, in
  // this tab, not on the server: it is one operator's, and each step is read fresh from the server anyway.
  workflow: null,
};
try {
  voice.soundOn = localStorage.getItem(VOICE_SOUND_PREF_KEY) !== "off";
} catch (_) {
  /* storage blocked (private window): the choice is simply not remembered */
}

function renderVoiceSound() {
  $voiceSound.setAttribute("aria-pressed", String(voice.soundOn));
  $voiceSound.title = voice.soundOn ? "Spoken replies on (click to mute)" : "Spoken replies off (click to turn on)";
  $voiceSound.setAttribute("aria-label", $voiceSound.title);
}
renderVoiceSound();
$voiceSound.addEventListener("click", () => {
  voice.soundOn = !voice.soundOn;
  if (!voice.soundOn) stopVoicePlayback();
  try {
    localStorage.setItem(VOICE_SOUND_PREF_KEY, voice.soundOn ? "on" : "off");
  } catch (_) {
    /* not remembered, see above */
  }
  renderVoiceSound();
});

function voiceScreen() {
  const run = voice.workflow;
  const item = run ? run.items[run.index] : null;
  return {
    view: state.view,
    account_id: state.detailAccountId,
    workflow: item
      ? { kind: run.kind, position: run.index + 1, total: run.items.length, case_ref: run.kind === "inbox" ? item.key : null }
      : null,
  };
}

function setVoiceBusy(busy) {
  voice.busy = busy;
  $voiceAsk.disabled = busy;
  $voiceMic.disabled = busy;
  $voiceInput.disabled = busy;
}

function showVoiceThinking(text) {
  $voiceReply.hidden = false;
  $voiceReply.innerHTML = `<div class="voice-thinking"><div class="spinner"></div><span>${escapeHtml(text)}</span></div>`;
}

function closeVoiceReply() {
  stopVoicePlayback();
  $voiceReply.hidden = true;
  $voiceReply.innerHTML = "";
}

// --- asking ---
async function voiceRequest(path, payload, thinking) {
  if (voice.busy) return;
  stopVoicePlayback();
  setVoiceBusy(true);
  showVoiceThinking(thinking);
  let command = null;
  try {
    const reply = await api(path, payload);
    if (reply.workflow) startVoiceWorkflow(reply.workflow);
    renderVoiceReply(reply);
    applyVoiceScreen(reply.screen);
    command = reply.workflow_command;
    // "Next" says nothing itself: the next step speaks. "Stop" is said by the run's summary.
    if (!command) speakVoiceReply(reply.say);
  } catch (err) {
    renderVoiceError(err);
  } finally {
    setVoiceBusy(false);
  }
  if (command) runWorkflowCommand(command);
}

function askVoiceText(text) {
  return voiceRequest(
    "/voice/ask-text",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, screen: voiceScreen() }) },
    "Thinking…",
  );
}

function askVoiceRecording(wavBlob) {
  const form = new FormData();
  form.append("audio", wavBlob, "question.wav");
  form.append("screen", JSON.stringify(voiceScreen()));
  return voiceRequest("/voice/ask", { method: "POST", body: form }, "Listening back…");
}

function runVoiceChoice(choice, heard) {
  return voiceRequest(
    "/voice/plan",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tool: choice.tool, arguments: choice.arguments, heard: heard || "", screen: voiceScreen() }),
    },
    "Checking…",
  );
}

// A suggestion that needs a figure or words only the operator has puts its sentence in the box instead of sending it;
// the cursor goes where the missing part is (a double space marks it) or at the end.
function fillVoicePrompt(text) {
  $voiceInput.value = text;
  $voiceInput.focus();
  const gap = text.indexOf("  ");
  const at = gap >= 0 ? gap + 1 : text.length;
  $voiceInput.setSelectionRange(at, at);
}

$voiceMenu.addEventListener("click", () => runVoiceChoice({ tool: "help", arguments: {} }, ""));

$voiceForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $voiceInput.value.trim();
  if (!text || voice.busy) return;
  $voiceInput.value = "";
  askVoiceText(text);
});

// --- where the page goes ---
function applyVoiceScreen(screen) {
  if (!screen) return;
  if (screen.open_form === "new_account") {
    if (state.detailAccountId) closeDetail();
    openNewAccountModal();
    return;
  }
  if (screen.account_id) {
    const tab = screen.tab || "overview";
    if (state.detailAccountId === screen.account_id && !$detailPanel.hidden) {
      state.detailTab = tab;
      applyDetailTab();
    } else {
      openDetail(screen.account_id, { tab });
    }
    return;
  }
  const changesList = screen.view || screen.filter || screen.sort || screen.search !== null || screen.since_hours;
  if (changesList && state.detailAccountId) closeDetail();
  if (screen.view) setView(screen.view);
  if (screen.filter) state.activeFilter = screen.filter;
  if (screen.sort) {
    state.sortBy = screen.sort;
    document.getElementById("sort-select").value = screen.sort;
  }
  if (screen.search !== null && screen.search !== undefined) {
    state.search = screen.search;
    document.getElementById("search-input").value = screen.search;
  }
  if (screen.filter || screen.sort || screen.search !== null) {
    renderFilterPills();
    renderAccountGrid();
  }
  if (screen.since_hours && screen.since_hours !== state.sinceHours) {
    state.sinceHours = screen.since_hours;
    getMetrics(state.sinceHours)
      .then((metrics) => {
        state.metrics = metrics;
        renderKpis();
      })
      .catch((err) => toast(err.message, true));
  }
}

// --- guided runs (ops/voice_workflows.py) ---
function startVoiceWorkflow(w) {
  voice.workflow = { kind: w.kind, title: w.title, stepTool: w.step_tool, items: w.items, index: 0, done: 0, skipped: 0 };
  renderVoiceWorkflowBar();
}

function renderVoiceWorkflowBar() {
  const run = voice.workflow;
  $voiceRun.hidden = !run;
  if (!run) {
    $voiceRun.innerHTML = "";
    return;
  }
  const item = run.items[run.index];
  $voiceRun.innerHTML = `
    <span class="voice-run-title">${escapeHtml(run.title)}</span>
    <span class="voice-run-pos">${run.index + 1} of ${run.items.length}</span>
    <span class="voice-run-item">${escapeHtml(item ? item.label : "")}</span>
    <button type="button" class="btn btn-secondary voice-run-btn" data-run="skip">Skip</button>
    <button type="button" class="btn btn-secondary voice-run-btn" data-run="stop">Stop</button>`;
}

$voiceRun.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-run]");
  if (btn && !voice.busy) runWorkflowCommand(btn.dataset.run);
});

function runVoiceStep() {
  const run = voice.workflow;
  renderVoiceWorkflowBar();
  return runVoiceChoice({ tool: run.stepTool, arguments: run.items[run.index].arguments }, "");
}

function runWorkflowCommand(command) {
  const run = voice.workflow;
  if (!run) return;
  if (command === "stop") return finishVoiceWorkflow(true);
  if (command === "repeat") return runVoiceStep();
  if (command === "previous") {
    run.index = Math.max(0, run.index - 1);
    return runVoiceStep();
  }
  if (command === "skip") run.skipped += 1;
  run.index += 1; // next or skip
  if (run.index >= run.items.length) return finishVoiceWorkflow(false);
  return runVoiceStep();
}

function finishVoiceWorkflow(stopped) {
  const run = voice.workflow;
  voice.workflow = null;
  renderVoiceWorkflowBar();
  const noun = run.kind === "call_list" ? "borrower" : "case";
  const total = run.items.length;
  const say =
    `${run.title} ${stopped ? "stopped" : "done"}: ${run.done} of ${total} ${noun}${total === 1 ? "" : "s"} dealt with` +
    (run.skipped ? `, ${run.skipped} skipped.` : ".");
  renderVoiceReply({
    kind: "answer", heard: null, title: null, say, rows: [], choices: [], proposal: null,
    suggestions: [{ label: "Start my day", tool: "daily_briefing", arguments: {} }],
  });
  speakVoiceReply(say);
}

// --- the reply card ---
function voiceRowHtml(r, i) {
  const href = r.href && r.href.startsWith("tel:") ? r.href : null; // the server only ever sends tel: links
  const clickable = !href && (r.tool || r.account_id);
  const tag = href ? "a" : clickable ? "button" : "div";
  const attrs = href ? `href="${escapeHtml(href)}"` : clickable ? `type="button" data-voice-row="${i}"` : "";
  return `<li><${tag} class="voice-row ${r.tone ? `tone-${r.tone}` : ""} ${href || clickable ? "clickable" : ""}" ${attrs}>
      <span class="voice-row-main"><span class="voice-row-label">${escapeHtml(r.label)}</span>${
        r.detail ? `<span class="voice-row-detail">${escapeHtml(r.detail)}</span>` : ""
      }</span>${r.value ? `<span class="voice-row-value">${escapeHtml(r.value)}</span>` : ""}
    </${tag}></li>`;
}

function voiceSuggestionsHtml(list) {
  if (!list.length) return "";
  const groups = [];
  list.forEach((sug, i) => {
    let group = groups.find((g) => g.name === (sug.group || ""));
    if (!group) groups.push((group = { name: sug.group || "", items: [] }));
    group.items.push([sug, i]);
  });
  return `<div class="voice-suggestions">${groups
    .map(
      (g) => `${g.name ? `<p class="voice-group">${escapeHtml(g.name)}</p>` : ""}<div class="voice-chips">${g.items
        .map(([sug, i]) => `<button type="button" class="voice-chip ${sug.prompt ? "prompt" : ""}" data-voice-suggestion="${i}">${escapeHtml(sug.label)}</button>`)
        .join("")}</div>`,
    )
    .join("")}</div>`;
}

function voiceHeardHtml(reply) {
  return reply.heard ? `<p class="voice-heard">Heard: <q>${escapeHtml(reply.heard)}</q></p>` : `<p class="voice-heard"></p>`;
}

function renderVoiceReply(reply) {
  $voiceReply.hidden = false;
  const choices = (reply.choices || [])
    .map((c, i) => `<button type="button" class="btn btn-secondary" data-voice-choice="${i}">${escapeHtml(c.label)}</button>`)
    .join("");
  const rows = (reply.rows || []).map(voiceRowHtml).join("");
  $voiceReply.innerHTML = `
    <div class="voice-reply-top">${voiceHeardHtml(reply)}${DETAIL_CLOSE_HTML.replace('class="detail-close"', 'class="detail-close voice-reply-close"')}</div>
    ${reply.title ? `<p class="voice-title">${escapeHtml(reply.title)}</p>` : ""}
    <p class="voice-say ${reply.kind === "clarify" ? "clarify" : ""}">${escapeHtml(reply.say)}</p>
    ${rows ? `<ul class="voice-rows">${rows}</ul>` : ""}
    ${choices ? `<div class="voice-choices">${choices}</div>` : ""}
    ${reply.proposal ? voiceProposalHtml(reply.proposal) : ""}
    ${voiceSuggestionsHtml(reply.suggestions || [])}`;
  $voiceReply.querySelector(".voice-reply-close").addEventListener("click", closeVoiceReply);
  $voiceReply.querySelectorAll("[data-voice-choice]").forEach((btn) =>
    btn.addEventListener("click", () => runVoiceChoice(reply.choices[Number(btn.dataset.voiceChoice)], reply.heard)),
  );
  // A row opens what it is about: the question it stands for, or the borrower's profile.
  $voiceReply.querySelectorAll("[data-voice-row]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const row = reply.rows[Number(btn.dataset.voiceRow)];
      if (row.tool) runVoiceChoice({ tool: row.tool, arguments: row.arguments || {} }, "");
      else applyVoiceScreen({ account_id: row.account_id, tab: row.tab || "overview" });
    }),
  );
  $voiceReply.querySelectorAll("[data-voice-suggestion]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const sug = reply.suggestions[Number(btn.dataset.voiceSuggestion)];
      if (sug.prompt) fillVoicePrompt(sug.prompt);
      else runVoiceChoice({ tool: sug.tool, arguments: sug.arguments || {} }, "");
    }),
  );
  if (reply.proposal) wireVoiceProposal(reply.proposal);
}

function renderVoiceError(err) {
  $voiceReply.hidden = false;
  const message = err instanceof ApiError ? err.message : "Couldn't reach the dashboard -- check your connection and try again.";
  $voiceReply.innerHTML = `
    <div class="voice-reply-top"><p class="voice-heard"></p>${DETAIL_CLOSE_HTML.replace('class="detail-close"', 'class="detail-close voice-reply-close"')}</div>
    <p class="voice-error">${escapeHtml(message)}</p>`;
  $voiceReply.querySelector(".voice-reply-close").addEventListener("click", closeVoiceReply);
}

function voiceProposalHtml(p) {
  const details = p.details.map((d) => `<dt>${escapeHtml(d.label)}</dt><dd>${escapeHtml(d.value)}</dd>`).join("");
  const warnings = (p.warnings || []).map((w) => `<p class="voice-warning">${escapeHtml(w)}</p>`).join("");
  const told = p.borrower_told
    ? `<div class="voice-told"><b>The borrower will be sent</b><span data-voice-told>${escapeHtml(p.borrower_told)}</span></div>`
    : "";
  const editValue = p.editable && p.body ? p.body[p.editable] || "" : "";
  const edit = p.editable
    ? `<label class="voice-edit">${escapeHtml(p.editable_label || "Text")}<textarea rows="3" data-voice-edit>${escapeHtml(editValue)}</textarea></label>`
    : "";
  const credit = p.credit_choice
    ? `<div class="voice-credit" role="radiogroup" aria-label="Credit the shortfall">
         <p>This is less than what's due this cycle. Apply it toward the next EMI?</p>
         <label><input type="radio" name="voice-credit" value="true" /> Yes, credit it</label>
         <label><input type="radio" name="voice-credit" value="false" /> No, just record it</label>
       </div>`
    : "";
  return `
    <div class="voice-card">
      <p class="voice-card-title">${escapeHtml(p.title)}</p>
      <dl class="voice-details">${details}</dl>
      ${warnings}${told}${edit}${credit}
      <p class="voice-error" data-voice-error hidden></p>
      <div class="voice-actions">
        <button type="button" class="btn btn-primary" data-voice-confirm>Confirm</button>
        <button type="button" class="btn btn-secondary" data-voice-cancel>Cancel</button>
      </div>
    </div>`;
}

function wireVoiceProposal(p) {
  const card = $voiceReply.querySelector(".voice-card");
  const confirmBtn = card.querySelector("[data-voice-confirm]");
  const editEl = card.querySelector("[data-voice-edit]");
  const errorEl = card.querySelector("[data-voice-error]");
  const creditChosen = () => card.querySelector('input[name="voice-credit"]:checked');
  // Confirm is only clickable once the card is complete: a required message isn't blank, and a short payment says
  // whether to credit it.
  // When the text being edited IS what the borrower is sent (a message, a reply), the preview follows the edit.
  const toldEl = p.borrower_told && p.body && p.borrower_told === p.body[p.editable] ? card.querySelector("[data-voice-told]") : null;
  const refresh = () => {
    confirmBtn.disabled = (p.editable_required && editEl && !editEl.value.trim()) || (p.credit_choice && !creditChosen());
    if (toldEl) toldEl.textContent = editEl.value;
  };
  editEl?.addEventListener("input", refresh);
  card.querySelectorAll('input[name="voice-credit"]').forEach((r) => r.addEventListener("change", refresh));
  refresh();

  card.querySelector("[data-voice-cancel]").addEventListener("click", () => {
    stopVoicePlayback();
    card.innerHTML = `<p class="voice-note">Cancelled -- nothing was changed.</p>`;
  });

  confirmBtn.addEventListener("click", async () => {
    const body = p.body ? { ...p.body } : null;
    if (editEl && body) body[p.editable] = editEl.value.trim() || null;
    if (p.credit_choice && body) body.apply_extra_to_next = creditChosen().value === "true";
    confirmBtn.disabled = true;
    errorEl.hidden = true;
    stopVoicePlayback();
    try {
      const result = await api(p.path, {
        method: p.method,
        ...(body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
      });
      // Then the card's follow-up requests, in order (a promise made on the call just logged). One failing stops the
      // rest and says which: what was done before it stays done, as with the profile's own forms.
      let followUpError = null;
      for (const req of p.then || []) {
        try {
          await api(req.path, {
            method: req.method,
            ...(req.body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(req.body) } : {}),
          });
        } catch (err) {
          followUpError = `${req.label} failed: ${err instanceof ApiError ? err.message : "couldn't reach the dashboard"}. The rest was done; add it from the profile.`;
          break;
        }
      }
      const done = voiceDoneText(p, result);
      card.innerHTML = `<p class="voice-done">${escapeHtml(done)}</p>${followUpError ? `<p class="voice-error">${escapeHtml(followUpError)}</p>` : ""}`;
      toast(followUpError || done, Boolean(followUpError));
      // In a call list or inbox run, an action on the item on screen moves the run on to the next one.
      const run = voice.workflow;
      const item = run ? run.items[run.index] : null;
      if (item && !followUpError && p.account_id && item.arguments.account === p.account_id) {
        run.done += 1;
        loadAll().catch(() => {}); // the dashboard behind catches up; the next step reads its own data fresh
        runWorkflowCommand("next");
        return;
      }
      speakVoiceReply("Done.");
      if (state.detailAccountId) {
        const [fresh, freshDocuments, freshConversation] = await _loadDetailData(state.detailAccountId);
        renderDetail(fresh, freshDocuments, freshConversation);
      }
      await loadAll();
    } catch (err) {
      // e.g. a 409: a colleague closed it first, or a payment looks like a duplicate -- the server's own words.
      errorEl.textContent = err instanceof ApiError ? err.message : "Couldn't reach the dashboard -- nothing was changed.";
      errorEl.hidden = false;
      confirmBtn.disabled = false;
    }
  });
}

function voiceDoneText(p, result) {
  if (result && typeof result.borrower_notified === "boolean") return `Done: ${p.title} -- ${notifiedText(result.borrower_notified)}.`;
  if (result && typeof result.delivered_via_telegram === "boolean") return `Done: ${p.title} -- ${notifiedText(result.delivered_via_telegram)}.`;
  if (result && Array.isArray(result.sent_to)) return `Done: sent to ${result.sent_to.length}, ${result.delivered_via_telegram.length} over Telegram.`;
  if (result && Array.isArray(result.reminders_sent)) return `Done: ${result.reminders_sent.length} reminder${result.reminders_sent.length === 1 ? "" : "s"} sent.`;
  return `Done: ${p.title}.`;
}

// --- spoken replies ---
// Each reply's speech belongs to one generation; stopping or starting another moves the generation on. Found in the
// preview: tapping on before a reply had finished loading or playing cut it off, and the cut-off playback was then
// reported on the NEXT card as "spoken reply unavailable". A superseded reply now just goes quiet.
let speechGeneration = 0;

function stopVoicePlayback() {
  speechGeneration += 1;
  if (voice.audio) voice.audio.pause();
  if (voice.audioUrl) URL.revokeObjectURL(voice.audioUrl);
  voice.audio = null;
  voice.audioUrl = null;
}

async function speakVoiceReply(text) {
  if (!voice.soundOn || !text) return;
  stopVoicePlayback();
  const mine = speechGeneration;
  try {
    const res = await fetch("/voice/speak", {
      method: "POST",
      headers: { "X-API-Key": state.apiKey, "Content-Type": "application/json" },
      body: JSON.stringify({ text: text.slice(0, 400) }),
    });
    if (!res.ok) throw new Error(`speak failed (${res.status})`);
    const blob = await res.blob();
    if (mine !== speechGeneration) return; // a newer reply, or a stop, came while this loaded
    const url = URL.createObjectURL(blob);
    voice.audioUrl = url;
    voice.audio = new Audio(url);
    voice.audio.addEventListener("ended", stopVoicePlayback);
    await voice.audio.play();
  } catch (err) {
    if (mine !== speechGeneration) return; // cut short on purpose, not a failure
    // The answer is already on screen; only its sound is missing. Said in the card, never a blocking error.
    console.warn("ops voice: the spoken reply could not be played", err);
    if (!$voiceReply.hidden && !$voiceReply.querySelector("[data-voice-silent]")) {
      $voiceReply.insertAdjacentHTML("beforeend", `<p class="voice-note" data-voice-silent>Spoken reply unavailable right now -- the answer is above.</p>`);
    }
  }
}

// --- recording ---
function audioBufferToWav(buffer) {
  // channels/static/app.js's audioBufferToWav: mono, 16-bit PCM WAV.
  const left = buffer.getChannelData(0);
  let samples = left;
  if (buffer.numberOfChannels > 1) {
    const right = buffer.getChannelData(1);
    samples = new Float32Array(left.length);
    for (let i = 0; i < left.length; i++) samples[i] = (left[i] + right[i]) / 2;
  }
  const dataSize = samples.length * 2;
  const view = new DataView(new ArrayBuffer(44 + dataSize));
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view.buffer], { type: "audio/wav" });
}

function setVoiceRecordingUi(recording) {
  $voiceMic.classList.toggle("recording", recording);
  $voiceMic.title = recording ? "Stop and ask" : "Ask by voice (tap again to stop)";
  $voiceMic.setAttribute("aria-label", recording ? "Stop recording and ask" : "Ask by voice");
  $voiceListening.hidden = !recording;
  $voiceInput.hidden = recording;
  $voiceAsk.hidden = recording;
}

// A fast double-tap must not start two recordings (the chat page found that live): guarded until getUserMedia settles.
async function startVoiceRecording() {
  if (voice.busy || voice.starting) return;
  voice.starting = true;
  try {
    voice.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
  } catch (err) {
    renderVoiceError(new ApiError(0, "Couldn't use the microphone -- allow microphone access for this site, or type the question."));
    return;
  } finally {
    voice.starting = false;
  }
  stopVoicePlayback();
  voice.chunks = [];
  voice.recorder = new MediaRecorder(voice.stream);
  voice.recorder.addEventListener("dataavailable", (e) => {
    if (e.data.size > 0) voice.chunks.push(e.data);
  });
  voice.recorder.addEventListener("stop", onVoiceRecordingStopped);
  voice.recorder.start();
  voice.startedAt = Date.now();
  $voiceTimer.textContent = "0:00";
  voice.timerId = setInterval(() => {
    const seconds = Math.floor((Date.now() - voice.startedAt) / 1000);
    $voiceTimer.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }, 250);
  voice.autoStopId = setTimeout(stopVoiceRecording, VOICE_MAX_RECORDING_MS);
  setVoiceRecordingUi(true);
}

// Safe to call when nothing is recording (lock() and closing the page call it unconditionally).
function stopVoiceRecording() {
  if (voice.recorder && voice.recorder.state !== "inactive") voice.recorder.stop();
  clearInterval(voice.timerId);
  clearTimeout(voice.autoStopId);
  voice.stream?.getTracks().forEach((t) => t.stop());
  voice.stream = null;
  setVoiceRecordingUi(false);
}

async function onVoiceRecordingStopped() {
  const blob = new Blob(voice.chunks, { type: voice.recorder.mimeType || "audio/webm" });
  const durationMs = Date.now() - voice.startedAt;
  voice.chunks = [];
  if (blob.size === 0 || durationMs < 400) return; // an accidental tap, not a question
  let audioCtx;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const wav = audioBufferToWav(await audioCtx.decodeAudioData(await blob.arrayBuffer()));
    await askVoiceRecording(wav);
  } catch (err) {
    renderVoiceError(new ApiError(0, "Couldn't process that recording -- please try again, or type the question."));
  } finally {
    audioCtx?.close();
  }
}

if (VOICE_RECORDING_SUPPORTED) {
  $voiceMic.addEventListener("click", () => {
    if (voice.recorder && voice.recorder.state === "recording") stopVoiceRecording();
    else startVoiceRecording();
  });
}
window.addEventListener("pagehide", () => {
  stopVoiceRecording();
  stopVoicePlayback();
});

/* ============================================================
   Boot / refresh
   ============================================================ */

let demoResetChecked = false;

async function loadAll() {
  const myGeneration = ++loadGeneration;
  // The escalation-rate figure is fetched alongside but never holds up the portfolio: accounts and escalations render
  // as soon as they arrive, and the rate's card fills in when its own request returns.
  const metricsRequest = getMetrics(state.sinceHours);
  const [accounts, inbox] = await Promise.all([getAccounts(), getInbox()]);
  if (myGeneration !== loadGeneration) return; // superseded by a newer load -- discard
  state.accounts = accounts;
  state.inbox = inbox;
  renderKpis();
  renderFilterPills();
  renderAccountGrid();
  renderInbox();
  loadBank(); // the Bank tab's waiting count, kept current with the rest
  loadHealth();
  if (!demoResetChecked) {
    demoResetChecked = true;
    initDemoReset();
  }
  metricsRequest
    .then((metrics) => {
      if (myGeneration !== loadGeneration) return;
      state.metrics = metrics;
      state.metricsFailed = false;
      renderKpis();
    })
    .catch((err) => {
      if (myGeneration !== loadGeneration) return;
      console.warn("ops: the escalation rate could not be loaded; the next refresh retries it", err);
      state.metricsFailed = true; // the card says so, rather than "loading…" forever
      renderKpis();
    });
}

function stopAutoRefresh() {
  if (state.refreshTimer) clearInterval(state.refreshTimer);
  state.refreshTimer = null;
}

function startAutoRefresh() {
  stopAutoRefresh();
  state.refreshTimer = setInterval(() => {
    if (state.detailAccountId) return; // don't yank the panel out from under an operator mid-review
    loadAll().catch(() => {});
  }, 30000);
}

/* ============================================================
   Entry
   ============================================================ */
(function init() {
  // Arriving from the shared login page's "Ops team" box, which redirects
  // here as `#key=...` rather than a query string -- the fragment never
  // gets sent to (or logged by) any server, unlike a query param would.
  // Stripped from the URL immediately either way, so it doesn't linger in
  // the address bar or browser history a moment longer than this load.
  const hashMatch = location.hash.match(/^#key=(.+)$/);
  const incomingKey = hashMatch ? decodeURIComponent(hashMatch[1]) : null;
  if (hashMatch) history.replaceState(null, "", location.pathname + location.search);

  const keyToTry = incomingKey || state.apiKey;
  if (keyToTry) signIn(keyToTry);
  else showUnlock();
})();

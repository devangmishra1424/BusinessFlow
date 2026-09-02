"use strict";

/* ============================================================
   State
   ============================================================ */
const state = {
  apiKey: localStorage.getItem("bf_ops_key") || "",
  view: "overview",
  accounts: [],
  metrics: null,
  escalations: [],
  activeFilter: "all",
  sortBy: "urgency", // "urgency" | "risk" | "id" -- see filteredAccounts()
  search: "",
  sinceHours: 24,
  detailAccountId: null,
  refreshTimer: null,
};

const FLAG_LABELS = {
  overdue: "Overdue",
  disputed: "Disputed",
  broken_promises: "Broken promises",
};

const LANGUAGE_LABELS = { hi: "Hindi", en: "English", hinglish: "Hinglish" };

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
  constructor(status, detail) {
    super(detail || `request failed (${status})`);
    this.status = status;
  }
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: { "X-API-Key": state.apiKey, ...(opts.headers || {}) },
  });
  if (res.status === 401) {
    lock("That key was rejected by the server.");
    throw new ApiError(401, "unauthorized");
  }
  if (!res.ok) {
    let detail = `request failed (${res.status})`;
    try {
      const body = await res.json();
      // FastAPI's own validation errors (422s) send detail as a LIST of
      // {loc, msg, type} objects, not a plain string -- passing that
      // straight into an Error's message stringifies to "[object
      // Object]" instead of anything readable. Every other error path
      // (ValueError, HTTPException(detail=str)) already sends a string.
      if (Array.isArray(body.detail)) {
        detail = body.detail.map((e) => e.msg || JSON.stringify(e)).join("; ");
      } else if (body.detail) {
        detail = body.detail;
      }
    } catch (_) {}
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

const getAccounts = (flag) => api(`/accounts${flag ? `?flag=${flag}` : ""}`);
const getAccount = (id) => api(`/accounts/${id}`);
const getEscalations = () => api("/escalations");
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
const resolveDispute = (accountId) => api(`/accounts/${accountId}/disputes/resolve`, { method: "POST" });
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
  $shell.hidden = true;
  $unlockScreen.hidden = false;
  $unlockInput.value = "";
  if (message) {
    $unlockError.textContent = message;
    $unlockError.hidden = false;
  }
  $unlockInput.focus();
}

async function tryUnlock(key) {
  state.apiKey = key;
  try {
    await api("/accounts");
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) {
      $unlockError.textContent = "That key was rejected — check it and try again.";
      $unlockError.hidden = false;
      state.apiKey = "";
      return false;
    }
    $unlockError.textContent = "Couldn't reach the ops API. Is it running?";
    $unlockError.hidden = false;
    state.apiKey = "";
    return false;
  }
  localStorage.setItem("bf_ops_key", key);
  $unlockError.hidden = true;
  $unlockScreen.hidden = true;
  $shell.hidden = false;
  boot();
  return true;
}

$unlockForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const key = $unlockInput.value.trim();
  if (key) tryUnlock(key);
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
  document.getElementById("view-escalations").hidden = view !== "escalations";
}

/* ============================================================
   Since-hours selector
   ============================================================ */
document.getElementById("since-hours").addEventListener("change", (e) => {
  state.sinceHours = Number(e.target.value);
  loadAll().catch((err) => toast(err.message, true));
});

/* ============================================================
   Avatar helpers
   ============================================================ */
const AVATAR_GRADIENTS = [
  ["#7bbde8", "#49769f"],
  ["#6ea2b3", "#0a4174"],
  ["#bdd8e9", "#4e8ea2"],
  ["#4e8ea2", "#001d39"],
  ["#7bbde8", "#4e8ea2"],
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
  const [a, b] = AVATAR_GRADIENTS[hashStr(seed) % AVATAR_GRADIENTS.length];
  return `background: linear-gradient(150deg, ${a}, ${b});`;
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
function renderPortfolioSummary(accounts, openEscalations) {
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
  if (openEscalations > 0) {
    parts.push(`${openEscalations} waiting on a human`);
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
  const openEscalations = state.escalations.filter((e) => e.status === "queued_for_human").length;
  renderPortfolioSummary(accounts, openEscalations);
  const rate = state.metrics ? Math.round(state.metrics.escalation_rate * 100) : null;

  const riskCounts = { low: 0, medium: 0, high: 0 };
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
      cls: openEscalations ? "warn" : "ok",
      label: "Awaiting a human",
      value: openEscalations,
      sub: "open escalations",
      action: "escalations",
    },
    {
      icon: iconPulse(),
      cls: "",
      label: `Escalation rate · ${SINCE_HOURS_LABELS[state.sinceHours] || `${state.sinceHours}h`}`,
      value: rate === null ? "–" : `${rate}%`,
      sub: state.metrics ? `${Object.values(state.metrics.event_counts).reduce((a, b) => a + b, 0)} events logged` : "loading…",
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
      valueHtml: `<span class="kpi-risk-mix"><span class="risk-low">${riskCounts.low}L</span><span class="risk-medium">${riskCounts.medium}M</span><span class="risk-high">${riskCounts.high}H</span></span>`,
      sub: "low · medium · high risk",
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
        <span class="kpi-label">${escapeHtml(c.label)}</span>
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
      if (action === "escalations") {
        setView("escalations");
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

  const badge = document.getElementById("escalation-count-badge");
  if (openEscalations > 0) {
    badge.hidden = false;
    badge.textContent = openEscalations;
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
function _accountSortComparator() {
  if (state.sortBy === "risk") {
    const rank = { high: 0, medium: 1, low: 2 };
    return (a, b) => (rank[a.risk_tier] - rank[b.risk_tier]) || (b.days_past_due - a.days_past_due);
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
  if (state.activeFilter !== "all") {
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
// scoped to whatever the current filter/search shows, not blindly
// everyone, so "send reminders to just my overdue accounts" is one click.
// A native confirm(), not just a disabled-state guard, since this sends
// real messages to real borrowers -- the one action in this whole
// dashboard where a stray click has an external, unrecallable effect.
document.getElementById("send-reminders-btn").addEventListener("click", async () => {
  const targets = filteredAccounts().map((a) => a.account_id);
  if (!targets.length) {
    toast("No accounts match the current filter.", true);
    return;
  }
  if (!confirm(`Send today's reminders to ${targets.length} account(s) matching the current filter?`)) return;

  const btn = document.getElementById("send-reminders-btn");
  btn.disabled = true;
  btn.textContent = "Sending…";
  try {
    const result = await triggerOutboundRun(targets);
    toast(`${result.reminders_sent.length} reminder(s) sent, ${result.escalated_for_broken_promises} escalated.`);
    await loadAll();
  } catch (err) {
    toast(err.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Send reminders now";
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
}

function accountCardHtml(a) {
  const flagsHtml = a.flags.length
    ? a.flags.map((f) => `<span class="flag-chip ${f.label}" title="${escapeHtml(f.reason)}">${escapeHtml(FLAG_LABELS[f.label] || f.label)}</span>`).join("")
    : `<span class="flag-chip clean">Clean</span>`;

  return `
  <button class="account-card" data-id="${a.account_id}">
    <div class="account-top">
      <div class="avatar" style="${avatarStyle(a.account_id)}">${initials(a.borrower_name)}</div>
      <div class="account-names">
        <div class="account-borrower">${escapeHtml(a.borrower_name)}</div>
        <div class="account-business">${escapeHtml(a.business_name)}</div>
      </div>
      <span class="risk-dot ${a.risk_tier}" title="${a.risk_tier} risk"></span>
    </div>
    <div class="account-meta-row">
      <span class="account-loan-type">${escapeHtml(a.loan_type)}</span>
      <div class="account-emi">
        <div class="amount">${fmtInr(a.emi_amount)}</div>
        <div class="label">EMI</div>
      </div>
    </div>
    <div class="account-flags">${flagsHtml}</div>
    <div class="account-footer">
      <span>${a.account_id}</span>
      <span class="dpd ${a.days_past_due > 0 ? "overdue" : ""}">${a.days_past_due > 0 ? `${a.days_past_due}d past due` : "current"}</span>
    </div>
  </button>`;
}

document.getElementById("search-input").addEventListener("input", (e) => {
  state.search = e.target.value;
  renderAccountGrid();
});

/* ============================================================
   Escalation queue view
   ============================================================ */
// An escalation open 20 minutes and one open 5 days are very different
// problems -- the latter is stalling a real restructuring request or
// dispute. Bucketed so staleness is visible without opening the tab and
// reading every card's relative time by hand.
function renderSlaBuckets() {
  const open = state.escalations.filter((e) => e.status === "queued_for_human");
  const container = document.getElementById("sla-buckets");
  if (!open.length) {
    container.innerHTML = "";
    return;
  }
  const now = Date.now();
  const buckets = { fresh: 0, aging: 0, stale: 0 };
  open.forEach((e) => {
    const hours = (now - new Date(e.created_at).getTime()) / 3_600_000;
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

function renderEscalationList() {
  renderSlaBuckets();
  const list = state.escalations.slice().sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
  const container = document.getElementById("escalation-list");
  document.getElementById("escalation-empty").hidden = list.length !== 0;
  container.innerHTML = list.map((e) => escalationCardHtml(e, { withAccountLink: true })).join("");

  container.querySelectorAll(".escalation-account").forEach((el) =>
    el.addEventListener("click", () => openDetail(el.dataset.id))
  );
  // Full reload, same reasoning as the detail panel's own escalation
  // actions -- an approval/rejection can change the account's own fields,
  // not just the escalation's status.
  wireEscalationActions(container, () => loadAll());
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

function escalationCardHtml(e, { withAccountLink }) {
  const isOpen = e.status === "queued_for_human";
  return `
  <div class="escalation-card" data-escalation-id="${e.escalation_id}">
    <div class="escalation-main">
      <div class="escalation-top">
        ${withAccountLink ? `<span class="escalation-account" data-id="${e.account_id}">${e.account_id}</span>` : ""}
        <span class="status-pill ${e.status}">${e.status.replace(/_/g, " ")}</span>
        <span class="escalation-time">${relativeTime(e.created_at)}</span>
      </div>
      <div class="escalation-reason">${escapeHtml(e.reason)}</div>
      ${proposedTermsHtml(e.proposed_changes)}
      ${e.resolution_reason ? `<div class="escalation-reason" style="margin-top:6px;color:var(--text-3)">Reason given: ${escapeHtml(e.resolution_reason)}</div>` : ""}
      <div class="reject-reason-row" id="reject-row-${e.escalation_id}">
        <input type="text" placeholder="Optional reason shown to the borrower" id="reject-input-${e.escalation_id}" />
        <button class="btn btn-reject confirm-reject" data-id="${e.escalation_id}">Confirm reject</button>
      </div>
    </div>
    ${
      isOpen
        ? `<div class="escalation-actions">
            <button class="btn btn-approve" data-action="approve" data-id="${e.escalation_id}">Approve</button>
            <button class="btn btn-reject" data-action="reject" data-id="${e.escalation_id}">Reject</button>
          </div>`
        : ""
    }
  </div>`;
}

function wireEscalationActions(root, onDone) {
  root.querySelectorAll('[data-action="approve"]').forEach((btn) =>
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        await approveEscalation(btn.dataset.id);
        toast("Escalation approved — borrower notified.");
        await onDone();
      } catch (e) {
        toast(e.message, true);
        btn.disabled = false;
      }
    })
  );
  root.querySelectorAll('[data-action="reject"]').forEach((btn) =>
    btn.addEventListener("click", () => {
      document.getElementById(`reject-row-${btn.dataset.id}`).classList.add("open");
    })
  );
  root.querySelectorAll(".confirm-reject").forEach((btn) =>
    btn.addEventListener("click", async () => {
      const id = btn.dataset.id;
      const reason = document.getElementById(`reject-input-${id}`).value.trim();
      btn.disabled = true;
      try {
        await rejectEscalation(id, reason);
        toast("Escalation rejected — borrower notified.");
        await onDone();
      } catch (e) {
        toast(e.message, true);
        btn.disabled = false;
      }
    })
  );
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
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="rgba(123,189,232,0.14)" stroke-width="${stroke}" />
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

async function openDetail(accountId) {
  state.detailAccountId = accountId;
  $detailPanel.hidden = false;
  $detailBackdrop.hidden = false;
  setTimeout(() => {
    $detailPanel.classList.add("open");
    $detailBackdrop.classList.add("open");
  }, 10);
  $detailPanel.innerHTML = `<div class="detail-close">✕</div><div class="detail-body"><p class="no-data">Loading…</p></div>`;
  wireDetailClose();

  try {
    const [account, documents, conversation] = await _loadDetailData(accountId);
    renderDetail(account, documents, conversation);
  } catch (e) {
    $detailPanel.innerHTML = `<div class="detail-close">✕</div><div class="detail-body"><p class="no-data">Couldn't load this account: ${escapeHtml(e.message)}</p></div>`;
    wireDetailClose();
  }
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
    naReviewRow("Risk tier", payload.risk_tier),
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
      ? `<a class="btn btn-approve na-invite-link" href="${escapeHtml(result.telegram_invite_link)}" target="_blank" rel="noopener">Open Telegram invite link</a>
         <p class="clarify-hint">Send this link to the borrower, or tap it now — it verifies them the instant it's opened.</p>`
      : `<p class="clarify-hint">This key won't be shown again — record it now if the borrower needs it to verify over chat/Telegram.</p>`;
    $newAccountSuccess.innerHTML = `
      <p class="na-success-title">✓ ${escapeHtml(result.account.account_id)} opened for ${escapeHtml(payload.borrower_name)}</p>
      <p class="na-success-key">Access key: <b>${escapeHtml(result.access_key)}</b></p>
      ${inviteHtml}
      <button class="btn ${result.telegram_invite_link ? "" : "btn-approve"}" id="na-done-btn">Done</button>`;
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
    `${a.tenure_months - a.months_remaining} of ${a.tenure_months} months paid down`,
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
    <button type="button" class="btn document-download-btn" data-filename="${escapeHtml(d.filename)}">Download</button>
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
  const paidOffFraction = a.tenure_months > 0 ? (a.tenure_months - a.months_remaining) / a.tenure_months : 0;
  const ringColor = a.risk_tier === "high" ? "var(--danger)" : a.risk_tier === "medium" ? "var(--warning)" : "var(--success)";
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

  const clarificationHistoryHtml = a.clarification_requests.length
    ? a.clarification_requests
        .map(
          (c) => `<div class="clarification-entry">
            <div class="clarification-entry-top">
              <span class="delivery-badge ${c.delivered_via_telegram ? "delivered" : "not-delivered"}">
                ${c.delivered_via_telegram ? "Delivered via Telegram" : "Not delivered — no linked Telegram"}
              </span>
              <span class="escalation-time">${relativeTime(c.created_at)}</span>
            </div>
            <div class="clarification-entry-message">${escapeHtml(c.message)}</div>
          </div>`
        )
        .join("")
    : `<p class="no-data">No clarification requests sent yet.</p>`;

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
      return { date: p.date, amount: p.amount, status: p.on_time ? "paid-on-time" : "paid-late", label: baseLabel };
    });

    const upcoming = [];
    if (account.months_remaining > 0 && account.emi_due_date) {
      const cursor = new Date(`${account.emi_due_date}T00:00:00`);
      const credit = account.pending_emi_credit || 0;
      for (let i = 0; i < account.months_remaining; i++) {
        const isNextDue = i === 0;
        const overdue = isNextDue && account.days_past_due > 0;
        let label = overdue ? `Overdue — ${account.days_past_due}d past due` : "Scheduled";
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

  const promisesHtml = a.promises.length
    ? a.promises
        .slice()
        .reverse()
        .map((p) => {
          const status = p.kept === null ? "pending" : p.kept ? "kept" : "broken";
          const icon = status === "kept" ? "✓" : status === "broken" ? "✕" : "…";
          const statusText = status === "kept" ? "kept" : status === "broken" ? "broken" : "pending";
          return `<div class="promise-row">
            <div class="promise-dot ${status}">${icon}</div>
            <div class="promise-info">
              <b>${fmtInr(p.promised_amount)}</b> promised for ${fmtDate(p.promised_date)}
              <div class="promise-meta">made ${fmtDate(p.made_on)} · ${statusText}</div>
            </div>
          </div>`;
        })
        .join("")
    : `<p class="no-data">No promises to pay logged.</p>`;

  // The real reason text behind THIS account's dispute(s) -- previously
  // invisible anywhere in the ops UI (the flags list only ever showed a
  // fixed generic "has an open, unresolved dispute" string; see
  // flags.py). Matters directly for the client dashboard's "Contest"
  // warning action: an operator opening this profile needs to see WHAT
  // was actually contested, not just that something was.
  const disputesHtml = a.disputes.length
    ? a.disputes
        .map(
          (d) => `<div class="promise-row">
            <div class="promise-dot ${d.status === "open" ? "broken" : "kept"}">${d.status === "open" ? "!" : "✓"}</div>
            <div class="promise-info">
              ${escapeHtml(d.reason)}
              <div class="promise-meta">opened ${fmtDate(d.opened_at.slice(0, 10))} · ${d.status}${d.resolved_at ? ` ${fmtDate(d.resolved_at.slice(0, 10))}` : ""}</div>
            </div>
            ${d.status === "open" ? `<button type="button" class="btn resolve-dispute-btn">Resolve</button>` : ""}
          </div>`
        )
        .join("")
    : `<p class="no-data">No disputes on this account.</p>`;

  const callLogHtml = a.call_log.length
    ? a.call_log
        .map(
          (c) => `<div class="promise-row">
            <div class="promise-dot ${c.outcome === "reached" ? "kept" : "pending"}">${c.outcome === "reached" ? "✓" : "…"}</div>
            <div class="promise-info">
              ${escapeHtml(CALL_OUTCOME_LABELS[c.outcome] || c.outcome)}${c.note ? ` — ${escapeHtml(c.note)}` : ""}
              <div class="promise-meta">${relativeTime(c.created_at)}</div>
            </div>
          </div>`
        )
        .join("")
    : `<p class="no-data">No calls logged yet.</p>`;

  const escalationsHtml = a.escalations.length
    ? a.escalations
        .slice()
        .sort((x, y) => new Date(y.created_at) - new Date(x.created_at))
        .map((e) => `<div class="escalation-mini">${escalationCardHtml(e, { withAccountLink: false })}</div>`)
        .join("")
    : `<p class="no-data">No escalations for this account.</p>`;

  $detailPanel.innerHTML = `
    <div class="detail-close">✕</div>
    <div class="detail-body">
      <div class="detail-header">
        <div class="detail-avatar" style="${avatarStyle(a.account_id)}">${initials(a.borrower_name)}</div>
        <div>
          <h2>${escapeHtml(a.borrower_name)}</h2>
          <div class="biz">${escapeHtml(a.business_name)} · ${a.account_id}</div>
        </div>
      </div>
      <div class="detail-badges">
        <span class="badge risk-${a.risk_tier}">${a.risk_tier} risk</span>
        <span class="badge">${escapeHtml(a.loan_type)}</span>
        <span class="badge">${LANGUAGE_LABELS[a.language_preference] || a.language_preference}</span>
        <a class="badge badge-link" href="tel:${escapeHtml(a.phone_number)}">${escapeHtml(a.phone_number)}</a>
        <button type="button" class="badge badge-edit-btn" id="edit-account-btn" title="Edit phone, language, or risk tier">✎ Edit</button>
      </div>

      <form class="upload-form" id="edit-account-form" hidden>
        <div class="upload-row">
          <input type="text" id="ea-phone" placeholder="Phone (E.164, e.g. +919812345001)" value="${escapeHtml(a.phone_number)}" />
          <select id="ea-language">
            <option value="en" ${a.language_preference === "en" ? "selected" : ""}>English</option>
            <option value="hi" ${a.language_preference === "hi" ? "selected" : ""}>Hindi</option>
            <option value="hinglish" ${a.language_preference === "hinglish" ? "selected" : ""}>Hinglish</option>
          </select>
          <select id="ea-risk">
            <option value="low" ${a.risk_tier === "low" ? "selected" : ""}>Low risk</option>
            <option value="medium" ${a.risk_tier === "medium" ? "selected" : ""}>Medium risk</option>
            <option value="high" ${a.risk_tier === "high" ? "selected" : ""}>High risk</option>
          </select>
        </div>
        <div class="clarify-actions-row">
          <button type="submit" class="btn btn-approve">Save changes</button>
          <button type="button" class="btn btn-reject" id="edit-account-cancel-btn">Cancel</button>
        </div>
        <p class="mini-form-error" id="edit-account-error" hidden></p>
      </form>

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
              <div>${a.tenure_months - a.months_remaining} of ${a.tenure_months} months paid down</div>
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

      <div class="detail-sectors">
        <div class="detail-sector">
          <p class="sector-heading">Issues &amp; actions</p>

          <div class="section-block">
            <p class="section-title">Flags <span class="count">${a.flags.length}</span></p>
            <p class="section-subtitle">Rule-based, not a black-box score — each flag below states exactly why it fired.</p>
            ${flagsHtml}
          </div>

          <div class="section-block">
            <p class="section-title">Telegram onboarding</p>
            <p class="section-subtitle">${
              a.telegram_linked
                ? "This borrower has verified over Telegram — automated reminders actually reach them."
                : "Not linked yet — every automated reminder to this account is logged, but has nowhere to actually deliver to."
            }</p>
            <button type="button" class="btn ${a.telegram_linked ? "" : "btn-approve"}" id="telegram-invite-btn">
              ${a.telegram_linked ? "Reset access key & get a new invite link" : "Get a Telegram invite link"}
            </button>
            <div id="telegram-invite-result"></div>
          </div>

          <div class="section-block">
            <p class="section-title">Send a clarification request</p>
            <form class="clarify-form" id="clarify-form">
              <label class="clarify-label" for="clarify-note">Your note (what's going on, in your own words)</label>
              <textarea id="clarify-note" rows="2" placeholder="e.g. Third missed promise this quarter, dispute still unresolved — need them to explain."></textarea>
              <div class="clarify-actions-row">
                <button type="button" class="btn btn-reject" id="clarify-polish-btn">✨ Polish wording</button>
                ${a.flags.length ? `<button type="button" class="btn btn-reject" id="clarify-summarize-btn">📋 Summarize all flags</button>` : ""}
                <span class="clarify-hint">Drafts a professional message below, grounded in this account's real flags — you still review it before sending.</span>
              </div>
              <label class="clarify-label" for="clarify-message">Message to send</label>
              <textarea id="clarify-message" rows="4" placeholder="Write directly, or use Polish wording above to draft one."></textarea>
              <div class="clarify-actions-row">
                <button type="submit" class="btn btn-approve" id="clarify-send-btn">Send to borrower</button>
                <span class="clarify-hint">Delivered over Telegram if linked; always logged either way.</span>
              </div>
              <p class="clarify-result" id="clarify-result" hidden></p>
            </form>
            <div class="clarification-history">${clarificationHistoryHtml}</div>
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
              <button type="submit" class="btn upload-btn">Upload &amp; ingest</button>
              <p class="upload-result" id="upload-result" hidden></p>
            </form>
          </div>
        </div>

        <div class="detail-sector">
          <p class="sector-heading">History &amp; activity</p>

          <div class="section-block">
            <p class="section-title">Payment history <span class="count">last ${payments.length}</span></p>
            <p class="section-subtitle">₹ amount per payment, colored by whether it arrived on time.</p>
            ${barsHtml}
            <form class="upload-form" id="record-payment-form" style="margin-top: 12px;">
              <div class="upload-row">
                <input type="number" id="rp-amount" placeholder="Amount (₹)" min="0.01" step="0.01" required />
                <input type="date" id="rp-date" title="Payment date (defaults to today)" />
              </div>
              <div class="mini-form-radio-row" id="rp-extra-question" hidden>
                <span class="clarify-hint" style="flex: 0 0 100%;">Less than what's due this cycle — apply it toward the next EMI?</span>
                <label><input type="radio" name="rp-apply" value="true" /> Yes, credit it</label>
                <label><input type="radio" name="rp-apply" value="false" /> No, just record it</label>
              </div>
              <button type="submit" class="btn upload-btn" id="record-payment-btn">Record payment</button>
              <p class="mini-form-error" id="record-payment-error" hidden></p>
              <p class="upload-result" id="record-payment-result" hidden></p>
            </form>
          </div>

          <div class="section-block">
            <p class="section-title">EMI timeline <span class="count">${timeline.length}</span></p>
            <p class="section-subtitle">Every real past payment, plus every EMI still scheduled ahead — projected monthly from the next due date.</p>
            <div class="timeline-list">${timelineHtml}</div>
          </div>

          <div class="section-block">
            <p class="section-title">Promises to pay <span class="count">${a.promises.length}</span></p>
            <p class="section-subtitle">Commitments the borrower made to pay by a specific date — kept, broken, or still pending.</p>
            ${promisesHtml}
            <form class="upload-form" id="log-promise-form" style="margin-top: 12px;">
              <div class="upload-row">
                <input type="number" id="lp-amount" placeholder="Promised amount (₹)" min="0.01" step="0.01" required />
                <input type="date" id="lp-date" required title="Date they promised to pay by" />
              </div>
              <button type="submit" class="btn upload-btn" id="log-promise-btn">Log a promise to pay</button>
              <p class="mini-form-error" id="log-promise-error" hidden></p>
              <p class="upload-result" id="log-promise-result" hidden></p>
            </form>
          </div>

          <div class="section-block">
            <p class="section-title">Disputes <span class="count">${a.disputes.length}</span></p>
            <p class="section-subtitle">The borrower's own stated reason for each dispute -- including one raised via "Contest" on their dashboard.</p>
            ${disputesHtml}
          </div>

          <div class="section-block">
            <p class="section-title">Call log <span class="count">${a.call_log.length}</span></p>
            <p class="section-subtitle">Every phone contact attempt on record, so a colleague opening this account later can see one already happened.</p>
            ${callLogHtml}
            <form class="upload-form" id="call-log-form" style="margin-top: 12px;">
              <div class="upload-row">
                <select id="cl-outcome">
                  <option value="reached">Reached</option>
                  <option value="no_answer">No answer</option>
                  <option value="voicemail">Left voicemail</option>
                  <option value="wrong_number">Wrong number</option>
                </select>
                <input type="text" id="cl-note" placeholder="Note (optional)" />
              </div>
              <button type="submit" class="btn upload-btn" id="call-log-btn">Log this call</button>
            </form>
          </div>

          <div class="section-block">
            <p class="section-title">Escalations <span class="count">${a.escalations.length}</span></p>
            <p class="section-subtitle">Requests handed off to a human — a restructuring proposal, or anything the agent couldn't resolve on its own.</p>
            ${escalationsHtml}
          </div>

          <div class="section-block">
            <p class="section-title">Conversation <span class="count">${conversation.length}</span></p>
            <p class="section-subtitle">What the AI agent actually said to this borrower, and what it did on their behalf — not just messages ops itself sent.</p>
            <div class="convo-list">${
              conversation.length
                ? conversation.map(conversationEntryHtml).join("")
                : `<p class="no-data">No AI conversation on record for this account yet.</p>`
            }</div>
          </div>

          <div class="section-block">
            <p class="section-title">Documents <span class="count" id="documents-count">${documents.length}</span></p>
            <p class="section-subtitle">Everything uploaded for this borrower — signed agreements, KYC, and more.</p>
            <div id="documents-list"></div>
          </div>
        </div>
      </div>
    </div>`;

  wireDetailClose();
  renderDocumentsList(documents);
  wireEscalationActions($detailPanel, async () => {
    const [fresh, freshDocuments, freshConversation] = await _loadDetailData(a.account_id);
    renderDetail(fresh, freshDocuments, freshConversation);
    // A full reload, not just the escalation queue -- approving/rejecting
    // can change the account's own fields (e.g. months_remaining, EMI on
    // an extend_tenure approval), and the portfolio grid card for this
    // account needs to reflect that immediately, not wait for the next
    // 30s auto-refresh.
    loadAll();
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
        "Summarize every currently open flag on this account for the borrower and ask them to respond."
      );
      document.getElementById("clarify-message").value = result.draft;
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
  // every one of the three actions below -- each can change data shown
  // elsewhere too (the account grid's EMI/flags after a payment, the
  // Disputed filter count after a resolve), same reasoning as the
  // escalation actions' own onDone callback above.
  async function refreshAfterAction() {
    const [fresh, freshDocuments, freshConversation] = await _loadDetailData(a.account_id);
    renderDetail(fresh, freshDocuments, freshConversation);
    loadAll();
  }

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
      toast(`Payment recorded — ${result.kind.replace(/_/g, " ")}.`);
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
  logPromiseForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const errorEl = document.getElementById("log-promise-error");
    errorEl.hidden = true;
    const amount = Number(document.getElementById("lp-amount").value);
    const promisedDate = document.getElementById("lp-date").value;
    if (!amount || amount <= 0 || !promisedDate) {
      errorEl.textContent = "Enter a valid amount and date.";
      errorEl.hidden = false;
      return;
    }
    const btn = document.getElementById("log-promise-btn");
    btn.disabled = true;
    btn.textContent = "Logging…";
    try {
      await logPromise(a.account_id, { promised_date: promisedDate, promised_amount: amount });
      toast("Promise logged.");
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
      await refreshAfterAction();
    } catch (err) {
      toast(err.message, true);
      btn.disabled = false;
      btn.textContent = "Log this call";
    }
  });

  $detailPanel.querySelectorAll(".resolve-dispute-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "Resolving…";
      try {
        await resolveDispute(a.account_id);
        toast("Dispute resolved.");
        await refreshAfterAction();
      } catch (err) {
        toast(err.message, true);
        btn.disabled = false;
        btn.textContent = "Resolve";
      }
    });
  });

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
        ? `<a class="btn btn-approve na-invite-link" href="${escapeHtml(result.telegram_invite_link)}" target="_blank" rel="noopener">Open Telegram invite link</a>
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
   Boot / refresh
   ============================================================ */

async function loadAll() {
  const myGeneration = ++loadGeneration;
  const [accounts, escalations, metrics] = await Promise.all([
    getAccounts(),
    getEscalations(),
    getMetrics(state.sinceHours),
  ]);
  if (myGeneration !== loadGeneration) return; // superseded by a newer load -- discard
  state.accounts = accounts;
  state.escalations = escalations;
  state.metrics = metrics;
  renderKpis();
  renderFilterPills();
  renderAccountGrid();
  renderEscalationList();
}

function stopAutoRefresh() {
  if (state.refreshTimer) clearInterval(state.refreshTimer);
  state.refreshTimer = null;
}

function boot() {
  loadAll().catch((e) => toast(e.message, true));
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
  if (keyToTry) {
    tryUnlock(keyToTry).then((ok) => {
      if (!ok) {
        $unlockScreen.hidden = false;
        $unlockError.hidden = true;
      }
    });
  } else {
    $unlockInput.focus();
  }
})();

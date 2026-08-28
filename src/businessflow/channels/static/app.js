"use strict";

/* ============================================================
   State -- deliberately NOT persisted to localStorage. Unlike the ops
   dashboard's shared operator key, a borrower's access_key is a personal
   credential to their own financial data; each visit re-verifies rather
   than remembering it on this device.
   ============================================================ */
const state = {
  mode: "verified", // "verified" | "anonymous"
  language: "en", // "en" | "hi" -- the only two start_conversation accepts
  conversationId: null,
  accountId: null,
  sending: false,
};

// Real tool names the agent can call (see src/businessflow/tools/*.py) --
// mapped to a plain-language chip so a borrower sees WHAT was checked,
// never the raw function/argument names. Grounded, not decorative: this
// only shows for a tool call that actually happened this turn.
const TOOL_LABELS = {
  get_payment_status: "Checked your account",
  get_payment_history: "Looked up your payment history",
  log_promise_to_pay: "Logged your promise to pay",
  flag_dispute: "Flagged your dispute",
  escalate_to_human: "Connecting you with our team",
  request_closure_certificate: "Requested your closure certificate",
  propose_restructuring: "Submitted your restructuring request",
  generate_payment_link: "Generated a payment link",
  propose_partial_payment: "Submitted your partial payment request",
  calculate_hypothetical: "Calculated your options",
  check_policy: "Checked our policy",
};

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
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let detail = `request failed (${res.status})`;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {}
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

const startConversation = (payload) => api("/conversations", { method: "POST", body: JSON.stringify(payload) });
const sendMessageApi = (conversationId, message) =>
  api(`/conversations/${conversationId}/messages`, { method: "POST", body: JSON.stringify({ message }) });

/* ============================================================
   Start screen
   ============================================================ */
const $startScreen = document.getElementById("start-screen");
const $chatScreen = document.getElementById("chat-screen");
const $startForm = document.getElementById("start-form");
const $startError = document.getElementById("start-error");
const $verifiedFields = document.getElementById("verified-fields");
const $opsFields = document.getElementById("ops-fields");
const $langField = document.getElementById("lang-field");

// The ops dashboard lives on its own subdomain (see the "Ops team" tab
// below) -- separate apps, separate asset paths, avoids the path-prefix
// asset collisions a single shared domain would hit. Kept as one constant
// since it's the only cross-service coupling point in this file.
const OPS_DASHBOARD_URL = "https://businessflowai-ops.duckdns.org";

document.querySelectorAll(".start-tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".start-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    state.mode = tab.dataset.mode;
    $verifiedFields.hidden = state.mode !== "verified";
    $opsFields.hidden = state.mode !== "ops";
    $langField.hidden = state.mode === "ops";
  })
);

document.querySelectorAll(".lang-btn").forEach((btn) =>
  btn.addEventListener("click", () => {
    document.querySelectorAll(".lang-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.language = btn.dataset.lang;
  })
);

$startForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  $startError.hidden = true;

  if (state.mode === "ops") {
    const opsKey = document.getElementById("start-ops-key").value.trim();
    if (!opsKey) {
      $startError.textContent = "Enter the ops API key.";
      $startError.hidden = false;
      return;
    }
    // A URL fragment, not a query param -- never sent to (or logged by)
    // any server, just read client-side by the ops app's own init() once
    // it loads. See ops/static/app.js's init() for the other half of this.
    window.location.href = `${OPS_DASHBOARD_URL}/#key=${encodeURIComponent(opsKey)}`;
    return;
  }

  const payload = { language: state.language };
  if (state.mode === "verified") {
    const accountId = document.getElementById("start-account-id").value.trim();
    const accessKey = document.getElementById("start-access-key").value.trim();
    if (!accountId || !accessKey) {
      $startError.textContent = "Enter both your account ID and access key.";
      $startError.hidden = false;
      return;
    }
    payload.account_id = accountId;
    payload.access_key = accessKey;
  }

  const submitBtn = document.getElementById("start-submit-btn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Starting…";
  try {
    const result = await startConversation(payload);
    state.conversationId = result.conversation_id;
    state.accountId = result.account_id;
    enterChat();
  } catch (err) {
    $startError.textContent = friendlyStartError(err);
    $startError.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = 'Start chatting<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  }
});

function friendlyStartError(err) {
  if (err.status === 401) return "That account ID or access key doesn't look right — check and try again.";
  if (err.status === 429) return err.message; // AccountLockedError's own message already states the real wait
  return "Couldn't start the chat — please try again in a moment.";
}

function enterChat() {
  $startScreen.hidden = true;
  $chatScreen.hidden = false;
  const chip = document.getElementById("account-chip");
  if (state.accountId) {
    chip.hidden = false;
    chip.textContent = state.accountId;
  } else {
    chip.hidden = true;
  }
  addSystemMessage(
    state.accountId
      ? `Verified — you're chatting about account ${state.accountId}.`
      : 'Chatting anonymously — to verify, send your account ID and 6-digit access key together, e.g. "BF-1001 482913".'
  );
  document.getElementById("message-input").focus();
}

document.getElementById("restart-btn").addEventListener("click", () => {
  state.conversationId = null;
  state.accountId = null;
  document.getElementById("message-list").innerHTML = "";
  document.getElementById("start-access-key").value = "";
  $chatScreen.hidden = true;
  $startScreen.hidden = false;
});

/* ============================================================
   Chat screen
   ============================================================ */
const $messageList = document.getElementById("message-list");
const $messageForm = document.getElementById("message-form");
const $messageInput = document.getElementById("message-input");
const $sendBtn = document.getElementById("send-btn");

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s ?? "";
  return div.innerHTML;
}

function scrollToBottom() {
  $messageList.scrollTop = $messageList.scrollHeight;
}

function addSystemMessage(text) {
  const row = document.createElement("div");
  row.className = "msg-row system";
  row.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
  $messageList.appendChild(row);
  scrollToBottom();
}

function addUserMessage(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
  $messageList.appendChild(row);
  scrollToBottom();
}

function addAssistantMessage(text, toolCalls = [], isError = false) {
  const row = document.createElement("div");
  row.className = `msg-row assistant${isError ? " error" : ""}`;
  const chipsHtml = toolCalls.length
    ? `<div class="tool-chips">${toolCalls.map((t) => `<span class="tool-chip">✓ ${escapeHtml(TOOL_LABELS[t.tool] || t.tool)}</span>`).join("")}</div>`
    : "";
  row.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>${chipsHtml}`;
  $messageList.appendChild(row);
  scrollToBottom();
}

function showTyping() {
  const row = document.createElement("div");
  row.className = "msg-row typing-row";
  row.id = "typing-indicator";
  row.innerHTML = `<div class="typing-bubble"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>`;
  $messageList.appendChild(row);
  scrollToBottom();
}

function hideTyping() {
  document.getElementById("typing-indicator")?.remove();
}

function friendlyMessageError(err) {
  if (err.status === 503) return "We're getting a lot of requests right now — please try again in a moment.";
  if (err.status === 502) return "Having trouble reaching the assistant right now — please try again.";
  if (err.status === 404) return "This conversation expired — please start over.";
  return "Something went wrong sending that — please try again.";
}

$messageForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $messageInput.value.trim();
  if (!text || state.sending || !state.conversationId) return;

  addUserMessage(text);
  $messageInput.value = "";
  state.sending = true;
  $messageInput.disabled = true;
  $sendBtn.disabled = true;
  showTyping();

  try {
    const result = await sendMessageApi(state.conversationId, text);
    hideTyping();
    addAssistantMessage(result.reply, result.tool_calls);
    if (result.verified_account_id) {
      state.accountId = result.verified_account_id;
      const chip = document.getElementById("account-chip");
      chip.hidden = false;
      chip.textContent = state.accountId;
    }
  } catch (err) {
    hideTyping();
    addAssistantMessage(friendlyMessageError(err), [], true);
  } finally {
    state.sending = false;
    $messageInput.disabled = false;
    $sendBtn.disabled = false;
    $messageInput.focus();
  }
});

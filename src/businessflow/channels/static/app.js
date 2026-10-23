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
  dashboardStale: false, // set when a tool call this chat could have changed dashboard data
};

// Real tool names the agent can call (see src/businessflow/tools/*.py) --
// mapped to a plain-language chip so a borrower sees WHAT was checked,
// never the raw function/argument names. Grounded, not decorative: this
// only shows for a tool call that actually happened this turn.
// The chip is chosen by tool NAME alone (the API doesn't send results), so each
// label may only claim what was certainly done by calling that tool -- e.g.
// propose_partial_payment only CHECKS an amount against policy and records
// nothing, and request_closure_certificate does nothing when the loan isn't
// fully repaid yet. (They used to read "Submitted your partial payment
// request" / "Requested your closure certificate".)
const TOOL_LABELS = {
  get_payment_status: "Checked your account",
  get_payment_history: "Looked up your payment history",
  log_promise_to_pay: "Noted your promise to pay",
  flag_dispute: "Passed this to our team to check",
  escalate_to_human: "Passed this to our team",
  request_closure_certificate: "Checked your loan closure status",
  propose_restructuring: "Reviewed your request for more time",
  generate_payment_link: "Prepared a payment link",
  propose_partial_payment: "Checked your partial-payment option",
  calculate_hypothetical: "Calculated your options",
  check_policy: "Checked our policy",
  compute: "Worked out the figures",
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
const getDashboard = (conversationId) => api(`/conversations/${conversationId}/dashboard`);
const postDispute = (conversationId, reason) =>
  api(`/conversations/${conversationId}/quick-actions/dispute`, { method: "POST", body: JSON.stringify({ reason }) });
const postPaymentClaim = (conversationId, details) =>
  api(`/conversations/${conversationId}/quick-actions/payment-claim`, { method: "POST", body: JSON.stringify({ details }) });
const postCaseNote = (conversationId, caseRef, message) =>
  api(`/conversations/${conversationId}/cases/${encodeURIComponent(caseRef)}/notes`, { method: "POST", body: JSON.stringify({ message }) });
const postAgent = (conversationId, reason) =>
  api(`/conversations/${conversationId}/quick-actions/agent`, { method: "POST", body: JSON.stringify({ reason: reason || null }) });
const postPaymentLink = (conversationId, amount) =>
  api(`/conversations/${conversationId}/quick-actions/payment-link`, { method: "POST", body: JSON.stringify({ amount }) });

/* ============================================================
   Formatting helpers
   ============================================================ */
function fmtInr(n) {
  return "₹" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

function fmtDate(d) {
  return new Date(`${d}T00:00:00`).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

function fmtDateTime(iso) {
  return new Date(iso).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric", hour: "numeric", minute: "2-digit" });
}

function fmtFileSize(bytes) {
  if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  if (bytes >= 1024) return (bytes / 1024).toFixed(1) + " KB";
  return bytes + " B";
}

function initials(name) {
  return (name || "")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0].toUpperCase())
    .join("");
}

/* ============================================================
   Start screen
   ============================================================ */
const $startScreen = document.getElementById("start-screen");
const $chatScreen = document.getElementById("chat-screen");
const $dashboardScreen = document.getElementById("dashboard-screen");
const $chatBackBtn = document.getElementById("chat-back-btn");
const $startForm = document.getElementById("start-form");
const $startError = document.getElementById("start-error");
const $verifiedFields = document.getElementById("verified-fields");
const $langField = document.getElementById("lang-field");
const $submitBtn = document.getElementById("start-submit-btn");
const $accountPicker = document.getElementById("start-account-picker");

// Deliberately embedded, same tradeoff and reasoning as _OPS_DEMO_LOGIN_KEY
// below -- a one-click "choose a borrower" list instead of typing an
// account ID + 6-digit key from memory. account_id/access_key still go
// through the exact same POST /conversations verification the typed form
// always used; this only fills the two fields for whoever's demoing
// instead of asking them to remember or look up a real key.
const _DEMO_ACCOUNTS = [
  { id: "BF-1001", name: "Priya Sharma", business: "Cotton Threads Boutique", key: "482913" },
  { id: "BF-1002", name: "Arjun Mehta", business: "Mehta Hardware & Electricals", key: "716044" },
  { id: "BF-1003", name: "Fatima Khan", business: "Khan Textiles Wholesale", key: "930571" },
  { id: "BF-1004", name: "Ravi Iyer", business: "Iyer Auto Spares", key: "205839" },
  { id: "BF-1005", name: "Sunita Patil", business: "Patil Dairy & Kirana Store", key: "128143" },
  { id: "BF-1006", name: "Rajesh Kumar Yadav", business: "Yadav Welding & Fabrication Works", key: "738570" },
  { id: "BF-1007", name: "Meera Nair", business: "Nair Spices & Provisions", key: "584558" },
  { id: "BF-1008", name: "Harpreet Singh Sodhi", business: "Sodhi Auto Care Garage", key: "778854" },
  { id: "BF-1009", name: "Anjali Deshmukh", business: "Deshmukh Boutique & Tailoring", key: "511301" },
  { id: "BF-1010", name: "Devang Mishra", business: "AI Start-Up", key: "844488" },
];

for (const acct of _DEMO_ACCOUNTS) {
  const opt = document.createElement("option");
  opt.value = acct.id;
  opt.textContent = `${acct.name} — ${acct.business} (${acct.id})`;
  $accountPicker.appendChild(opt);
}

// The ops dashboard lives on its own subdomain (see the "Ops team" tab
// below) -- separate apps, separate asset paths, avoids the path-prefix
// asset collisions a single shared domain would hit. Kept as one constant
// since it's the only cross-service coupling point in this file.
const OPS_DASHBOARD_URL = "https://businessflowai-ops.duckdns.org";

// Deliberately embedded, not typed in by whoever's demoing -- a one-click
// "Login" straight into the ops dashboard instead of an API-key prompt.
// This means the key is public the moment this file is served (this repo
// is public too), not just visible to whoever opens the demo -- a
// conscious tradeoff for a project that isn't handling real financial
// data, not an accidental leak.
const _OPS_DEMO_LOGIN_KEY = "Hm5eYlNVIwqNbl8U78slihAKfLDcibxe";

const _SUBMIT_BTN_LABELS = {
  verified: 'Start chatting<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  anonymous: 'Start chatting<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  ops: 'Login<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
};

function _resetSubmitBtnLabel() {
  $submitBtn.innerHTML = _SUBMIT_BTN_LABELS[state.mode] || _SUBMIT_BTN_LABELS.verified;
}

// One line under the tabs saying what each choice leads to, so nobody has to
// try all three to find out.
const _MODE_HINTS = {
  verified: "Pick your account to see what's due, pay, and chat about it.",
  anonymous: "Ask a general question, no account needed. You can still verify later in the chat.",
  ops: "For the operations team: opens the portfolio dashboard.",
};
const $modeHint = document.getElementById("start-mode-hint");
$modeHint.textContent = _MODE_HINTS[state.mode];

document.querySelectorAll(".start-tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".start-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    state.mode = tab.dataset.mode;
    $verifiedFields.hidden = state.mode !== "verified";
    $langField.hidden = state.mode === "ops";
    $modeHint.textContent = _MODE_HINTS[state.mode];
    _resetSubmitBtnLabel();
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
    // A URL fragment, not a query param -- never sent to (or logged by)
    // any server, just read client-side by the ops app's own init() once
    // it loads. See ops/static/app.js's init() for the other half of this.
    window.location.href = `${OPS_DASHBOARD_URL}/#key=${encodeURIComponent(_OPS_DEMO_LOGIN_KEY)}`;
    return;
  }

  const payload = { language: state.language };
  if (state.mode === "verified") {
    const selected = _DEMO_ACCOUNTS.find((a) => a.id === $accountPicker.value);
    if (!selected) {
      $startError.textContent = "Choose an account to continue.";
      $startError.hidden = false;
      return;
    }
    payload.account_id = selected.id;
    payload.access_key = selected.key;
  }

  $submitBtn.disabled = true;
  $submitBtn.textContent = "Starting…";
  try {
    const result = await startConversation(payload);
    state.conversationId = result.conversation_id;
    state.accountId = result.account_id;
    if (state.accountId) {
      enterDashboard();
    } else {
      enterChat();
    }
  } catch (err) {
    $startError.textContent = friendlyStartError(err);
    $startError.hidden = false;
  } finally {
    $submitBtn.disabled = false;
    _resetSubmitBtnLabel();
  }
});

function friendlyStartError(err) {
  if (err.status === 401) return "That account ID or access key doesn't look right — check and try again.";
  if (err.status === 429) return err.message; // AccountLockedError's own message already states the real wait
  return "Couldn't start the chat — please try again in a moment.";
}

// Shared by both entry points below -- account chip + the one-time greet
// message are identical whether chat is opened fullscreen (anonymous, no
// dashboard behind it at all) or as a panel over the dashboard (verified).
function _prepareChatContent() {
  const chip = document.getElementById("account-chip");
  if (state.accountId) {
    chip.hidden = false;
    chip.textContent = state.accountId;
  } else {
    chip.hidden = true;
  }
  if (!$messageList.children.length) {
    addSystemMessage(
      state.accountId
        ? `Verified — you're chatting about account ${state.accountId}.`
        : 'Chatting anonymously — to verify, send your account ID and 6-digit access key together, e.g. "BF-1001 482913".'
    );
    showAbilities();
  }
  toneNote = ""; // a "not available" note belongs to the chat language it was shown for
  renderToneBar();
  document.getElementById("message-input").focus();
}

// Anonymous entry point only (start-form's own submit handler, see
// below) -- chat genuinely IS the whole screen here, nothing exists
// behind it to return to, so there's no close/back button and no panel
// treatment at all.
function enterChat() {
  $startScreen.hidden = true;
  $dashboardScreen.hidden = true;
  $chatScreen.classList.remove("as-panel", "open");
  $chatScreen.hidden = false;
  $chatBackBtn.hidden = true;
  _prepareChatContent();
}

// The dashboard's "Chat with us" entry point -- docks as a real side panel
// next to the dashboard (Edge Copilot's split-view convention), not an
// overlay dimming it: the dashboard reflows to fill the remaining width
// and stays fully interactive the whole time, so there's nothing to
// "step out of" to keep using the page. Closing this is just removing
// the two classes below (panel slides out, dashboard reflows back); there
// was never a screen swap to get a borrower stuck behind.
function openChatPanel() {
  $chatBackBtn.hidden = false;
  $chatScreen.classList.add("as-panel");
  $chatScreen.hidden = false;
  requestAnimationFrame(() => {
    $chatScreen.classList.add("open");
    $dashboardScreen.classList.add("chat-open");
  });
  _prepareChatContent();
}

function closeChatPanel() {
  // Found live: closing the panel mid-recording never stopped the
  // MediaRecorder/mic stream -- it kept running in the background (mic
  // stays live, timer keeps firing), and the NEXT mic tap in a brand new
  // conversation would see it still "recording" and finalize + upload
  // that stale, orphaned audio instead of starting a fresh one. Safe to
  // call unconditionally: stopRecording() itself is a no-op when nothing
  // is actually recording.
  stopRecording();
  stopPlayback(); // and a reply being read aloud stops with the chat
  $chatScreen.classList.remove("open");
  $dashboardScreen.classList.remove("chat-open");
  setTimeout(() => {
    $chatScreen.hidden = true;
    $chatScreen.classList.remove("as-panel");
  }, 300);
  // Chat may have driven tool calls (a dispute flagged, an escalation
  // raised, a payment logged) since the dashboard was last shown -- the
  // dashboard was never hidden, but its numbers could be stale now. Only
  // refresh if something could actually have changed, and do it silently
  // (no loading-spinner swap) so closing the panel never flashes the
  // whole dashboard away mid-transition.
  if (state.dashboardStale) {
    state.dashboardStale = false;
    refreshDashboardSilently();
  }
}

// Shared by both the chat screen's restart button and the dashboard's --
// wherever a borrower is, "start over" means the same thing: forget this
// conversation and go back to the start screen.
function restartAll() {
  // Same reasoning as closeChatPanel's stopRecording() call above -- a
  // recording still in flight when "Start over" is tapped must not be
  // left running into the new conversation. Safe unconditionally.
  stopRecording();
  stopPlayback();
  state.conversationId = null;
  state.accountId = null;
  timelineExpanded = false; // the next account starts with the short view again
  document.getElementById("message-list").innerHTML = "";
  $accountPicker.value = "";
  $chatScreen.hidden = true;
  $chatScreen.classList.remove("as-panel", "open");
  $dashboardScreen.classList.remove("chat-open");
  $dashboardScreen.hidden = true;
  $startScreen.hidden = false;
}

document.getElementById("restart-btn").addEventListener("click", restartAll);

// Closing the panel and the anonymous full-screen exit both land here --
// anonymous chat has no dashboard to return to (restartAll goes to the
// start screen instead), so this only ever needs to close the panel; the
// button itself stays hidden the whole time for anonymous (see enterChat).
$chatBackBtn.addEventListener("click", closeChatPanel);

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

const SPEAKER_ICON = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none"><path d="M11 5 6 9H3v6h3l5 4V5Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M16 8a5 5 0 0 1 0 8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`;

function addAssistantMessage(text, toolCalls = [], isError = false) {
  const row = document.createElement("div");
  row.className = `msg-row assistant${isError ? " error" : ""}`;
  const chipsHtml = toolCalls.length
    ? `<div class="tool-chips">${toolCalls.map((t) => `<span class="tool-chip">✓ ${escapeHtml(TOOL_LABELS[t.tool] || t.tool)}</span>`).join("")}</div>`
    : "";
  const speakHtml = isError ? "" : `<button type="button" class="speak-btn" title="Listen to this reply" aria-label="Listen to this reply">${SPEAKER_ICON}</button>`;
  // The agent writes light markdown (**bold**, lists); richtext.js shows it as formatting, escaping everything else. If that
  // script failed to load, the reply is still shown, as plain escaped text.
  const bodyHtml = typeof RichText === "undefined" ? escapeHtml(text) : RichText.formatReply(text);
  row.innerHTML = `<div class="msg-bubble-row"><div class="msg-bubble">${bodyHtml}</div>${speakHtml}</div>${chipsHtml}`;
  let speakBtn = null;
  if (!isError) {
    speakBtn = row.querySelector(".speak-btn");
    speakBtn.addEventListener("click", (e) => playSpeech(text, e.currentTarget));
  }
  $messageList.appendChild(row);
  scrollToBottom();
  return speakBtn;
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

// Sends `text` as the borrower's next message and shows the reply. Typing and the "what I can help with" buttons both come
// through here, so tapping a button is exactly the same as typing its sentence.
async function sendUserText(text, { clearInput = false } = {}) {
  if (!text || state.sending || !state.conversationId) return;

  addUserMessage(text);
  if (clearInput) $messageInput.value = "";
  state.sending = true;
  setAbilityChipsDisabled(true);
  $messageInput.disabled = true;
  $sendBtn.disabled = true;
  showTyping();

  try {
    const result = await sendMessageApi(state.conversationId, text);
    hideTyping();
    addAssistantMessage(result.reply, result.tool_calls);
    if (result.tool_calls && result.tool_calls.length) state.dashboardStale = true;
    if (result.verified_account_id) {
      state.accountId = result.verified_account_id;
      const chip = document.getElementById("account-chip");
      chip.hidden = false;
      chip.textContent = state.accountId;
      showAbilities(); // just verified: the list now includes the account entries
    }
  } catch (err) {
    hideTyping();
    addAssistantMessage(friendlyMessageError(err), [], true);
  } finally {
    state.sending = false;
    setAbilityChipsDisabled(false);
    $messageInput.disabled = false;
    $sendBtn.disabled = false;
    $messageInput.focus();
  }
}

$messageForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendUserText($messageInput.value.trim(), { clearInput: true });
});

/* ============================================================
   Option buttons -- "what I can help with", from GET .../abilities (channels/abilities.py, the same list the Telegram bot
   shows). The server decides what is offered: an unverified chat only gets what it can actually do (policy answers, how to
   verify), never a button the assistant would have to refuse. Deliberately modest: a short list, and the last button is
   always "More questions or doubts". A button either sends its sentence as the borrower's next message (the normal assistant
   turn) or, for "how do I verify" and "more questions", is answered at once with no assistant turn.
   ============================================================ */
function setAbilityChipsDisabled(disabled) {
  document.querySelectorAll(".ability-chip").forEach((chip) => {
    chip.disabled = disabled;
  });
}

function onAbilityTap(ability) {
  if (state.sending) return;
  if (ability.prompt) {
    sendUserText(ability.prompt);
    return;
  }
  // Answered as it stands: show what was tapped, then the answer, without an assistant turn.
  addUserMessage(ability.label);
  addAssistantMessage(ability.reply);
  if (ability.id === "more") $messageInput.focus();
}

// Shows the list as a row in the conversation; only the latest one is kept. `explicit` is a tap on "What can you help with?":
// then a failure is shown. Loading it automatically is a nicety, so a failure there is only logged.
async function showAbilities({ explicit = false } = {}) {
  if (!state.conversationId) return;
  let data;
  try {
    const res = await fetch(`/conversations/${state.conversationId}/abilities`);
    if (!res.ok) throw new ApiError(res.status, `abilities request failed (${res.status})`);
    data = await res.json();
  } catch (err) {
    if (explicit) addAssistantMessage("Couldn't load the options -- please try again.", [], true);
    else console.warn("could not load the option buttons", err);
    return;
  }
  document.querySelectorAll(".msg-row.abilities").forEach((row) => row.remove());
  const row = document.createElement("div");
  row.className = "msg-row abilities";
  const intro = document.createElement("p");
  intro.className = "abilities-intro";
  intro.textContent = data.intro;
  const chips = document.createElement("div");
  chips.className = "ability-chips";
  for (const ability of data.abilities) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "ability-chip";
    chip.textContent = ability.label; // textContent, never innerHTML: these strings come from the server
    chip.disabled = state.sending;
    chip.addEventListener("click", () => onAbilityTap(ability));
    chips.appendChild(chip);
  }
  row.append(intro, chips);
  $messageList.appendChild(row);
  scrollToBottom();
}

document.getElementById("abilities-btn").addEventListener("click", () => showAbilities({ explicit: true }));

/* ============================================================
   Voice input -- record via MediaRecorder, decode+re-encode to WAV
   client-side, upload to POST .../messages/voice.

   Why re-encode at all: Chrome/Edge's MediaRecorder only ever emits
   WebM/Opus, which the backend's decode step (soundfile/libsndfile) can't
   read -- there's no ffmpeg on the server and no codec dependency was
   added for it (see browser_api.py's module docstring). AudioContext can
   always decode whatever the browser itself just recorded, so decoding
   here and re-encoding to plain WAV (which soundfile reads natively, same
   as it already does for Telegram's OGG/Opus voice notes) needs no new
   dependency on either side.
   ============================================================ */
const $micBtn = document.getElementById("mic-btn");
const $recordingIndicator = document.getElementById("recording-indicator");
const $recordingTimer = document.getElementById("recording-timer");

const VOICE_SUPPORTED = !!(
  navigator.mediaDevices && window.MediaRecorder && (window.AudioContext || window.webkitAudioContext)
);
if (!VOICE_SUPPORTED) $micBtn.hidden = true;

let mediaRecorder = null;
let recordedChunks = [];
let recordingStartedAt = 0;
let recordingTimerId = null;
let micStream = null;

function audioBufferToWav(buffer) {
  const left = buffer.getChannelData(0);
  const channelData = buffer.numberOfChannels > 1
    ? (() => {
        const right = buffer.getChannelData(1);
        const mono = new Float32Array(left.length);
        for (let i = 0; i < left.length; i++) mono[i] = (left[i] + right[i]) / 2;
        return mono;
      })()
    : left;

  const sampleRate = buffer.sampleRate;
  const pcm = new Int16Array(channelData.length);
  for (let i = 0; i < channelData.length; i++) {
    const s = Math.max(-1, Math.min(1, channelData[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }

  const dataSize = pcm.length * 2; // 16-bit mono
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (mono, 16-bit)
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);
  for (let i = 0; i < pcm.length; i++) view.setInt16(44 + i * 2, pcm[i], true);

  return new Blob([buf], { type: "audio/wav" });
}

function fmtElapsed(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, "0")}`;
}

function setRecordingUi(recording) {
  $recordingIndicator.hidden = !recording;
  $messageInput.hidden = recording;
  $sendBtn.hidden = recording;
  $micBtn.classList.toggle("recording", recording);
  $micBtn.title = recording ? "Stop recording" : "Record a voice message";
  $micBtn.setAttribute("aria-label", recording ? "Stop recording" : "Record voice message");
}

// Found live: nothing guarded the window between a click and
// getUserMedia's permission prompt resolving -- a fast double-tap (most
// likely once permission is already granted, so it resolves almost
// immediately) could start two concurrent recordings, each overwriting
// the shared mediaRecorder/micStream variables and leaking the first
// stream (mic stays live, orphaned MediaRecorder never stopped).
let startingRecording = false;

async function startRecording() {
  if (!state.conversationId || state.sending || startingRecording) return;
  startingRecording = true;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
  } catch (err) {
    addAssistantMessage("Couldn't access your microphone -- please allow microphone access and try again.", [], true);
    return;
  } finally {
    startingRecording = false;
  }

  recordedChunks = [];
  mediaRecorder = new MediaRecorder(micStream);
  mediaRecorder.addEventListener("dataavailable", (e) => {
    if (e.data.size > 0) recordedChunks.push(e.data);
  });
  mediaRecorder.addEventListener("stop", handleRecordingStopped);
  mediaRecorder.start();

  recordingStartedAt = Date.now();
  $recordingTimer.textContent = "0:00";
  recordingTimerId = setInterval(() => {
    $recordingTimer.textContent = fmtElapsed(Date.now() - recordingStartedAt);
  }, 250);
  setRecordingUi(true);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  clearInterval(recordingTimerId);
  micStream?.getTracks().forEach((t) => t.stop());
  setRecordingUi(false);
}

async function handleRecordingStopped() {
  const recordedMimeType = mediaRecorder.mimeType || "audio/webm";
  const recordingDurationMs = Date.now() - recordingStartedAt;
  const blob = new Blob(recordedChunks, { type: recordedMimeType });
  recordedChunks = [];
  if (blob.size === 0 || recordingDurationMs < 300) return; // too short to be a real recording -- likely an accidental tap

  state.sending = true;
  setAbilityChipsDisabled(true);
  $micBtn.disabled = true;
  showTyping();

  let audioCtx;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioCtx.decodeAudioData(await blob.arrayBuffer());
    const wavBlob = audioBufferToWav(decoded);

    const formData = new FormData();
    formData.append("audio", wavBlob, "recording.wav");
    const res = await fetch(`/conversations/${state.conversationId}/messages/voice`, { method: "POST", body: formData });
    if (!res.ok) {
      let detail = `request failed (${res.status})`;
      try {
        detail = (await res.json()).detail || detail;
      } catch (_) {}
      throw new ApiError(res.status, detail);
    }
    const result = await res.json();
    hideTyping();
    if (result.transcript) addUserMessage(result.transcript);
    const speakBtn = addAssistantMessage(result.reply, result.tool_calls);
    if (speakBtn) playSpeech(result.reply, speakBtn);
    if (result.tool_calls && result.tool_calls.length) state.dashboardStale = true;
    if (result.verified_account_id) {
      state.accountId = result.verified_account_id;
      const chip = document.getElementById("account-chip");
      chip.hidden = false;
      chip.textContent = state.accountId;
      showAbilities(); // just verified: the list now includes the account entries
    }
  } catch (err) {
    hideTyping();
    addAssistantMessage(
      err instanceof ApiError ? friendlyMessageError(err) : "Couldn't process that recording -- please try again.",
      [],
      true,
    );
  } finally {
    audioCtx?.close();
    state.sending = false;
    setAbilityChipsDisabled(false);
    $micBtn.disabled = false;
  }
}

if (VOICE_SUPPORTED) {
  $micBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      stopRecording();
    } else {
      startRecording();
    }
  });
}

/* ============================================================
   Voice output -- on-demand TTS for one reply, played inline when the
   borrower taps the speaker icon on an assistant bubble (see
   addAssistantMessage above). Not generated automatically for every
   reply -- only for one someone actually wants to hear.

   Two voices can say it. By default the server makes the audio (Piper,
   POST .../speech). With "Naturalise tone" on, the page asks the server for the
   words it would have spoken (POST .../speech-text, same verbalize() step) and
   a natural-sounding voice from the borrower's own browser says them (see
   naturalvoice.js). If that voice cannot be used -- none for this language, the
   spoken text could not be fetched, or it never starts -- the server audio plays
   instead, so a tap never ends in silence.
   ============================================================ */

// Only this on/off preference is kept on the device -- no credential or account data, so it does not break the rule at the
// top of this file. The voice itself is looked up again on every page load: voices differ between browsers and computers,
// so a choice made elsewhere can never point at a voice that is not there.
const NATURAL_PREF_KEY = "bf.naturalTone";
const LANGUAGE_NAME = { en: "English", hi: "Hindi" };
const TONE_SAMPLE = { en: "This is the natural tone.", hi: "यह प्राकृतिक आवाज़ है।" };

// If naturalvoice.js failed to load the option is simply hidden; nothing else on the page depends on it.
const speaker = typeof NaturalVoice === "undefined"
  ? { supported: false }
  : NaturalVoice.createSpeaker({ synth: window.speechSynthesis || null, utteranceCtor: window.SpeechSynthesisUtterance || null });

const $toneBar = document.getElementById("tone-bar");
const $toneBtn = document.getElementById("tone-btn");
const $toneNote = document.getElementById("tone-note");

function readNaturalPref() {
  try {
    return localStorage.getItem(NATURAL_PREF_KEY) === "on";
  } catch (_) {
    return false; // storage can be missing or blocked (private window): the choice is then simply not remembered
  }
}

let naturalOn = readNaturalPref();
let toneChecking = false;
let toneNote = ""; // a one-off message: "not available", or "the natural voice did not start"
let loadedVoices = []; // the browser's voices as of the last look; used to name the voice in the note

function setNatural(on) {
  naturalOn = on;
  try {
    localStorage.setItem(NATURAL_PREF_KEY, on ? "on" : "off");
  } catch (_) {
    /* a blocked store only means the choice is not remembered */
  }
}

function toneMessage() {
  if (toneNote) return toneNote;
  if (!naturalOn) return "";
  const voice = NaturalVoice.pickNatural(loadedVoices, state.language);
  if (!voice) {
    return loadedVoices.length
      ? `Natural tone is on, but this browser has no natural ${LANGUAGE_NAME[state.language]} voice, so replies use the standard voice.`
      : "Natural tone is on.";
  }
  // Online voices (Edge's Natural ones) are streamed from the browser's provider, so the reply text goes there to be spoken.
  const online = NaturalVoice.isOnline(voice) ? " This voice is online: the reply text is sent to your browser's provider to be spoken." : "";
  return `Natural tone on: ${voice.name}.${online}`;
}

function renderToneBar() {
  $toneBtn.hidden = !speaker.supported; // the bar also holds "What can you help with?", which every browser can use
  if (!speaker.supported) {
    $toneNote.textContent = "";
    return;
  }
  $toneBtn.disabled = toneChecking;
  $toneBtn.textContent = toneChecking ? "Checking…" : naturalOn ? "Back to standard tone" : "Naturalise tone";
  $toneBtn.setAttribute("aria-pressed", naturalOn ? "true" : "false");
  $toneNote.textContent = toneMessage();
}

// "Naturalise tone": look for a natural voice for this chat's language. None -> say so and change nothing. Found -> switch to
// it and say a short sample so the borrower can hear it.
async function onToneClick() {
  if (naturalOn) {
    setNatural(false);
    toneNote = "";
    stopPlayback();
    renderToneBar();
    return;
  }
  toneChecking = true;
  toneNote = "";
  renderToneBar();
  loadedVoices = await speaker.loadVoices();
  toneChecking = false;
  const voice = NaturalVoice.pickNatural(loadedVoices, state.language);
  if (!voice) {
    toneNote = `Sorry, not available: this browser has no natural-sounding ${LANGUAGE_NAME[state.language]} voice. Google Chrome and Microsoft Edge usually have them.`;
    renderToneBar();
    return;
  }
  setNatural(true);
  stopPlayback();
  renderToneBar();
  speaker.speak(TONE_SAMPLE[state.language], voice, {
    onFail: (failure) => {
      if (failure.started) return;
      setNatural(false);
      toneNote = "The natural voice did not start, so the standard voice stays on.";
      renderToneBar();
    },
  });
}

if ($toneBtn) $toneBtn.addEventListener("click", onToneClick);
if (speaker.supported && naturalOn) {
  // A remembered "on": look the voices up now. If this browser has no natural voice in either language, quietly go back to off.
  speaker.loadVoices().then((voices) => {
    loadedVoices = voices;
    if (!NaturalVoice.pickNatural(voices, "en") && !NaturalVoice.pickNatural(voices, "hi")) setNatural(false);
    renderToneBar();
  });
}
renderToneBar();

// One reply plays at a time: tapping the one that is playing stops it, and starting another stops the one before it.
let currentPlayback = null; // { cancelled, stop() } for whichever reply is loading or playing now

function stopPlayback() {
  const playback = currentPlayback;
  currentPlayback = null;
  if (playback) playback.stop();
}
window.addEventListener("pagehide", stopPlayback);

// true: this reply has been dealt with (spoken, stopped, or failed after some sound). false: the natural voice could not be
// used and nothing was heard, so the caller plays the server audio instead.
async function playNatural(text, buttonEl, playback) {
  const language = state.language;
  loadedVoices = await speaker.loadVoices();
  const voice = NaturalVoice.pickNatural(loadedVoices, language);
  if (!voice) return false; // e.g. natural tone was switched on for an English chat and this one is Hindi
  let spoken;
  try {
    const res = await fetch(`/conversations/${state.conversationId}/speech-text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new ApiError(res.status, `speech-text request failed (${res.status})`);
    spoken = (await res.json()).text;
  } catch (err) {
    console.warn("naturalise: could not get the spoken text; using the server voice instead", err);
    return false;
  }
  if (playback.cancelled) return true; // the borrower moved on while this was loading
  buttonEl.classList.remove("loading");
  buttonEl.classList.add("playing");
  return new Promise((resolve) => {
    playback.stopNatural = () => speaker.stop(); // calls onEnd below, which tidies the button
    speaker.speak(spoken, voice, {
      onEnd: () => {
        playback.stopNatural = null;
        buttonEl.classList.remove("loading", "playing");
        if (currentPlayback === playback) currentPlayback = null;
        resolve(true);
      },
      onFail: (failure) => {
        playback.stopNatural = null;
        buttonEl.classList.remove("playing");
        if (failure.started) {
          // Some of it was heard, so playing the server audio now would say it twice.
          if (currentPlayback === playback) currentPlayback = null;
          buttonEl.classList.remove("loading");
          addAssistantMessage("Couldn't finish playing that reply -- please try again.", [], true);
          resolve(true);
          return;
        }
        // Nothing was heard: the natural voice is not working here. Back to the standard voice, and it says this reply.
        setNatural(false);
        toneNote = "The natural voice did not start, so the standard voice was used.";
        renderToneBar();
        buttonEl.classList.add("loading");
        resolve(false);
      },
    });
  });
}

async function playSpeech(text, buttonEl) {
  if (!state.conversationId || buttonEl.classList.contains("loading")) return;
  if (buttonEl.classList.contains("playing")) {
    stopPlayback();
    return;
  }
  stopPlayback();
  const playback = {
    cancelled: false,
    audioEl: null,
    cleanup: null,
    stopNatural: null, // set while a natural voice is speaking this reply, cleared if the server audio takes over
    stop() {
      this.cancelled = true;
      if (this.stopNatural) this.stopNatural();
      if (this.audioEl) this.audioEl.pause();
      if (this.cleanup) this.cleanup();
      buttonEl.classList.remove("loading", "playing");
    },
  };
  currentPlayback = playback;
  buttonEl.classList.add("loading");
  try {
    if (naturalOn && speaker.supported && (await playNatural(text, buttonEl, playback))) return;
    if (playback.cancelled) return;
    const res = await fetch(`/conversations/${state.conversationId}/speech`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new ApiError(res.status, `speech request failed (${res.status})`);
    const blob = await res.blob();
    if (playback.cancelled) return;
    const url = URL.createObjectURL(blob);
    const audioEl = new Audio(url);
    playback.audioEl = audioEl;
    const cleanup = () => {
      buttonEl.classList.remove("playing");
      URL.revokeObjectURL(url);
      if (currentPlayback === playback) currentPlayback = null;
    };
    playback.cleanup = cleanup;
    audioEl.addEventListener("ended", cleanup);
    audioEl.addEventListener("error", cleanup);
    buttonEl.classList.remove("loading");
    buttonEl.classList.add("playing");
    await audioEl.play();
  } catch (err) {
    if (playback.cancelled) return; // a play() cut short by the borrower's own stop is not an error
    // Found live: this silently reverted the button to idle with zero
    // feedback on ANY failure (backend TTS error, expired conversation,
    // network drop, or a rejected audioEl.play()) -- a borrower tapping
    // the speaker icon had no way to tell the feature broke versus their
    // tap just not registering. Matches the same friendly-error pattern
    // the text/voice-send paths already use.
    if (currentPlayback === playback) currentPlayback = null;
    buttonEl.classList.remove("loading", "playing");
    addAssistantMessage(
      err instanceof ApiError ? friendlyMessageError(err) : "Couldn't play that reply -- please try again.",
      [],
      true,
    );
  }
}

/* ============================================================
   Dashboard screen
   ============================================================ */
const $dashLoading = document.getElementById("dash-loading");
const $dashError = document.getElementById("dash-error");
const $dashErrorText = document.getElementById("dash-error-text");
const $dashBody = document.getElementById("dash-body");

function enterDashboard() {
  $startScreen.hidden = true;
  $chatScreen.hidden = true;
  $dashboardScreen.hidden = false;
  document.getElementById("dash-account-chip").textContent = state.accountId || "";
  loadDashboard();
}

function friendlyDashboardError(err) {
  if (err.status === 403) return "Your session isn't verified anymore — please start over.";
  if (err.status === 404) return "This conversation expired — please start over.";
  return "Couldn't load your account right now — please try again.";
}

async function loadDashboard() {
  if (!state.conversationId) return;
  $dashLoading.hidden = false;
  $dashError.hidden = true;
  $dashBody.hidden = true;
  try {
    const data = await getDashboard(state.conversationId);
    renderDashboard(data);
    $dashLoading.hidden = true;
    $dashBody.hidden = false;
  } catch (err) {
    $dashLoading.hidden = true;
    $dashError.hidden = false;
    $dashErrorText.textContent = friendlyDashboardError(err);
  }
}

// Re-fetch and re-render in place, without ever hiding the currently
// visible dashboard behind a loading spinner -- used when closing the
// chat panel, where the dashboard was never hidden and a spinner-swap
// would just be a jarring flash. A failure here is silently ignored:
// the borrower still has whatever the dashboard last showed, and the
// next real loadDashboard() (a manual refresh, a fresh page load) will
// surface the error properly if it persists.
async function refreshDashboardSilently() {
  if (!state.conversationId || $dashBody.hidden) return;
  try {
    const data = await getDashboard(state.conversationId);
    renderDashboard(data);
  } catch {
    // deliberately silent -- see comment above
  }
}

// Own implementation of the ops dashboard's ring-chart math (fraction of
// the circle's circumference to hide via stroke-dashoffset) -- same idea,
// written fresh for this app rather than imported cross-app.
function ringChartSvg(fraction, color) {
  const size = 96;
  const stroke = 9;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(1, fraction));
  const offset = c * (1 - clamped);
  return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" style="transform:rotate(-90deg)">
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="var(--border)" stroke-width="${stroke}" />
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="${color}" stroke-width="${stroke}"
      stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${offset}" />
  </svg>`;
}

const WARNING_ICON =
  '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 9v4m0 4h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const CHEVRON_ICON =
  '<svg class="dash-warning-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="m6 9 6 6 6-6" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const DOC_ICON =
  '<svg class="doc-icon" width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M14 2v6h6" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>';

// Opens the payment page in a NEW TAB in one click: the blank tab is opened
// synchronously inside the click (so pop-up blockers allow it) and pointed at
// the link once it exists. Before, the borrower got a plain URL printed in a
// green box and had to click that too -- five interactions to pay an overdue
// EMI. If the browser blocks the tab, a real button is shown instead.
async function startPayment(amount, $result) {
  const tab = window.open("about:blank", "_blank");
  if (tab) tab.opener = null;
  try {
    const result = await postPaymentLink(state.conversationId, amount);
    if (tab) {
      tab.location.href = result.payment_link;
      showActionResult(
        $result,
        `Payment page opened in a new tab for ${escapeHtml(fmtInr(result.amount))}. <a href="${escapeHtml(result.payment_link)}" target="_blank" rel="noopener noreferrer">Open it again</a>`,
        true
      );
    } else {
      showActionResult(
        $result,
        `<a class="pay-link-btn" href="${escapeHtml(result.payment_link)}" target="_blank" rel="noopener noreferrer">Pay ${escapeHtml(fmtInr(result.amount))} &rarr;</a>`,
        true
      );
    }
    return result;
  } catch (err) {
    if (tab) tab.close();
    showActionResult($result, friendlyActionError(err), false);
    return null;
  }
}

// Which warning labels can be acted on, and what each button really does
// -- wired to the exact same quick-action endpoints as the Quick Actions
// cards below, never a separate/fake code path. "disputed" is deliberately
// NOT in this map: it's already an active claim under review, so its
// expanded body shows a plain note instead of buttons -- see ops/flags.py
// and _build_warnings in browser_api.py for the label contract.
//
// "This isn't right" (contestPrefix) used to raise a dispute in ONE tap with
// the banner's own sentence as its "reason" -- information-free for ops, easy
// to hit by accident right next to the Pay button, and it froze the account's
// reminders. Then it opened its own little form, a second way to do what the
// Get help card does. Now it opens Get help itself: "I've already paid" or
// "Something's wrong", in the borrower's own words, with contestPrefix telling
// ops which warning it was about.
const WARNING_ACTIONS = {
  overdue: {
    resolveLabel: (account) => `Pay ${fmtInr(account.amount_due_now)} now`,
    resolve: (account, $result) => startPayment(account.amount_due_now, $result),
    refreshAfterResolve: false,
    contestPrefix: "Disputing the overdue status",
  },
  // A late fee still owed after the EMI was paid (the account is no longer overdue). Paying it now pays the fee first:
  // the next EMI isn't due yet (accounts/dues.py's allocate_payment).
  fee_due: {
    resolveLabel: (account) => `Pay the ${fmtInr(account.late_fee_amount || 0)} late fee`,
    resolve: (account, $result) => startPayment(account.late_fee_amount || 0, $result),
    refreshAfterResolve: false,
    contestPrefix: "Disputing the late fee",
  },
  broken_promises: {
    resolveLabel: "Talk to our team",
    resolve: async () => {
      await postAgent(state.conversationId, "Following up on the earlier payment promises on my account");
      return "Request sent \u2014 our team will pick it up. You can follow it under Requests & messages.";
    },
    refreshAfterResolve: true, // a new request just appeared under "Requests & messages"
    contestPrefix: "Disputing the missed-promise count",
  },
};

// Only money owed is red. A dispute is information; a promise history is a notice.
const WARNING_STYLE = { overdue: "", fee_due: "", disputed: "info", broken_promises: "notice" };

// Set fresh by every renderDashboard call below -- the warning-action
// click handler (wired once, outside renderDashboard) reads these at
// click time so it always acts on the row the borrower actually sees,
// not a stale snapshot from an earlier render.
let _lastWarnings = [];
let _lastAccount = null;

const MESSAGES_SHOWN = 4; // the rest behind "Show all": a long reminder history shouldn't bury the requests above it
let messagesExpanded = false;
let _lastMessages = [];

const MESSAGE_KIND_LABELS = {
  message: "Message from our team",
  decision: "Decision on your request",
  dispute: "Update on your dispute",
  fee: "Update on a late fee",
  reminder: "Reminder",
};

// "Due now": the one number (account.amount_due_now -- EMI less credit, plus
// the late fee once it applies; the same figure reminders and Telegram's /pay
// use) and one button. Neutral unless money is actually overdue.
function renderDueCard(account) {
  const $card = document.getElementById("due-card");
  const credit = account.pending_emi_credit || 0;
  let left;
  let cta = "";
  if (account.months_remaining <= 0) {
    left = `<div><p class="due-card-label">Loan status</p><p class="due-card-amount">Fully repaid</p><p class="due-card-sub">Nothing is due on this loan. Ask in chat if you need your closure certificate.</p></div>`;
  } else if (account.amount_due_now <= 0) {
    left = `<div><p class="due-card-label">Due now</p><p class="due-card-amount">${fmtInr(0)}</p><p class="due-card-sub">This month's EMI is already covered by your earlier extra payment. Next EMI of ${fmtInr(account.emi_amount)} is due ${fmtDate(account.emi_due_date)}.</p></div>`;
  } else {
    const overdue = account.days_past_due > 0;
    const feeFor = (account.charges || []).map((c) => fmtDate(c.for_emi_due_date)).join(", ");
    const composition = account.late_fee_applicable
      ? `EMI ${fmtInr(account.amount_due_now - account.late_fee_amount)} + late fee ${fmtInr(account.late_fee_amount)}${feeFor ? ` (for the ${feeFor} EMI)` : ""}`
      : credit > 0.01
        ? `EMI ${fmtInr(account.emi_amount)} less ${fmtInr(credit)} already credited`
        : "This month's EMI";
    const when = overdue
      ? `was due ${fmtDate(account.emi_due_date)} (${account.days_past_due} day${account.days_past_due === 1 ? "" : "s"} ago)${account.late_fee_applicable ? "" : " \u2014 no late fee yet"}`
      : `due ${fmtDate(account.emi_due_date)}`;
    left = `<div><p class="due-card-label">Due now</p><p class="due-card-amount${account.late_fee_applicable ? " overdue" : ""}">${fmtInr(account.amount_due_now)}</p><p class="due-card-sub">${escapeHtml(composition)} \u00b7 ${escapeHtml(when)}</p></div>`;
    cta = `<div class="due-card-cta"><button type="button" class="btn-primary" id="due-pay-btn">Pay ${escapeHtml(fmtInr(account.amount_due_now))}</button><button type="button" class="due-card-alt" id="due-other-btn">Pay a different amount</button></div>`;
  }
  $card.innerHTML = `${left}${cta}${promiseStrip(account.promise)}<div class="action-result" id="due-result" style="flex-basis:100%" hidden></div>`;
  $card.hidden = false;
  const $promisePay = document.getElementById("promise-pay-btn");
  if ($promisePay) {
    $promisePay.addEventListener("click", async () => {
      $promisePay.disabled = true;
      try {
        await startPayment(account.promise.next_part.amount_left, document.getElementById("due-result"));
      } finally {
        $promisePay.disabled = false;
      }
    });
  }
  const $pay = document.getElementById("due-pay-btn");
  if ($pay) {
    $pay.addEventListener("click", async () => {
      $pay.disabled = true;
      try {
        await startPayment(account.amount_due_now, document.getElementById("due-result"));
      } finally {
        $pay.disabled = false;
      }
    });
    document.getElementById("due-other-btn").addEventListener("click", () => {
      const $btn = document.getElementById("action-payment-btn");
      $btn.click();
      $btn.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }
}

// The borrower's promise, in their own words, part by part (accounts/promises.py plan_view): shown while it runs,
// and for two weeks after it was kept or missed so the outcome is never a surprise. A replaced promise is not shown.
const PROMISE_PART_LABEL = { kept: "Paid", broken: "Missed", void: "Cancelled", open: "To pay" };
function promiseStrip(promise) {
  if (!promise || promise.status === "replaced") return "";
  const parts = promise.parts || [];
  const lastDate = parts.length ? parts[parts.length - 1].date : promise.made_on;
  const daysSince = (Date.now() - new Date(`${lastDate}T00:00:00`).getTime()) / 86400000;
  if (promise.status !== "on_track" && daysSince > 14) return "";
  const head = {
    on_track: "Your promise",
    kept: "Promise kept \u2014 thank you",
    broken: "Promise missed",
  }[promise.status] || "Your promise";
  const rows = parts
    .map((p) => {
      const partial = p.status === "open" && p.paid_toward > 0 ? ` \u00b7 ${fmtInr(p.paid_toward)} paid so far` : "";
      return `<li class="promise-part ${p.status}"><span>${fmtInr(p.amount)} by ${fmtDate(p.date)}${partial}</span><span class="promise-chip">${PROMISE_PART_LABEL[p.status] || ""}</span></li>`;
    })
    .join("");
  const note = promise.status === "broken"
    ? "A part was not paid in full by its date, so this promise is marked missed. Talk to us in chat to agree a new one."
    : promise.status === "on_track"
      ? "Each part must be paid in full by its date. While this promise runs, we only remind you of it."
      : "";
  const next = promise.status === "on_track" && promise.next_part
    ? `<button type="button" class="btn-cancel promise-pay" id="promise-pay-btn">Pay ${escapeHtml(fmtInr(promise.next_part.amount_left))} (next part)</button>`
    : "";
  return `<div class="promise-strip ${promise.status}"><p class="due-card-label">${escapeHtml(head)}</p><ul class="promise-parts">${rows}</ul>${note ? `<p class="due-card-sub">${escapeHtml(note)}</p>` : ""}${next}</div>`;
}

function renderDashboard(data) {
  const { account, timeline, warnings, requests, documents, messages } = data;
  _lastWarnings = warnings;
  _lastAccount = account;

  renderDueCard(account);

  // Warnings -- text is rendered exactly as the backend wrote it, never
  // re-worded; label decides which action(s) (if any) the expanded body
  // offers. Collapsed by default, same as requests/documents below.
  document.getElementById("dash-warnings").innerHTML = warnings
    .map((w, i) => {
      const cfg = WARNING_ACTIONS[w.label];
      const resolveLabel = cfg ? (typeof cfg.resolveLabel === "function" ? cfg.resolveLabel(account) : cfg.resolveLabel) : "";
      const body = cfg
        ? `<div class="dash-warning-actions">
             <button type="button" class="btn-primary" data-warning-action="resolve" data-warning-idx="${i}">${escapeHtml(resolveLabel)}</button>
             <button type="button" class="btn-cancel" data-warning-action="contest" data-warning-idx="${i}">This isn't right</button>
           </div>`
        : `<p class="dash-warning-note">This is already open with our team \u2014 nothing more is needed from you. You can follow it, and add anything, under Requests & messages.</p>`;
      return `<div class="dash-warning ${WARNING_STYLE[w.label] || ""}">
        <button type="button" class="dash-warning-head" data-warning-toggle="${i}">
          ${WARNING_ICON}<span>${escapeHtml(w.text)}</span>${CHEVRON_ICON}
        </button>
        <div class="dash-warning-body" id="warning-body-${i}" hidden>${body}</div>
      </div>`;
    })
    .join("");
  document.getElementById("attention-section").hidden = warnings.length === 0;


  // Hero identity (no risk label: it described the borrower, not the loan)
  document.getElementById("dash-avatar").textContent = initials(account.borrower_name);
  document.getElementById("dash-borrower-name").textContent = account.borrower_name;
  document.getElementById("dash-business-name").textContent = `${account.business_name} \u00b7 ${account.account_id}`;

  // Progress ring -- fraction of the loan term elapsed so far. Neutral colour:
  // progress is not danger, whatever the account's state. Clamped at 0: after
  // a tenure extension months_remaining can exceed tenure_months.
  const elapsed = account.tenure_months - account.months_remaining;
  const fraction = account.tenure_months > 0 ? Math.max(0, elapsed) / account.tenure_months : 0;
  document.getElementById("dash-ring-stack").innerHTML = ringChartSvg(fraction, "var(--accent)");
  document.getElementById("dash-ring-value").textContent = account.months_remaining;
  document.getElementById("dash-tenure").textContent = account.tenure_months;

  // Loan snapshot stats -- only surface the conditional fields when the
  // backend actually sent a value for them (interest_rate_pct is null
  // until a signed agreement is parsed; late_fee_amount is null unless
  // late_fee_applicable is true).
  const overdue = account.days_past_due > 0;
  const stats = [
    ["Principal", fmtInr(account.principal_amount)],
    ["EMI amount", fmtInr(account.emi_amount)],
    [overdue ? "EMI was due" : "Next due", fmtDate(account.emi_due_date)],
    // outstanding_balance_approx is EMI x months left (all remaining EMIs,
    // interest included) -- not a payoff figure, so don't label it one.
    ["EMIs left to pay", fmtInr(account.outstanding_balance_approx)],
    ["Days past due", String(account.days_past_due), overdue ? "flagged" : "ok"],
    ["Auto-debit (NACH)", account.nach_mandate_active ? "Active" : "Inactive", account.nach_mandate_active ? "ok" : "flagged"],
  ];
  if (account.interest_rate_pct != null) stats.push(["Interest rate", `${account.interest_rate_pct}% p.a.`]);
  if (account.late_fee_applicable) stats.push(["Late fee", fmtInr(account.late_fee_amount), "flagged"]);
  document.getElementById("dash-hero-stats").innerHTML = stats
    .map(
      ([label, value, cls]) =>
        `<div><div class="hero-stat-label">${escapeHtml(label)}</div><div class="hero-stat-value${cls ? " " + cls : ""}">${escapeHtml(value)}</div></div>`
    )
    .join("");

  _lastTimeline = timeline;
  renderTimeline();

  renderRequestsAndMessages(requests, messages);

  // Documents -- real download links, scoped to this conversation's verified account.
  document.getElementById("dash-documents").innerHTML = documents.length
    ? documents
        .map(
          (d) =>
            `<div class="doc-row">${DOC_ICON}<div class="doc-info"><div class="doc-name">${escapeHtml(d.filename)}</div><div class="doc-meta">${fmtFileSize(d.size_bytes)} \u00b7 uploaded ${fmtDateTime(d.uploaded_at)}</div></div><a class="doc-download" href="/conversations/${state.conversationId}/documents/${encodeURIComponent(d.filename)}" download>Download</a></div>`
        )
        .join("")
    : `<p class="dash-no-data">No documents uploaded yet.</p>`;
}

// "Requests & messages": each request is one conversation -- what the borrower
// said, our replies, how it ended -- with a box to add a note while it's open;
// then everything else we've sent (reminders, general messages). Before, a
// request and our reply to it sat in two different panels ("Your requests",
// "Messages from us") with nothing connecting them.
const DISPUTE_KINDS = ["dispute", "payment_claim"];

function threadMsgHtml(author, text, when) {
  const you = author === "borrower";
  return `<div class="thread-msg ${you ? "you" : "team"}"><p class="thread-who">${you ? "You" : "Our team"} <span class="thread-time">${fmtDateTime(when)}</span></p><p class="thread-text">${escapeHtml(text)}</p></div>`;
}

function requestCardHtml(r) {
  const isDispute = DISPUTE_KINDS.includes(r.kind);
  const thread = r.thread || [];
  // A dispute's detail is the borrower's own words, so it opens the conversation; a request's detail (e.g. the
  // terms asked for) is a summary line under the title.
  const opening = isDispute && r.detail ? threadMsgHtml("borrower", r.detail, r.created_at) : "";
  const summary = !isDispute && r.detail ? `<p class="esc-detail">${escapeHtml(r.detail)}</p>` : "";
  // How it ended is a message in the thread for anything closed since threads existed; older ones only have the note.
  const outcome = r.outcome && !thread.some((m) => m.author === "ops") ? threadMsgHtml("ops", r.outcome, r.resolved_at || r.created_at) : "";
  const note = r.can_reply
    ? `<form class="case-note-form" data-case-ref="${escapeHtml(r.id)}">
         <label class="sr-only" for="note-${escapeHtml(r.id)}">Add a note for our team</label>
         <textarea id="note-${escapeHtml(r.id)}" rows="1" maxlength="1000" placeholder="${r.kind === "payment_claim" ? "Add a payment reference (UTR) or anything else" : "Add a note for our team"}"></textarea>
         <button type="submit" class="btn-cancel">Send</button>
       </form>`
    : "";
  return `<div class="esc-card case-card${r.status === "open" ? " is-open" : ""}">
    <div class="case-head"><p class="esc-title">${escapeHtml(r.title)}</p><span class="esc-status ${escapeHtml(r.status)}">${escapeHtml(r.status_label)}</span></div>
    ${summary}
    <div class="thread">${opening}${thread.map((m) => threadMsgHtml(m.author, m.message, m.created_at)).join("")}${outcome}</div>
    <p class="esc-meta">Opened ${fmtDateTime(r.created_at)}${r.resolved_at ? ` \u00b7 closed ${fmtDateTime(r.resolved_at)}` : ""}</p>
    ${note}
  </div>`;
}

// What the borrower can do with a message: answer a message from our team (it becomes a request our team works on,
// or joins the one their earlier reply opened), or act on a reminder (pay what is due now, or ask about it).
function messageActionsHtml(m) {
  if (m.kind === "message" && m.id) {
    if (m.replied_case_ref) {
      return `<p class="dash-msg-replied">You replied \u2014 it's request ${escapeHtml(m.replied_case_ref)} under Your requests, where our answer will appear.</p>
        <form class="msg-reply-form" data-message-id="${m.id}"><textarea rows="2" maxlength="1000" placeholder="Add to your reply\u2026" aria-label="Add to your reply"></textarea><button type="submit" class="btn-cancel">Send</button></form>`;
    }
    return `<form class="msg-reply-form" data-message-id="${m.id}"><textarea rows="2" maxlength="1000" placeholder="Reply to our team\u2026" aria-label="Reply to our team"></textarea><button type="submit" class="btn-primary">Reply</button></form>`;
  }
  if (m.kind === "reminder") {
    return `<div class="dash-msg-actions"><button type="button" class="btn-primary" data-msg-action="pay">Pay what's due now</button><button type="button" class="btn-cancel" data-msg-action="ask" data-msg-date="${escapeHtml(fmtDateTime(m.created_at))}">Ask about this</button></div><div class="action-result" hidden></div>`;
  }
  return "";
}

function renderMessages() {
  const messages = _lastMessages;
  const $toggle = document.getElementById("messages-toggle-btn");
  const shown = messagesExpanded ? messages : messages.slice(0, MESSAGES_SHOWN);
  document.getElementById("dash-messages").innerHTML = messages.length
    ? `<p class="dash-subhead">Other messages from us</p>` +
      shown
        .map(
          (m) =>
            `<div class="dash-msg-card"><div class="dash-msg-body"><p class="dash-msg-kind">${escapeHtml(MESSAGE_KIND_LABELS[m.kind] || "Message")}</p><p class="dash-msg-text">${escapeHtml(m.message)}</p><p class="dash-msg-meta">${fmtDateTime(m.created_at)}${m.kind === "reminder" ? " \u00b7 figures as of that day" : ""}</p>${messageActionsHtml(m)}</div>${m.delivered_via_telegram ? `<span class="dash-msg-badge yes">Also sent on Telegram</span>` : ""}</div>`
        )
        .join("")
    : "";
  $toggle.hidden = messages.length <= MESSAGES_SHOWN;
  $toggle.setAttribute("aria-expanded", String(messagesExpanded));
  $toggle.textContent = messagesExpanded ? "Show fewer messages" : `Show all ${messages.length} messages`;
}

function renderRequestsAndMessages(requests, messages) {
  // Open requests first (they may need something from the borrower), then the rest, newest first -- the server
  // already sends newest first, so a stable sort keeps that order within each group.
  const ordered = requests.slice().sort((a, b) => (a.status === "open" ? 0 : 1) - (b.status === "open" ? 0 : 1));
  document.getElementById("dash-escalations").innerHTML = ordered.length
    ? ordered.map(requestCardHtml).join("")
    : messages.length
      ? ""
      : `<p class="dash-no-data">Nothing yet. If you ask our team for something with Get help, you can follow it here.</p>`;
  _lastMessages = messages;
  renderMessages();
}

document.getElementById("messages-toggle-btn").addEventListener("click", () => {
  messagesExpanded = !messagesExpanded;
  renderMessages();
});

// Delegated, like the warnings below: the list is re-rendered on every load.
document.getElementById("dash-messages").addEventListener("submit", async (e) => {
  const form = e.target.closest(".msg-reply-form");
  if (!form) return;
  e.preventDefault();
  const $text = form.querySelector("textarea");
  const text = $text.value.trim();
  if (!text) {
    $text.focus();
    return;
  }
  const $btn = form.querySelector("button[type=submit]");
  $btn.disabled = true;
  $btn.textContent = "Sending\u2026";
  try {
    await api(`/conversations/${state.conversationId}/messages-from-us/${form.dataset.messageId}/reply`, { method: "POST", body: JSON.stringify({ message: text }) });
    await loadDashboard(); // the reply now shows as a request, with its thread
  } catch (err) {
    $btn.disabled = false;
    $btn.textContent = "Reply";
    let $err = form.querySelector(".case-note-error");
    if (!$err) {
      $err = document.createElement("p");
      $err.className = "case-note-error";
      form.appendChild($err);
    }
    $err.innerHTML = friendlyActionError(err);
  }
});

document.getElementById("dash-messages").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-msg-action]");
  if (!btn) return;
  if (btn.dataset.msgAction === "pay") {
    btn.disabled = true;
    try {
      await startPayment(_lastAccount.amount_due_now, btn.closest(".dash-msg-body").querySelector(".action-result"));
    } finally {
      btn.disabled = false;
    }
    return;
  }
  const $input = document.getElementById("message-input");
  $input.value = `About your reminder of ${btn.dataset.msgDate}: `;
  $input.scrollIntoView({ behavior: "smooth", block: "center" });
  $input.focus();
});

document.getElementById("dash-escalations").addEventListener("submit", async (e) => {
  const form = e.target.closest(".case-note-form");
  if (!form) return;
  e.preventDefault();
  const $text = form.querySelector("textarea");
  const text = $text.value.trim();
  if (!text) {
    $text.focus();
    return;
  }
  const $btn = form.querySelector("button[type=submit]");
  $btn.disabled = true;
  $btn.textContent = "Sending\u2026";
  try {
    await postCaseNote(state.conversationId, form.dataset.caseRef, text);
    await loadDashboard(); // shows the note in its thread
  } catch (err) {
    $btn.disabled = false;
    $btn.textContent = "Send";
    let $err = form.querySelector(".case-note-error");
    if (!$err) {
      $err = document.createElement("p");
      $err.className = "case-note-error";
      form.appendChild($err);
    }
    $err.innerHTML = err.status === 409 ? "This request was just closed \u2014 use Get help to raise something new." : friendlyActionError(err);
  }
});

// Wired once, not per-render -- #dash-warnings' own innerHTML is fully
// replaced on every renderDashboard call, so binding to individual
// buttons there would leak listeners (and silently stop working the
// moment a warning is added/removed). Delegation on the stable container
// means this keeps working across every reload.
document.getElementById("dash-warnings").addEventListener("click", async (e) => {
  const toggleBtn = e.target.closest("[data-warning-toggle]");
  if (toggleBtn) {
    const idx = toggleBtn.dataset.warningToggle;
    const $body = document.getElementById(`warning-body-${idx}`);
    const opening = $body.hidden;
    $body.hidden = !opening;
    toggleBtn.closest(".dash-warning").classList.toggle("open", opening);
    return;
  }

  const actionBtn = e.target.closest("[data-warning-action]");
  if (!actionBtn) return;
  const idx = Number(actionBtn.dataset.warningIdx);
  const warning = _lastWarnings[idx];
  const cfg = warning && WARNING_ACTIONS[warning.label];
  if (!cfg) return;
  const action = actionBtn.dataset.warningAction;

  if (action === "contest") {
    // Overdue: it may be a payment we haven't seen, or a wrong figure -- let them say which. A promise count can only
    // be wrong.
    openHelp({ choice: warning.label === "overdue" ? null : "wrong", context: cfg.contestPrefix });
    return;
  }

  const $result = document.getElementById("warnings-action-result");
  actionBtn.disabled = true;
  try {
    if (warning.label === "overdue") {
      await cfg.resolve(_lastAccount, $result); // opens the payment page itself and reports its own result
    } else {
      const html = await cfg.resolve(_lastAccount);
      showActionResult($result, escapeHtml(html), true);
      if (cfg.refreshAfterResolve) loadDashboard();
    }
  } catch (err) {
    showActionResult($result, friendlyActionError(err), false);
  } finally {
    actionBtn.disabled = false;
  }
});

document.getElementById("dash-retry-btn").addEventListener("click", loadDashboard);
document.getElementById("dash-refresh-btn").addEventListener("click", loadDashboard);
document.getElementById("dash-restart-btn").addEventListener("click", restartAll);

// Payment timeline -- real history, then projected EMIs, one chronological
// list. It can run to dozens of scheduled EMIs on a long tenure, so by default
// it shows what matters now: the latest entries that already happened (paid,
// or overdue) and the next few scheduled. "Show all" lists every entry. The
// near term is never hidden, only the far end of the schedule.
const TIMELINE_RECENT = 3;
const TIMELINE_UPCOMING = 3;
const $timelineToggleBtn = document.getElementById("timeline-toggle-btn");
const $dashTimeline = document.getElementById("dash-timeline");
let _lastTimeline = [];
let timelineExpanded = false;

function timelineRowHtml(t) {
  return `<div class="dtl-row ${t.status}"><span class="dtl-dot ${t.status}"></span><div class="dtl-info"><div class="dtl-label">${escapeHtml(t.label)}</div><div class="dtl-meta">${fmtDate(t.date)}</div></div><div class="dtl-amount">${fmtInr(t.amount)}</div></div>`;
}

function renderTimeline() {
  const timeline = _lastTimeline;
  if (!timeline.length) {
    $dashTimeline.innerHTML = `<p class="dash-no-data">No payment history and no scheduled EMIs.</p>`;
    $timelineToggleBtn.hidden = true;
    return;
  }
  const firstUpcoming = timeline.findIndex((t) => t.status === "upcoming");
  const nowIdx = firstUpcoming === -1 ? timeline.length : firstUpcoming;
  const start = Math.max(0, nowIdx - TIMELINE_RECENT);
  const end = Math.min(timeline.length, nowIdx + TIMELINE_UPCOMING);
  const hiddenCount = timeline.length - (end - start);
  const shown = timelineExpanded || hiddenCount === 0 ? timeline : timeline.slice(start, end);
  $dashTimeline.innerHTML = shown.map(timelineRowHtml).join("");
  $timelineToggleBtn.hidden = hiddenCount === 0;
  $timelineToggleBtn.setAttribute("aria-expanded", String(timelineExpanded));
  $timelineToggleBtn.textContent = timelineExpanded ? "Show less" : `Show all ${timeline.length} entries`;
}

$timelineToggleBtn.addEventListener("click", () => {
  timelineExpanded = !timelineExpanded;
  renderTimeline();
});
document.getElementById("dash-chat-btn").addEventListener("click", openChatPanel);

/* ============================================================
   Quick actions -- each POSTs directly (bypassing the LLM, exactly like
   the backend's own quick-action endpoints), shows its own real result or
   error inline, and refreshes the dashboard afterward since a dispute or
   escalation changes data the hero/warnings/escalations sections show.
   ============================================================ */
function showActionResult($el, html, ok) {
  $el.hidden = false;
  $el.className = `action-result ${ok ? "ok" : "err"}`;
  $el.innerHTML = html;
}

function friendlyActionError(err) {
  if (err.status === 400) return escapeHtml(err.message || "Please fill this in and try again.");
  if (err.status === 422) return "Enter a valid amount.";
  if (err.status === 403) return "Your session isn't verified anymore — please start over.";
  if (err.status === 404) return "This conversation expired — please start over.";
  if (err instanceof ApiError) return escapeHtml(err.message);
  return "Something went wrong — please check your connection and try again.";
}

// Found live: each quick-action card toggled only its OWN form, so
// opening "Get a payment link" never closed "Raise a dispute" if it was
// already open -- clicking through all three left all three expanded at
// once instead of one at a time. _allActionForms is the shared registry
// every wireActionToggle call adds itself to, so opening one can close
// the rest.
const _allActionForms = [];

function wireActionToggle(btnId, formId, resultId) {
  const $btn = document.getElementById(btnId);
  const $form = document.getElementById(formId);
  const $result = document.getElementById(resultId);
  _allActionForms.push($form);
  $btn.addEventListener("click", () => {
    const opening = $form.hidden;
    _allActionForms.forEach((f) => { f.hidden = true; });
    $form.hidden = !opening;
    if (!$form.hidden) $result.hidden = true;
  });
  return { $form, $result };
}

// Get help: one card for everything a borrower brings to us -- "I've already
// paid" (a payment claim the team checks against the bank record), "Something's
// wrong" (a dispute), "Talk to a person" (a request). These were two cards
// ("Something looks wrong", "Talk to our team") plus a form under each warning,
// and "I already paid" had no place of its own: it was filed as a dispute.
const HELP_CHOICES = {
  paid: {
    label: "When and how did you pay?",
    placeholder: "e.g. \u20b912,500 by UPI on 3 Oct, reference 4521\u2026",
    hint: "Our team checks it against the bank record and updates your account. Reminders pause until then.",
    required: true,
  },
  wrong: {
    label: "What isn't right?",
    placeholder: "e.g. A late fee was added although I paid on time",
    hint: "Reminders pause while our team looks into it. You'll see the reply under Requests & messages.",
    required: true,
  },
  person: {
    label: "What do you need help with? (optional)",
    placeholder: "Anything that helps our team prepare",
    hint: "A person on our team picks this up and replies under Requests & messages.",
    required: false,
  },
};

const $helpBtn = document.getElementById("action-help-btn");
const { $form: $helpForm, $result: $helpResult } = wireActionToggle("action-help-btn", "help-form", "help-result");
const $helpFields = document.getElementById("help-fields");
const $helpText = document.getElementById("help-text");
let helpChoice = null;
let helpContext = null; // which warning "This isn't right" came from, so ops knows what it is about

function selectHelpChoice(choice) {
  helpChoice = choice;
  $helpForm.querySelectorAll("[data-help-choice]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.helpChoice === choice)));
  const cfg = HELP_CHOICES[choice];
  $helpFields.hidden = !cfg;
  if (!cfg) return;
  document.getElementById("help-label").textContent = cfg.label;
  $helpText.placeholder = cfg.placeholder;
  document.getElementById("help-hint").textContent = cfg.hint;
  $helpText.focus();
}

function resetHelp() {
  helpContext = null;
  $helpText.value = "";
  selectHelpChoice(null);
}

/** Opens Get help, optionally on one choice -- also what a warning's "This isn't right" does. */
function openHelp({ choice = null, context = null } = {}) {
  if ($helpForm.hidden) $helpBtn.click();
  $helpResult.hidden = true;
  helpContext = context;
  selectHelpChoice(choice);
  $helpBtn.scrollIntoView({ behavior: "smooth", block: "center" });
}

$helpBtn.addEventListener("click", () => {
  $helpBtn.setAttribute("aria-expanded", String(!$helpForm.hidden));
  if ($helpForm.hidden) resetHelp();
});
$helpForm.querySelectorAll("[data-help-choice]").forEach((btn) =>
  btn.addEventListener("click", () => selectHelpChoice(btn.dataset.helpChoice))
);
document.getElementById("help-cancel-btn").addEventListener("click", () => {
  $helpForm.hidden = true;
  $helpBtn.setAttribute("aria-expanded", "false");
  resetHelp();
});

function helpResultText(choice, result) {
  if (choice === "person") return "Request sent \u2014 a person on our team will pick it up. You can follow it under Requests & messages.";
  if (result.already_open) {
    return result.kind === "payment_claim"
      ? "Added to the payment you already reported \u2014 our team will see it there."
      : "You already have an open case with us \u2014 we've added this to it.";
  }
  if (choice === "paid") {
    const days = result.review_within_working_days;
    return `Thanks \u2014 our team will check this payment against the bank record${days ? ` within ${days} working day${days === 1 ? "" : "s"}` : ""} and update your account. Reminders are paused until then. You can follow it under Requests & messages.`;
  }
  return "Thanks \u2014 our team will look into this and reply under Requests & messages. Reminders are paused meanwhile.";
}

$helpForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const cfg = HELP_CHOICES[helpChoice];
  if (!cfg) return;
  const text = $helpText.value.trim();
  if (cfg.required && !text) {
    showActionResult($helpResult, "Please add a few words first.", false);
    $helpText.focus();
    return;
  }
  const $btn = document.getElementById("help-submit-btn");
  $btn.disabled = true;
  const choice = helpChoice;
  try {
    let result;
    if (choice === "paid") result = await postPaymentClaim(state.conversationId, text);
    else if (choice === "wrong") result = await postDispute(state.conversationId, helpContext ? `${helpContext}: ${text}` : text);
    else result = await postAgent(state.conversationId, text);
    showActionResult($helpResult, escapeHtml(helpResultText(choice, result)), true);
    $helpForm.hidden = true;
    $helpBtn.setAttribute("aria-expanded", "false");
    resetHelp();
    loadDashboard(); // the request (and, for a claim, the paused-reminders notice) just appeared
  } catch (err) {
    showActionResult($helpResult, friendlyActionError(err), false);
  } finally {
    $btn.disabled = false;
  }
});

const { $form: $paymentForm, $result: $paymentResult } = wireActionToggle("action-payment-btn", "payment-form", "payment-result");
document.getElementById("payment-cancel-btn").addEventListener("click", () => { $paymentForm.hidden = true; });
$paymentForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const $amountInput = document.getElementById("payment-amount");
  const amount = Number($amountInput.value);
  if (!$amountInput.value || !(amount > 0)) {
    showActionResult($paymentResult, "Enter a valid amount.", false);
    return;
  }
  const $btn = document.getElementById("payment-submit-btn");
  $btn.disabled = true;
  try {
    // No account/request/document field changes as a result of this action
    // (see browser_api.py) -- nothing on the dashboard to refresh.
    await startPayment(amount, $paymentResult);
  } finally {
    $btn.disabled = false;
  }
});

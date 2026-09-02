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

// Escalation status -> borrower-facing label (the API's own status strings
// are fine as CSS class names but "queued_for_human" isn't fit to print).
const STATUS_LABELS = {
  queued_for_human: "Pending",
  approved: "Approved",
  rejected: "Rejected",
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
    if (state.accountId) {
      enterDashboard();
    } else {
      enterChat();
    }
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
  }
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
  state.conversationId = null;
  state.accountId = null;
  document.getElementById("message-list").innerHTML = "";
  document.getElementById("start-access-key").value = "";
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
  // No speaker button on an error bubble -- there's nothing real to speak,
  // and playSpeech would just re-send the friendly error text as if it
  // were part of the conversation.
  const speakHtml = isError ? "" : `<button type="button" class="speak-btn" title="Listen to this reply" aria-label="Listen to this reply">${SPEAKER_ICON}</button>`;
  row.innerHTML = `<div class="msg-bubble-row"><div class="msg-bubble">${escapeHtml(text)}</div>${speakHtml}</div>${chipsHtml}`;
  if (!isError) {
    row.querySelector(".speak-btn").addEventListener("click", (e) => playSpeech(text, e.currentTarget));
  }
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
    if (result.tool_calls && result.tool_calls.length) state.dashboardStale = true;
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

async function startRecording() {
  if (!state.conversationId || state.sending) return;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
  } catch (err) {
    addAssistantMessage("Couldn't access your microphone -- please allow microphone access and try again.", [], true);
    return;
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
    addAssistantMessage(result.reply, result.tool_calls);
    if (result.tool_calls && result.tool_calls.length) state.dashboardStale = true;
    if (result.verified_account_id) {
      state.accountId = result.verified_account_id;
      const chip = document.getElementById("account-chip");
      chip.hidden = false;
      chip.textContent = state.accountId;
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
   ============================================================ */
async function playSpeech(text, buttonEl) {
  if (!state.conversationId || buttonEl.classList.contains("loading") || buttonEl.classList.contains("playing")) return;
  buttonEl.classList.add("loading");
  try {
    const res = await fetch(`/conversations/${state.conversationId}/speech`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`speech request failed (${res.status})`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audioEl = new Audio(url);
    const cleanup = () => {
      buttonEl.classList.remove("playing");
      URL.revokeObjectURL(url);
    };
    audioEl.addEventListener("ended", cleanup);
    audioEl.addEventListener("error", cleanup);
    buttonEl.classList.remove("loading");
    buttonEl.classList.add("playing");
    await audioEl.play();
  } catch (err) {
    buttonEl.classList.remove("loading", "playing");
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

// Which warning labels can be acted on, and what each button really does
// -- wired to the exact same quick-action endpoints as the Quick Actions
// cards below, never a separate/fake code path. "disputed" is deliberately
// NOT in this map: it's already an active claim under review (raised via
// one of these same "Contest" buttons, or by ops directly), so its
// expanded body shows a plain note instead of buttons -- see ops/flags.py
// and _build_warnings in browser_api.py for the label contract.
const WARNING_ACTIONS = {
  overdue: {
    resolveLabel: "Get a payment link",
    resolve: (account) =>
      postPaymentLink(state.conversationId, account.emi_amount + (account.late_fee_applicable ? account.late_fee_amount : 0)),
    resolveResultHtml: (result) =>
      `Payment link for ${escapeHtml(fmtInr(result.amount))}:<br><a href="${escapeHtml(result.payment_link)}" target="_blank" rel="noopener noreferrer">${escapeHtml(result.payment_link)}</a>`,
    // A link alone doesn't move any account field until it's actually
    // paid -- same reasoning as the Quick Actions "Get a payment link"
    // card, which also never refreshes the dashboard after generating one.
    refreshAfterResolve: false,
    contestReason: (text) => `Contesting overdue status: ${text}`,
  },
  broken_promises: {
    resolveLabel: "Talk to a human",
    resolve: () => postAgent(state.conversationId, "Following up on the missed payment promises flagged on my account"),
    resolveResultHtml: (result) =>
      escapeHtml(`Request sent — our team has been notified (status: ${STATUS_LABELS[result.status] || result.status}).`),
    refreshAfterResolve: true, // a new escalation just appeared in "Your requests"
    contestReason: (text) => `Contesting missed payment promises: ${text}`,
  },
};

// Set fresh by every renderDashboard call below -- the warning-action
// click handler (wired once, outside renderDashboard) reads these at
// click time so it always acts on the row the borrower actually sees,
// not a stale snapshot from an earlier render.
let _lastWarnings = [];
let _lastAccount = null;

function renderDashboard(data) {
  const { account, timeline, warnings, escalations, documents, messages } = data;
  _lastWarnings = warnings;
  _lastAccount = account;

  // Warnings -- text is rendered exactly as the backend wrote it, never
  // re-worded; label decides which action(s) (if any) the expanded body
  // offers. Collapsed by default, same as escalations/documents below.
  document.getElementById("dash-warnings").innerHTML = warnings
    .map((w, i) => {
      const cfg = WARNING_ACTIONS[w.label];
      const body = cfg
        ? `<div class="dash-warning-actions">
             <button type="button" class="btn-primary" data-warning-action="resolve" data-warning-idx="${i}">${escapeHtml(cfg.resolveLabel)}</button>
             <button type="button" class="btn-cancel" data-warning-action="contest" data-warning-idx="${i}">Contest this</button>
           </div>`
        : `<p class="dash-warning-note">This is already an open claim -- our team is reviewing it, no further action needed here.</p>`;
      return `<div class="dash-warning">
        <button type="button" class="dash-warning-head" data-warning-toggle="${i}">
          ${WARNING_ICON}<span>${escapeHtml(w.text)}</span>${CHEVRON_ICON}
        </button>
        <div class="dash-warning-body" id="warning-body-${i}" hidden>${body}</div>
      </div>`;
    })
    .join("");

  // Messages from ops -- same list regardless of Telegram delivery, so a
  // clarification request is never lost just because Telegram wasn't
  // linked or delivery failed (see MessageOut's docstring in browser_api.py).
  document.getElementById("dash-messages").innerHTML = messages.length
    ? messages
        .map(
          (m) =>
            `<div class="dash-msg-card"><div><p class="dash-msg-text">${escapeHtml(m.message)}</p><p class="dash-msg-meta">${fmtDateTime(m.created_at)}</p></div><span class="dash-msg-badge ${m.delivered_via_telegram ? "yes" : "no"}">${m.delivered_via_telegram ? "Sent via Telegram" : "Telegram not linked"}</span></div>`
        )
        .join("")
    : `<p class="dash-no-data">No messages from us yet.</p>`;

  // Hero identity
  document.getElementById("dash-avatar").textContent = initials(account.borrower_name);
  document.getElementById("dash-borrower-name").textContent = account.borrower_name;
  document.getElementById("dash-business-name").textContent = `${account.business_name} · ${account.account_id}`;
  const $risk = document.getElementById("dash-risk-chip");
  $risk.textContent = `${account.risk_tier} risk`;
  $risk.className = `risk-chip ${account.risk_tier}`;

  // Progress ring -- fraction of the loan term elapsed so far.
  const fraction = account.tenure_months > 0 ? (account.tenure_months - account.months_remaining) / account.tenure_months : 0;
  const ringColor =
    account.risk_tier === "high" ? "var(--danger)" : account.risk_tier === "medium" ? "var(--warning)" : "var(--success)";
  document.getElementById("dash-ring-stack").innerHTML = ringChartSvg(fraction, ringColor);
  document.getElementById("dash-ring-value").textContent = account.months_remaining;
  document.getElementById("dash-tenure").textContent = account.tenure_months;

  // Loan snapshot stats -- only surface the conditional fields when the
  // backend actually sent a value for them (interest_rate_pct is null
  // until a signed agreement is parsed; late_fee_amount is null unless
  // late_fee_applicable is true).
  const stats = [
    ["Principal", fmtInr(account.principal_amount)],
    ["EMI amount", fmtInr(account.emi_amount)],
    ["Next due", fmtDate(account.emi_due_date)],
    ["Outstanding", fmtInr(account.outstanding_balance_approx)],
    ["Days past due", String(account.days_past_due), account.days_past_due > 0 ? "flagged" : "ok"],
    ["NACH mandate", account.nach_mandate_active ? "Active" : "Inactive", account.nach_mandate_active ? "ok" : "flagged"],
  ];
  if (account.interest_rate_pct != null) stats.push(["Interest rate", `${account.interest_rate_pct}% p.a.`]);
  if (account.late_fee_applicable) stats.push(["Late fee", fmtInr(account.late_fee_amount), "flagged"]);
  document.getElementById("dash-hero-stats").innerHTML = stats
    .map(
      ([label, value, cls]) =>
        `<div><div class="hero-stat-label">${escapeHtml(label)}</div><div class="hero-stat-value${cls ? " " + cls : ""}">${escapeHtml(value)}</div></div>`
    )
    .join("");

  // Payment timeline -- real history then projected EMIs, one chronological list.
  document.getElementById("dash-timeline").innerHTML = timeline.length
    ? timeline
        .map(
          (t) =>
            `<div class="dtl-row"><div class="dtl-dot ${t.status}"></div><div class="dtl-info"><b>${fmtInr(t.amount)}</b> — ${escapeHtml(t.label)}<div class="dtl-meta">${fmtDate(t.date)}</div></div></div>`
        )
        .join("")
    : `<p class="dash-no-data">No payment history and no scheduled EMIs.</p>`;

  // This account's own escalation history, newest first (server order).
  document.getElementById("dash-escalations").innerHTML = escalations.length
    ? escalations
        .map(
          (e) =>
            `<div class="esc-card"><div><p class="esc-reason">${escapeHtml(e.reason)}</p><p class="esc-meta">Requested ${fmtDateTime(e.created_at)}${e.resolved_at ? ` · resolved ${fmtDateTime(e.resolved_at)}` : ""}</p></div><span class="esc-status ${e.status}">${escapeHtml(STATUS_LABELS[e.status] || e.status)}</span></div>`
        )
        .join("")
    : `<p class="dash-no-data">No requests raised yet.</p>`;

  // Documents -- real download links, scoped to this conversation's verified account.
  document.getElementById("dash-documents").innerHTML = documents.length
    ? documents
        .map(
          (d) =>
            `<div class="doc-row">${DOC_ICON}<div class="doc-info"><div class="doc-name">${escapeHtml(d.filename)}</div><div class="doc-meta">${fmtFileSize(d.size_bytes)} · uploaded ${fmtDateTime(d.uploaded_at)}</div></div><a class="doc-download" href="/conversations/${state.conversationId}/documents/${encodeURIComponent(d.filename)}" download>Download</a></div>`
        )
        .join("")
    : `<p class="dash-no-data">No documents uploaded yet.</p>`;
}

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
  const warning = _lastWarnings[Number(actionBtn.dataset.warningIdx)];
  const cfg = warning && WARNING_ACTIONS[warning.label];
  if (!cfg) return;
  const $result = document.getElementById("warnings-action-result");
  actionBtn.disabled = true;
  try {
    if (actionBtn.dataset.warningAction === "resolve") {
      const result = await cfg.resolve(_lastAccount);
      showActionResult($result, cfg.resolveResultHtml(result), true);
      if (cfg.refreshAfterResolve) loadDashboard();
    } else {
      const result = await postDispute(state.conversationId, cfg.contestReason(warning.text));
      showActionResult(
        $result,
        escapeHtml(
          result.already_open
            ? "You already have an open dispute — our team is reviewing it."
            : "Dispute raised — our team will review it."
        ),
        true
      );
      loadDashboard(); // a "disputed" warning + a new entry in the ops-side Disputes section just appeared
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

// Payment timeline can run to dozens of scheduled EMIs on a long tenure --
// collapsible so it doesn't dominate the dashboard by default weight, but
// starts open since it's real, relevant data, not something to hide.
const $timelineToggleBtn = document.getElementById("timeline-toggle-btn");
const $dashTimeline = document.getElementById("dash-timeline");
$timelineToggleBtn.addEventListener("click", () => {
  const expanded = $timelineToggleBtn.getAttribute("aria-expanded") === "true";
  $timelineToggleBtn.setAttribute("aria-expanded", String(!expanded));
  $dashTimeline.hidden = expanded;
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

const { $form: $disputeForm, $result: $disputeResult } = wireActionToggle("action-dispute-btn", "dispute-form", "dispute-result");
document.getElementById("dispute-cancel-btn").addEventListener("click", () => { $disputeForm.hidden = true; });
$disputeForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const $reasonInput = document.getElementById("dispute-reason");
  const reason = $reasonInput.value.trim();
  if (!reason) {
    showActionResult($disputeResult, "Enter a reason first.", false);
    return;
  }
  const $btn = document.getElementById("dispute-submit-btn");
  $btn.disabled = true;
  try {
    const result = await postDispute(state.conversationId, reason);
    showActionResult(
      $disputeResult,
      escapeHtml(
        result.already_open
          ? "You already have an open dispute — our team is reviewing it."
          : "Dispute raised — our team will review it."
      ),
      true
    );
    $disputeForm.hidden = true;
    $reasonInput.value = "";
    loadDashboard(); // dispute_open + warnings just changed
  } catch (err) {
    showActionResult($disputeResult, friendlyActionError(err), false);
  } finally {
    $btn.disabled = false;
  }
});

const { $form: $agentForm, $result: $agentResult } = wireActionToggle("action-agent-btn", "agent-form", "agent-result");
document.getElementById("agent-cancel-btn").addEventListener("click", () => { $agentForm.hidden = true; });
$agentForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const $reasonInput = document.getElementById("agent-reason");
  const reason = $reasonInput.value.trim();
  const $btn = document.getElementById("agent-submit-btn");
  $btn.disabled = true;
  try {
    const result = await postAgent(state.conversationId, reason);
    showActionResult(
      $agentResult,
      escapeHtml(`Request sent — our team has been notified (status: ${STATUS_LABELS[result.status] || result.status}).`),
      true
    );
    $agentForm.hidden = true;
    $reasonInput.value = "";
    loadDashboard(); // a new escalation just appeared in the list
  } catch (err) {
    showActionResult($agentResult, friendlyActionError(err), false);
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
    const result = await postPaymentLink(state.conversationId, amount);
    showActionResult(
      $paymentResult,
      `Payment link for ${escapeHtml(fmtInr(result.amount))}:<br><a href="${escapeHtml(result.payment_link)}" target="_blank" rel="noopener noreferrer">${escapeHtml(result.payment_link)}</a>`,
      true
    );
    // No account/escalation/document field changes as a result of this
    // action (see browser_api.py) -- nothing on the dashboard to refresh.
  } catch (err) {
    showActionResult($paymentResult, friendlyActionError(err), false);
  } finally {
    $btn.disabled = false;
  }
});

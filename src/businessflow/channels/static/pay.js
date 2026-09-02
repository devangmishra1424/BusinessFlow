"use strict";

// The token is the last path segment of /pay/{token} -- this page is a
// standalone destination (reached from a Telegram button or a copied
// link), not part of the chat SPA's own routing, so it reads directly
// from location.pathname rather than any app state.
const TOKEN = location.pathname.split("/").filter(Boolean).pop();

function fmtInr(n) {
  return "₹" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

function fmtDate(d) {
  return new Date(`${d}T00:00:00`).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

function show(id) {
  ["pay-loading", "pay-pending", "pay-success", "pay-error"].forEach((s) => {
    document.getElementById(s).hidden = s !== id;
  });
}

function showError(text) {
  document.getElementById("pay-error-text").textContent = text;
  show("pay-error");
}

async function loadInfo() {
  let res;
  try {
    res = await fetch(`/pay/${TOKEN}/info`);
  } catch (_) {
    showError("Couldn't reach BusinessFlow -- check your connection and reload this page.");
    return;
  }
  if (res.status === 404) {
    showError("This payment link isn't valid.");
    return;
  }
  if (!res.ok) {
    showError("Something went wrong loading this payment link -- please try again.");
    return;
  }
  const info = await res.json();
  if (info.status === "used") {
    showError("This payment link has already been used.");
    return;
  }
  if (info.status === "expired") {
    showError("This payment link has expired -- ask for a new one.");
    return;
  }
  document.getElementById("pay-amount").textContent = fmtInr(info.amount);
  document.getElementById("pay-business").textContent = `${info.business_name} · ${info.borrower_name}`;

  // Three cases, compared against what's actually due this cycle (already
  // net of any earlier credit -- see PaymentTokenInfoOut.emi_amount_due):
  // matches (or nothing to compare against) -> the plain Confirm button
  // below handles it as always; short of it -> ask before confirming at
  // all, since store.record_payment requires an answer for this case;
  // more than it -> just a heads-up, no question needed, the excess is
  // credited automatically.
  const dueThisCycle = info.emi_amount_due;
  const confirmBtn = document.getElementById("pay-confirm-btn");
  const extraQuestion = document.getElementById("pay-extra-question");
  const overpayNote = document.getElementById("pay-overpay-note");

  if (typeof dueThisCycle === "number" && info.amount + 0.01 < dueThisCycle) {
    confirmBtn.hidden = true;
    document.getElementById("pay-extra-note").textContent =
      `${fmtInr(info.amount)} is less than the ${fmtInr(dueThisCycle)} due this cycle.`;
    extraQuestion.hidden = false;
  } else if (typeof dueThisCycle === "number" && info.amount - 0.01 > dueThisCycle) {
    const excess = info.amount - dueThisCycle;
    overpayNote.textContent =
      `This is ${fmtInr(excess)} more than the ${fmtInr(dueThisCycle)} due this cycle -- ` +
      `the extra will be credited toward your next EMI automatically.`;
    overpayNote.hidden = false;
  }

  show("pay-pending");
}

async function confirmPayment(applyExtraToNext) {
  const btn = document.getElementById("pay-confirm-btn");
  const busy = applyExtraToNext === null ? btn : null;
  if (busy) {
    busy.disabled = true;
    busy.textContent = "Confirming…";
  }
  document.querySelectorAll("#pay-extra-yes-btn, #pay-extra-no-btn").forEach((b) => (b.disabled = true));

  const body = applyExtraToNext === null ? undefined : JSON.stringify({ apply_extra_to_next: applyExtraToNext });
  let res;
  try {
    res = await fetch(`/pay/${TOKEN}/confirm`, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body,
    });
  } catch (_) {
    if (busy) {
      busy.disabled = false;
      busy.textContent = "Confirm payment";
    }
    document.querySelectorAll("#pay-extra-yes-btn, #pay-extra-no-btn").forEach((b) => (b.disabled = false));
    showError("Couldn't reach BusinessFlow -- check your connection and try again.");
    return;
  }
  if (!res.ok) {
    let detail = "Something went wrong confirming this payment.";
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {}
    showError(detail);
    return;
  }
  const result = await res.json();
  document.getElementById("pay-success-amount").textContent = fmtInr(result.amount);
  const nextLine = document.getElementById("pay-success-next");
  if (result.kind === "extra_unapplied") {
    nextLine.textContent = "Recorded as an extra payment -- your EMI schedule is unchanged.";
  } else if (result.kind === "extra_applied") {
    nextLine.textContent = `Credited toward your next EMI -- you'll owe ${fmtInr(result.pending_emi_credit)} less next cycle.`;
  } else if (result.kind === "overpayment_applied") {
    nextLine.textContent =
      `${result.months_remaining} months remaining · next EMI due ${fmtDate(result.next_emi_due_date)}, ` +
      `reduced by ${fmtInr(result.pending_emi_credit)} from this overpayment.`;
  } else {
    nextLine.textContent =
      result.months_remaining > 0
        ? `${result.months_remaining} months remaining · next EMI due ${fmtDate(result.next_emi_due_date)}`
        : "Your loan is now fully repaid.";
  }
  show("pay-success");
}

document.getElementById("pay-confirm-btn").addEventListener("click", () => confirmPayment(null));
document.getElementById("pay-extra-yes-btn").addEventListener("click", () => confirmPayment(true));
document.getElementById("pay-extra-no-btn").addEventListener("click", () => confirmPayment(false));

loadInfo();

"use strict";

// The token is the last path segment of /pay/{token} -- this page is a
// standalone destination (reached from a Telegram button or a copied
// link), not part of the chat SPA's own routing, so it reads directly
// from location.pathname rather than any app state.
const TOKEN = location.pathname.split("/").filter(Boolean).pop();

const EPS = 0.01;

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

// What the confirm button should send for the page's current case. Set once
// by loadInfo; "plain" is the default (nothing to decide).
let confirmBody = null;

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

  // What this payment covers, in the order it is applied (PaymentTokenInfoOut.breakdown, the server's own
  // accounts/dues.py allocate_payment -- the same split the payment will get):
  //   covers what's owed, nothing extra -> a plain payment: one button, and it says what it pays (EMI, late fee).
  //     The fee is part of what's owed, NOT "extra" -- this page used to call it an overpayment.
  //   short of this month's EMI -> a part-payment toward it: one button, and it says what is still due afterwards.
  //   more than what's owed (by choice) -> ask how to use the extra amount.
  const b = info.breakdown;
  const note = document.getElementById("pay-note");
  const confirmBtn = document.getElementById("pay-confirm-btn");

  if (b) {
    const parts = [];
    if (b.to_emi > EPS && b.emi_retired) parts.push(`your EMI of ${fmtInr(b.to_emi)}`);
    for (const c of b.to_charges) parts.push(`the ${fmtInr(c.amount)} ${c.label.charAt(0).toLowerCase()}${c.label.slice(1)}`);
    const covers = parts.length ? `This covers ${parts.join(" and ")}.` : "";
    if (b.short_of_emi > EPS) {
      note.textContent = `${parts.length ? covers + " " : ""}This pays ${fmtInr(b.to_emi)} toward this month's EMI; ${fmtInr(b.short_of_emi)} of it will still be due afterwards.`;
      note.hidden = false;
      confirmBody = { apply_extra_to_next: true };
      confirmBtn.firstChild.textContent = `Pay ${fmtInr(info.amount)} `;
    } else if (b.extra > EPS) {
      confirmBtn.hidden = true;
      document.getElementById("pay-overpay-note-text").textContent =
        `${covers ? covers + " " : ""}That leaves ${fmtInr(b.extra)} extra.`;
      document.getElementById("pay-overpay-question").hidden = false;
    } else if (b.to_charges.length || parts.length > 1) {
      note.textContent = covers;
      note.hidden = false;
    }
  }

  show("pay-pending");
}

function setBusy(on) {
  document.querySelectorAll("#pay-confirm-btn, .pay-scheme-actions button").forEach((b) => (b.disabled = on));
}

function successLines(result) {
  const lines = [];
  const stillDue = Math.max(0, (result.emi_amount || 0) - (result.pending_emi_credit || 0));
  if (result.kind === "extra_unapplied") {
    lines.push("Recorded as a separate payment -- your EMI schedule is unchanged.");
  } else if (result.kind === "extra_applied") {
    lines.push("Received toward this month's EMI.");
    return { next: lines.join(" "), stillDue };
  } else if (result.kind === "overpayment_applied") {
    lines.push(
      `${result.months_remaining} months remaining · next EMI due ${fmtDate(result.next_emi_due_date)}, ` +
        `reduced by ${fmtInr(result.pending_emi_credit)} from the extra amount.`
    );
  } else if (result.kind === "principal_prepayment_reduce_emi") {
    lines.push(`Extra applied to your loan. Your new EMI is ${fmtInr(result.emi_amount)} with ${result.months_remaining} months remaining.`);
  } else if (result.kind === "principal_prepayment_reduce_tenure") {
    lines.push(`Extra applied to your loan. ${result.months_remaining} months remaining.`);
  } else {
    // A payment that went only to fees (kind "charge") paid the fee; one that also paid the EMI included it.
    if (result.late_fee_paid > 0) lines.push(result.kind === "charge" ? `This paid the ${fmtInr(result.late_fee_paid)} late fee.` : `Includes the ${fmtInr(result.late_fee_paid)} late fee.`);
    if (result.late_fee_waived_paid_in_grace) lines.push("The late fee for this EMI was removed: you paid within the grace period.");
    lines.push(
      result.months_remaining > 0
        ? `${result.months_remaining} months remaining · next EMI due ${fmtDate(result.next_emi_due_date)}`
        : "Your loan is now fully repaid."
    );
  }
  return { next: lines.join(" "), stillDue: 0 };
}

async function confirmPayment(body) {
  setBusy(true);
  const btn = document.getElementById("pay-confirm-btn");
  const originalLabel = btn.firstChild.textContent;
  if (!btn.hidden) btn.firstChild.textContent = "Confirming… ";

  let res;
  try {
    res = await fetch(`/pay/${TOKEN}/confirm`, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (_) {
    setBusy(false);
    btn.firstChild.textContent = originalLabel;
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
  const { next, stillDue } = successLines(result);
  document.getElementById("pay-success-next").textContent = next;
  const $still = document.getElementById("pay-success-still-due");
  if (stillDue > 0) {
    $still.textContent = `${fmtInr(stillDue)} is still due for this EMI.`;
    $still.hidden = false;
  } else {
    $still.hidden = true;
  }
  show("pay-success");
}

document.getElementById("pay-confirm-btn").addEventListener("click", () => confirmPayment(confirmBody));
document.getElementById("pay-scheme-next-emi")?.addEventListener("click", () => confirmPayment({ payment_scheme: "credit_next_emi" }));
document.getElementById("pay-scheme-reduce-emi")?.addEventListener("click", () => confirmPayment({ payment_scheme: "reduce_emi" }));
document.getElementById("pay-scheme-reduce-tenure")?.addEventListener("click", () => confirmPayment({ payment_scheme: "reduce_tenure" }));

loadInfo();

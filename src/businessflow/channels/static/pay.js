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

  // Four cases, compared against what's actually due this cycle (EMI less any
  // credit -- PaymentTokenInfoOut.emi_amount_due) and the late fee that applies
  // right now (late_fee_due):
  //   exactly what's due, or what's due + the fee -> a plain payment: one
  //     button, no questions. The fee is part of what's owed, NOT "extra" --
  //     the overdue "Pay" button and every reminder mint EMI + fee, and this
  //     page used to call that fee an overpayment and ask how to apply it.
  //   less than what's due -> a part-payment toward THIS month's EMI: one
  //     button, and it says plainly what is still due afterwards.
  //   more (by choice) -> ask how to use the extra amount.
  const due = info.emi_amount_due;
  const fee = info.late_fee_due || 0;
  const note = document.getElementById("pay-note");
  const confirmBtn = document.getElementById("pay-confirm-btn");

  if (typeof due === "number") {
    if (Math.abs(info.amount - due) <= EPS) {
      // plain
    } else if (fee > 0 && Math.abs(info.amount - (due + fee)) <= EPS) {
      note.textContent = `This covers your EMI of ${fmtInr(due)} and the ${fmtInr(fee)} late fee.`;
      note.hidden = false;
    } else if (info.amount + EPS < due) {
      note.textContent = `This pays part of this month's EMI of ${fmtInr(due)}. ${fmtInr(due - info.amount)} will still be due afterwards.`;
      note.hidden = false;
      confirmBody = { apply_extra_to_next: true };
      confirmBtn.firstChild.textContent = `Pay ${fmtInr(info.amount)} `;
    } else {
      const excess = info.amount - due;
      confirmBtn.hidden = true;
      document.getElementById("pay-overpay-note-text").textContent =
        `This is ${fmtInr(excess)} more than the ${fmtInr(due)} due this cycle.`;
      document.getElementById("pay-overpay-question").hidden = false;
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
    if (result.late_fee_paid > 0) lines.push(`Includes the ${fmtInr(result.late_fee_paid)} late fee.`);
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

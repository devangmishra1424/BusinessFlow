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
  show("pay-pending");
}

document.getElementById("pay-confirm-btn").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  btn.textContent = "Confirming…";
  let res;
  try {
    res = await fetch(`/pay/${TOKEN}/confirm`, { method: "POST" });
  } catch (_) {
    btn.disabled = false;
    btn.textContent = "Confirm payment";
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
  document.getElementById("pay-success-next").textContent =
    result.months_remaining > 0
      ? `${result.months_remaining} months remaining · next EMI due ${fmtDate(result.next_emi_due_date)}`
      : "Your loan is now fully repaid.";
  show("pay-success");
});

loadInfo();

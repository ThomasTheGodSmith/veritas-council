// Market Council website. Talks to web/server.py; every number shown comes from the council's output.
const $ = (s, el = document) => el.querySelector(s);
const NAMES = {
  momentum: "Momentum", mean_reversion: "Mean Reversion", ml_forecaster: "ML Forecaster",
  risk_officer: "Risk Officer", news_reaction: "News-Reaction", pattern_oracle: "Pattern Oracle",
  raw_reader: "Raw Reader", market_regime: "Market Regime", macro: "Macro", dow_breadth: "Dow Breadth",
  devils_advocate: "Devil's Advocate",
};
const ORDER = ["Always buy", "Random, same buy rate", "Always buy + stop-loss", "Council"];
const state = { programs: [], last: null, lastRequest: null };

// The prediction covers the next N trading days after the decision day; this is when it can be checked.
function addTradingDays(isoDate, n) {
  const d = new Date(isoDate + "T00:00:00");
  let left = n;
  while (left > 0) {
    d.setDate(d.getDate() + 1);
    if (d.getDay() !== 0 && d.getDay() !== 6) left--;
  }
  return d.toLocaleDateString("en-SG", { day: "numeric", month: "short", year: "numeric" });
}

const money = (v, d = 0) => "S$" + Number(v).toLocaleString("en-SG", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 0) => (v * 100).toFixed(d) + "%";
const signed = (v) => (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(0) + "%";
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const program = () => state.programs.find((p) => p.key === $("#program").value);

// --------------------------------------------------------------------------- guilloché rosette
function rosette(seed = 1) {
  // a banknote-style guilloché: a few closed hypotrochoids, drawn once (about 2,000 points in total)
  const R = 96, parts = [];
  for (let k = 0; k < 5; k++) {
    const r = 12 + 3 * k + seed, d = 34 + 7 * k, n = 420;
    let path = "";
    for (let i = 0; i <= n; i++) {
      const t = (i / n) * Math.PI * 2 * r;            // integer r: the curve closes after r turns
      const x = (R - r) * Math.cos(t) + d * Math.cos(((R - r) / r) * t);
      const y = (R - r) * Math.sin(t) - d * Math.sin(((R - r) / r) * t);
      path += (i ? "L" : "M") + (210 + x * 1.4).toFixed(1) + " " + (210 + y * 1.4).toFixed(1);
    }
    parts.push(`<path d="${path}"/>`);
  }
  return `<svg class="rosette" viewBox="0 0 420 420" aria-hidden="true">${parts.join("")}</svg>`;
}

// --------------------------------------------------------------------------- navigation
function show(view) {
  document.querySelectorAll("section.view").forEach((s) => (s.hidden = s.dataset.view !== view));
  document.querySelectorAll(".views a").forEach((a) => a.toggleAttribute("aria-current", a.dataset.view === view) || a.setAttribute("aria-current", a.dataset.view === view ? "page" : "false"));
  document.querySelectorAll(".views a").forEach((a) => { if (a.dataset.view !== view) a.removeAttribute("aria-current"); });
  if (view === "portfolio") loadPortfolio();
}
window.addEventListener("hashchange", () => show(location.hash.slice(1) || "ask"));

// --------------------------------------------------------------------------- councils
async function init() {
  const progs = await fetch("/api/programs").then((r) => r.json());
  state.programs = progs.programs;
  $("#disclaimer").textContent = progs.disclaimer;
  $("#program").innerHTML = state.programs
    .map((p) => `<option value="${p.key}" ${p.ready ? "" : "disabled"}>${esc(p.label)}: ${esc(p.detail)}${p.ready ? "" : " (not trained)"}</option>`)
    .join("");
  $("#program").addEventListener("change", onProgram);
  onProgram();
  show(location.hash.slice(1) || "ask");
}

function onProgram() {
  const p = program();
  $("#ticker").innerHTML = p.tickers.map((t) => `<option value="${esc(t.ticker)}">${esc(t.name)}${t.ticker !== "^DJI" ? " (" + esc(t.ticker) + ")" : ""}</option>`).join("");
  $("#ticker").disabled = p.tickers.length === 1;
  $("#council-note").textContent = p.strategy
    ? `The ${p.label} council. Settings chosen on its own training years: ${p.strategy}.`
    : `The ${p.label} council.`;
}

// --------------------------------------------------------------------------- ask
$("#plan").addEventListener("submit", (e) => {
  e.preventDefault();
  const request = {
    ticker: $("#ticker").value,
    amount: Number($("#amount").value),
    max_loss: Number($("#max_loss").value),
    horizon_days: Number($("#horizon").value),
    avoid_sectors: $("#avoid").value.split(",").map((s) => s.trim()).filter(Boolean),
    thinks_up: document.querySelector('input[name="dir"]:checked').value === "up",
  };
  request.idea_text = request.thinks_up ? "I think it'll go up" : "I think it'll go down";
  ask(request, null);
});

async function ask(request, declineReason) {
  state.lastRequest = request;
  const btn = $("#ask");
  btn.disabled = true;
  const out = $("#outcome");
  out.innerHTML = $("#tpl-loading").innerHTML;
  $(".seats", out).innerHTML = Array.from({ length: 11 }, (_, i) => `<li style="animation-delay:${i * 0.12}s"></li>`).join("");
  try {
    const r = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ program: program().key, request, decline_reason: declineReason }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "The council couldn't run.");
    state.last = data;
    renderRun(data);
  } catch (err) {
    out.innerHTML = `<div class="toast bad"><b>The council couldn't run.</b> ${esc(err.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
}

function renderRun(data) {
  const r = data.result, j = r.judge, req = r.request, risk = r.risk || {};
  const plan = (r.mediator.plans || []).find((p) => p.id === j.plan_id);
  const name = $("#ticker").selectedOptions[0]?.textContent || req.ticker;
  const denom = j.decision === "buy" ? `Buy ${money(j.amount, 2)}` : j.decision === "wait" ? "Wait" : "Don't buy";
  const stop = plan && plan.stop_loss_pct && j.decision === "buy" ? plan.stop_loss_pct : null;
  const facts = [
    ["Data up to", r.data_as_of],
    ["Buy at the open on", r.decision_date],
    ["Prediction covers", `${req.horizon_days} trading days`],
    ["You'll know if it was right around", addTradingDays(r.decision_date, req.horizon_days)],
    ["Chance of a rise", pct(r.mediator.p_council)],
    ["1-in-20 bad period", risk.var95_unprotected_pct != null ? "−" + pct(risk.var95_unprotected_pct) : risk.var95_used_pct != null ? "−" + pct(risk.var95_used_pct) : "–"],
    ...(stop ? [["Stop-loss", pct(stop, 1) + " below the price"]] : []),
  ];
  const signals = r.signals.map((s) => {
    const p = s.probability_up, down = p < 0.5;
    return `<li>
      <span class="name">${esc(NAMES[s.agent] || s.agent)}</span>
      <span class="meter ${down ? "down" : ""}" title="${pct(p)} chance of a rise"><i style="width:${(p * 100).toFixed(1)}%"></i><b></b></span>
      <span class="pct">${pct(p)} up</span>
      <span class="tag ${s.user_idea_verdict === "supports" ? "good" : "bad"}">${s.user_idea_verdict === "supports" ? "supports you" : "challenges you"}</span>
    </li>`;
  }).join("");
  const rules = (r.rules || []).map((c) => `<li class="${c.passed ? "" : "fail"}">${esc(c.text)}</li>`).join("");
  const debate = (r.crossexam || []).slice(0, 4).map((d) => `<li>${esc(d.summary)}</li>`).join("");
  const da = r.devils_advocate && r.devils_advocate.text ? `<p>${esc(r.devils_advocate.text)}</p>` : "";
  const change = (j.would_change_if || []).map((w) => `<li>${esc(w)}</li>`).join("");
  const plans = (r.mediator.plans || []).map((p) => `<li>${esc(p.id)}: ${esc(p.action)}${p.amount ? " " + money(p.amount, 2) : ""}${p.checklist === "fails" ? " (fails a rule)" : ""}</li>`).join("");

  $("#outcome").innerHTML = `
    <article class="note ${j.decision}">
      ${rosette(j.decision === "buy" ? 1 : j.decision === "wait" ? 2 : 3)}
      <p class="context">${esc(program().label)} council on ${esc(name)}</p>
      <h2 class="denom">${denom}</h2>
      <dl class="facts">${facts.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>
      <p class="reason">${esc(j.reason)}</p>
      <div class="actions">
        ${j.decision === "buy" && j.amount > 0 ? `<button class="primary approve" id="approve">Approve and invest (paper)</button>` : ""}
        <a class="secondary" href="${esc(data.report_url)}" target="_blank" rel="noopener">Open the full report</a>
      </div>
      <form class="decline-row" id="decline">
        <input id="decline-reason" placeholder="Not happy? Say why, e.g. too risky, too much money, no tech" aria-label="Reason for declining">
        <button class="secondary" type="submit">Decline and ask again</button>
      </form>
      <div id="order-msg"></div>
    </article>

    <section class="block">
      <h2>The council: eleven analysts</h2>
      <p class="hint">Each bar is that analyst's chance of a rise over the next ${req.horizon_days} trading days. The line marks 50%.</p>
      <ul class="analysts">${signals}</ul>
    </section>

    <div class="cols">
      <section class="block"><h2>Your rules</h2><ul class="checks">${rules}</ul></section>
      <section class="block"><h2>Plans the Mediator considered</h2><ul class="lines">${plans}</ul></section>
    </div>
    <div class="cols">
      <section class="block"><h2>The debate</h2>${da}${debate ? `<ul class="lines">${debate}</ul>` : `<p class="hint">No two analysts disagreed today.</p>`}</section>
      <section class="block"><h2>What would change the decision</h2>${change ? `<ul class="lines">${change}</ul>` : `<p class="hint">Nothing listed.</p>`}</section>
    </div>
    ${(r.notes || []).length ? `<p class="hint small">Notes: ${esc(r.notes.join(" "))}</p>` : ""}
  `;
  $("#approve")?.addEventListener("click", approve);
  $("#decline").addEventListener("submit", (e) => {
    e.preventDefault();
    const reason = $("#decline-reason").value.trim() || "too risky";
    ask(state.lastRequest, reason);
  });
}

async function approve() {
  const btn = $("#approve");
  btn.disabled = true;
  const r = await fetch("/api/approve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ result: state.last.result }) });
  const data = await r.json();
  const ok = data.status === "filled" || data.status === "placed" || data.status === "submitted";
  $("#order-msg").innerHTML = `<p class="toast ${ok ? "" : "bad"}">${esc(data.message || data.status)}</p>`;
  if (!ok) btn.disabled = false;
}

// --------------------------------------------------------------------------- portfolio
function table(el, rows, cols) {
  if (!rows || !rows.length) { el.innerHTML = `<tbody><tr><td class="hint">Nothing yet.</td></tr></tbody>`; return; }
  const keys = cols || Object.keys(rows[0]);
  el.innerHTML = `<thead><tr>${keys.map((k) => `<th>${esc(k.replace(/_/g, " "))}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${keys.map((k) => `<td>${esc(typeof r[k] === "number" ? r[k].toLocaleString("en-SG", { maximumFractionDigits: 2 }) : r[k])}</td>`).join("")}</tr>`).join("")}</tbody>`;
}
async function loadPortfolio() {
  const d = await fetch("/api/portfolio").then((r) => r.json());
  table($("#positions"), d.positions);
  table($("#orders"), d.orders);
  table($("#pending"), d.pending);
}

init();

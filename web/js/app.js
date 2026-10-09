import { ApiBackend, DemoBackend } from "./backends.js";
import { MONTHS } from "./recommender.js";

const cfg = window.TRAVEL_CONFIG || { mode: "demo", modelUrl: "model/model.json" };
const backend = cfg.mode === "api" ? new ApiBackend(cfg.apiBase || "") : new DemoBackend();
const K = 5;

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const c of children) if (c != null) node.append(c);
  return node;
};

const state = {
  destinations: [],
  byId: new Map(),
  types: [],
  profile: null, // { name, preferences, ratings: [[id, rating], ...] }
  filters: { month: null, type: null, state: null },
  shown: new Set(),
  search: "",
};

function showError(err) {
  $("error").textContent = err.message || String(err);
  $("error").hidden = false;
}

function monthRange(months) {
  return months.map((m) => MONTHS[m - 1]).join(", ");
}

// ---------- stars ----------
function stars(name, current, onRate, small = false) {
  const wrap = el("span", { className: `stars${small ? " small" : ""}` });
  wrap.setAttribute("role", "group");
  wrap.setAttribute("aria-label", `Your rating for ${name}`);
  for (let v = 1; v <= 5; v++) {
    const b = el("button", { type: "button", className: `star${current >= v ? " on" : ""}`, textContent: "★" });
    b.setAttribute("aria-label", `Rate ${name} ${v} out of 5`);
    b.setAttribute("aria-pressed", String(current === v));
    b.addEventListener("click", () => onRate(v));
    b.addEventListener("mouseenter", () => wrap.querySelectorAll(".star").forEach((s, i) => s.classList.toggle("hover", i < v)));
    wrap.append(b);
  }
  wrap.addEventListener("mouseleave", () => wrap.querySelectorAll(".star").forEach((s) => s.classList.remove("hover")));
  return wrap;
}

// ---------- panel ----------
function renderWho() {
  const root = $("who");
  root.replaceChildren();
  if (backend.kind === "demo") {
    const group = el("div", { className: "who-options" });
    group.setAttribute("role", "radiogroup");
    for (const c of backend.choices()) {
      const input = el("input", { type: "radio", name: "who", value: c.id, checked: c.id === backend.active });
      input.addEventListener("change", async () => { await backend.select(c.id); await refreshProfile(); });
      group.append(el("label", { className: "who-option" }, input,
        el("span", {}, c.label, el("small", { textContent: c.detail }))));
    }
    root.append(group);
    return;
  }
  // API mode
  if (backend.signedIn) {
    const out = el("button", { type: "button", className: "text-button", textContent: "Sign out" });
    out.addEventListener("click", () => { backend.signOut(); boot(); });
    root.append(el("p", {}, el("strong", { textContent: state.profile?.name || "" })),
      el("p", { className: "account-switch" }, state.profile?.email || "", " ", out));
    return;
  }
  renderAccountForm(root, "register");
}

function renderAccountForm(root, mode) {
  root.replaceChildren();
  const isNew = mode === "register";
  const err = el("p", { className: "form-error", hidden: true });
  const name = el("input", { type: "text", id: "acct-name", required: true, autocomplete: "name" });
  const email = el("input", { type: "email", id: "acct-email", required: true, autocomplete: "email" });
  const password = el("input", {
    type: "password", id: "acct-password", required: true, minLength: isNew ? 8 : 1,
    autocomplete: isNew ? "new-password" : "current-password",
  });
  const form = el("form", { className: "account-form" },
    el("p", { className: "muted", textContent: isNew ? "Create a profile to save your ratings." : "Welcome back." }),
    isNew ? el("label", {}, "Name", name) : null,
    el("label", {}, "Email", email),
    el("label", {}, isNew ? "Password (8 characters or more)" : "Password", password),
    el("button", { type: "submit", className: "button", textContent: isNew ? "Create profile" : "Sign in" }),
    err);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      if (isNew) await backend.register({ name: name.value.trim(), email: email.value.trim(), password: password.value });
      else await backend.login({ email: email.value.trim(), password: password.value });
      await boot();
    } catch (ex) { err.textContent = ex.message; err.hidden = false; }
  });
  const swap = el("button", { type: "button", className: "text-button",
    textContent: isNew ? "Sign in instead" : "Create a profile instead" });
  swap.addEventListener("click", () => renderAccountForm(root, isNew ? "login" : "register"));
  root.append(form, el("p", { className: "account-switch" }, isNew ? "Have a profile? " : "New here? ", swap));
}

function renderInterests() {
  const root = $("interests");
  root.replaceChildren();
  const prefs = new Set(state.profile?.preferences || []);
  for (const t of state.types) {
    const b = el("button", { type: "button", className: "chip", textContent: t });
    b.setAttribute("aria-pressed", String(prefs.has(t)));
    b.disabled = !state.profile;
    b.addEventListener("click", async () => {
      prefs.has(t) ? prefs.delete(t) : prefs.add(t);
      const next = state.types.filter((x) => prefs.has(x));
      state.profile.preferences = next;
      renderInterests();
      await guarded(() => backend.setPreferences(next));
      await renderRecommendations();
    });
    root.append(b);
  }
}

function renderRated() {
  const list = $("rated");
  list.replaceChildren();
  const ratings = state.profile?.ratings || [];
  $("rated-count").textContent = ratings.length ? `(${ratings.length})` : "";
  $("clear-ratings").hidden = ratings.length === 0;
  if (!ratings.length) {
    list.append(el("li", { className: "rated-empty", textContent: "None yet. Rate a suggestion or search below." }));
    return;
  }
  for (const [id, r] of ratings) {
    const d = state.byId.get(id);
    if (!d) continue;
    const remove = el("button", { type: "button", className: "icon-button", textContent: "✕" });
    remove.setAttribute("aria-label", `Remove your rating for ${d.name}`);
    remove.addEventListener("click", () => changeRating(id, null));
    list.append(el("li", {}, el("span", { className: "name", textContent: d.name, title: d.name }),
      stars(d.name, r, (v) => changeRating(id, v), true), remove));
  }
}

// ---------- results ----------
const STRATEGY_TEXT = {
  popular: () => "Pick some interests or rate a place you've been to make these yours. For now, these are the places travellers rate highest.",
  content: () => "Based on the interests you picked. Rate places you've been to bring in travellers like you.",
  hybrid: (n, a) => `Mixing your interests with what travellers like you rated highly (${a}% of the weight), based on your ${n} rating${n === 1 ? "" : "s"}.`,
  cf: () => "Based on travellers whose ratings resemble yours.",
};

function ticket(item, fresh) {
  const d = item.destination;
  const node = $("ticket-tpl").content.firstElementChild.cloneNode(true);
  if (fresh) node.classList.add("fresh");
  node.querySelector(".ticket-name").textContent = d.name;
  node.querySelector(".t-state").textContent = d.state;
  node.querySelector(".t-type").textContent = d.type;
  node.querySelector(".ticket-reason").textContent = item.reason;

  const season = node.querySelector(".season");
  season.setAttribute("aria-label", `In season: ${monthRange(d.best_months)}`);
  MONTHS.forEach((m, i) => {
    const cell = el("span", { textContent: m[0], title: m });
    if (d.best_months.includes(i + 1)) cell.classList.add("in");
    if (state.filters.month === i + 1) cell.classList.add("picked");
    season.append(cell);
  });
  if (d.description) node.querySelector(".ticket-reason").after(el("p", { className: "ticket-desc", textContent: d.description }));

  const current = state.profile?.ratings.find(([id]) => id === d.id)?.[1] || 0;
  node.querySelector(".stars").replaceWith(stars(d.name, current, (v) => changeRating(d.id, v)));

  // The list is ordered by match score, so that is the headline number.
  node.querySelector(".stub-value").textContent = `${Math.round(item.score * 100)}%`;
  node.querySelector(".stub-label").textContent = item.predicted_rating != null
    ? `match, you'd rate it about ${item.predicted_rating.toFixed(1)}`
    : "match";
  return node;
}

let renderToken = 0;
async function renderRecommendations() {
  if (!state.profile) {
    $("tickets").replaceChildren();
    $("strategy").textContent = "Create a profile to see your picks.";
    $("blend").hidden = true;
    return;
  }
  const token = ++renderToken;
  let res;
  try {
    res = await backend.recommend({ k: K, filters: state.filters });
  } catch (e) { showError(e); return; }
  if (token !== renderToken) return; // a newer render started meanwhile
  $("error").hidden = true;

  const n = state.profile.ratings.length;
  $("strategy").textContent = STRATEGY_TEXT[res.strategy](n, Math.round(res.alpha * 100));
  $("blend").hidden = res.strategy === "popular";
  $("blend-cf").style.width = `${Math.round(res.alpha * 100)}%`;

  const ids = new Set(res.items.map((i) => i.destination.id));
  $("tickets").replaceChildren(...res.items.map((i) => ticket(i, !state.shown.has(i.destination.id))));
  state.shown = ids;
  $("empty").hidden = res.items.length > 0;
}

function renderCatalogue() {
  const q = state.search.trim().toLowerCase();
  const rated = new Map(state.profile?.ratings || []);
  const rows = state.destinations
    .filter((d) => !q || `${d.name} ${d.state} ${d.type}`.toLowerCase().includes(q))
    .map((d) => el("li", {},
      el("span", { className: "c-name" }, el("strong", { textContent: d.name, title: d.name }),
        el("span", { textContent: `${d.state}, ${d.type}` })),
      state.profile ? stars(d.name, rated.get(d.id) || 0, (v) => changeRating(d.id, v), true) : null));
  $("catalogue").replaceChildren(...rows);
  if (!rows.length) $("catalogue").append(el("li", { className: "muted", textContent: `No places match "${state.search}".` }));
}

// ---------- actions ----------
async function guarded(fn) {
  try { await fn(); } catch (e) { showError(e); }
}

async function changeRating(id, value) {
  const ratings = state.profile.ratings;
  const i = ratings.findIndex(([d]) => d === id);
  if (value == null) { if (i >= 0) ratings.splice(i, 1); } else if (i >= 0) ratings[i] = [id, value]; else ratings.push([id, value]);
  renderRated();
  renderCatalogue();
  await guarded(() => (value == null ? backend.unrate(id) : backend.rate(id, value)));
  await renderRecommendations();
}

async function refreshProfile() {
  state.profile = await backend.current();
  state.shown = new Set();
  renderWho();
  renderInterests();
  renderRated();
  renderCatalogue();
  await renderRecommendations();
}

function fillSelect(id, options, key, parse = (v) => v) {
  const select = $(id);
  select.length = 1;
  for (const [value, label] of options) select.append(el("option", { value, textContent: label }));
  select.addEventListener("change", async () => {
    state.filters[key] = select.value ? parse(select.value) : null;
    await renderRecommendations();
  });
}

function renderMeta(meta) {
  const parts = [];
  if (meta.n_ratings) parts.push(`trained on ${meta.n_ratings.toLocaleString("en-IN")} ratings`);
  if (meta.cf_model) parts.push(meta.cf_model);
  if (meta.generated_at) parts.push(`built ${new Date(meta.generated_at).toLocaleDateString("en-IN", { dateStyle: "medium" })}`);
  if (meta.model_version) parts.unshift(`Model ${meta.model_version}`);
  $("model-meta").textContent = parts.join("; ");
}

function renderMetrics(metrics) {
  const root = $("metrics");
  root.replaceChildren();
  if (!metrics?.test) return;
  const rows = Object.entries(metrics.test).map(([name, m]) => {
    const tr = el("tr", {},
      el("td", { textContent: name }),
      el("td", { textContent: m.precision.toFixed(3) }),
      el("td", { textContent: m.recall.toFixed(3) }),
      el("td", { textContent: m.ndcg.toFixed(3) }),
      el("td", { textContent: `${Math.round(m.coverage * 100)}%` }));
    if (name.includes("published")) tr.className = "active";
    return tr;
  });
  const head = el("tr", {}, ...["Model", "Precision@5", "Recall@5", "NDCG@5", "Coverage"].map((h) => el("th", { textContent: h, scope: "col" })));
  root.append(
    el("h3", { textContent: "How good are the picks?" }),
    el("p", { className: "muted", textContent: "Each simulated traveller's most recent 20% of trips were hidden from training. A pick counts as a hit if they later rated that place 4 or 5. Higher is better." }),
    el("div", { className: "metrics-wrap" }, el("table", { className: "metrics" }, el("thead", {}, head), el("tbody", {}, ...rows))));
}

let booted = false;
async function boot() {
  try {
    const { destinations, types, meta, metrics } = await backend.init(cfg.modelUrl);
    state.destinations = destinations;
    state.byId = new Map(destinations.map((d) => [d.id, d]));
    state.types = types;
    renderMeta(meta);
    renderMetrics(metrics);
    if (!booted) {
      booted = true;
      fillSelect("f-month", MONTHS.map((m, i) => [i + 1, m]), "month", Number);
      fillSelect("f-type", types.map((t) => [t, t]), "type");
      fillSelect("f-state", [...new Set(destinations.map((d) => d.state))].sort().map((s) => [s, s]), "state");
      $("search").addEventListener("input", (e) => { state.search = e.target.value; renderCatalogue(); });
      $("clear-ratings").addEventListener("click", async () => {
        state.profile.ratings = [];
        renderRated(); renderCatalogue();
        await guarded(() => backend.clearRatings());
        await renderRecommendations();
      });
    }
    const note = $("mode-note");
    note.hidden = false;
    note.textContent = backend.kind === "demo"
      ? "Demo version: the recommender runs in your browser and your ratings stay on this device."
      : "Connected to the live API. Ratings are saved to the database.";

    if (backend.kind === "api" && !backend.signedIn) {
      state.profile = null;
      renderWho(); renderInterests(); renderRated(); renderCatalogue();
      await renderRecommendations();
      return;
    }
    await refreshProfile();
  } catch (e) {
    showError(e);
  }
}

boot();

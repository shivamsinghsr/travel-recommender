// Browser port of recsys/hybrid.py. Keep the two in step: tests/test_parity.py
// runs both on the same inputs and fails if their rankings differ.

export const MONTHS = ["January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December"];

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const norm = (a) => Math.sqrt(dot(a, a));

/** Mirrors recsys.content.user_profile. `ratings` is an array of [destinationId, rating]. */
export function userProfile(model, preferences, ratings) {
  const { features: F, types, itemIndex } = model;
  const dim = F[0].length;
  const profile = new Array(dim).fill(0);

  const prefs = preferences.filter((p) => types.includes(p));
  if (prefs.length) {
    const stated = new Array(dim).fill(0);
    for (const p of prefs) stated[types.indexOf(p)] = 1;
    const n = norm(stated);
    for (let i = 0; i < dim; i++) profile[i] += stated[i] / n;
  }

  const rated = ratings.filter(([d]) => itemIndex.has(d)).map(([d, r]) => [itemIndex.get(d), r]);
  if (rated.length) {
    const learned = new Array(dim).fill(0);
    for (const [j, r] of rated) for (let i = 0; i < dim; i++) learned[i] += (r - 3) * F[j][i];
    const n = norm(learned);
    if (n > 0) {
      const w = rated.length / (rated.length + 3);
      for (let i = 0; i < dim; i++) profile[i] += (w * learned[i]) / n;
    }
  }
  const n = norm(profile);
  return n > 0 ? profile.map((v) => v / n) : null;
}

/** Solve (M) x = b with Gaussian elimination and partial pivoting (M is small and symmetric positive definite). */
function solve(M, b) {
  const n = b.length;
  const A = M.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = A[r][n];
    for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k];
    x[r] = s / A[r][r];
  }
  return x;
}

/** Mirrors recsys.mf.MatrixFactorization.predict: fold the user in, then score every item. */
export function mfPredict(model, ratings) {
  const { mu, item_bias: bi, Q, factors: f, reg, bias_reg: breg } = model.cf;
  const nItems = bi.length;
  const base = bi.map((b) => mu + b);
  const rated = ratings.filter(([d]) => model.itemIndex.has(d)).map(([d, r]) => [model.itemIndex.get(d), r]);
  if (!rated.length) return { pred: base.map((v) => Math.min(5, Math.max(1, v))), best: new Array(nItems).fill(-1) };

  // ridge regression for [p_u, b_u]:  (AᵀA + diag(reg)) x = Aᵀy,  A = [Q_rated | 1]
  const A = rated.map(([j]) => [...Q[j], 1]);
  const y = rated.map(([j, r]) => r - mu - bi[j]);
  const dim = f + 1;
  const M = Array.from({ length: dim }, (_, a) => Array.from({ length: dim }, (_, b) =>
    A.reduce((s, row) => s + row[a] * row[b], 0) + (a === b ? (a < f ? reg : breg) : 0)));
  const rhs = Array.from({ length: dim }, (_, a) => A.reduce((s, row, i) => s + row[a] * y[i], 0));
  const x = solve(M, rhs);
  const p = x.slice(0, f), bu = x[f];

  const pred = Q.map((q, j) => Math.min(5, Math.max(1, base[j] + bu + dot(q, p))));
  const unit = Q.map((q) => { const n = Math.max(norm(q), 1e-12); return q.map((v) => v / n); });
  const cols = rated.map(([j]) => j);
  const resid = rated.map(([j, r]) => r - (base[j] + bu));
  const best = unit.map((u) => {
    let bestVal = -Infinity, bestLocal = 0;
    cols.forEach((c, i) => { const v = dot(u, unit[c]) * resid[i]; if (v > bestVal) { bestVal = v; bestLocal = i; } });
    return bestVal > 0 ? cols[bestLocal] : -1;
  });
  return { pred, best };
}

/** Mirrors recsys.item_knn.ItemKNN.predict. Returns {pred, best} arrays over the item axis. */
export function itemKnnPredict(model, ratings) {
  const { baseline, sim, k } = model.cf;
  const nItems = baseline.length;
  const rated = ratings.filter(([d]) => model.itemIndex.has(d)).map(([d, r]) => [model.itemIndex.get(d), r]);
  if (!rated.length) return { pred: baseline.slice(), best: new Array(nItems).fill(-1) };

  const cols = rated.map(([j]) => j);
  const dev = rated.map(([j, r]) => r - baseline[j]);
  const offset = dev.reduce((s, v) => s + v, 0) / (dev.length + 2);
  const resid = dev.map((v) => v - offset);

  const pred = new Array(nItems);
  const best = new Array(nItems);
  for (let t = 0; t < nItems; t++) {
    let S = cols.map((c) => sim[t][c]);
    if (cols.length > k) {
      // keep the k most similar rated items; stable sort matches numpy's kind="stable"
      const keep = new Set(S.map((v, i) => [Math.abs(v), i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(([, i]) => i));
      S = S.map((v, i) => (keep.has(i) ? v : 0));
    }
    let support = 0, num = 0, bestVal = -Infinity, bestLocal = 0;
    S.forEach((s, i) => {
      support += Math.abs(s);
      num += s * resid[i];
      const c = s * resid[i];
      if (c > bestVal) { bestVal = c; bestLocal = i; }
    });
    const p = baseline[t] + offset + num / (support + 1);
    pred[t] = Math.min(5, Math.max(1, p));
    best[t] = bestVal > 0 ? cols[bestLocal] : -1;
  }
  return { pred, best };
}

function reason(model, d, j, ratingMap, prefs, filters, alpha, best) {
  if (best && alpha > 0 && alpha >= 0.4 * (model.params.alpha_max ?? 1) && best[j] >= 0) {
    const src = model.destinations[best[j]];
    const r = ratingMap.get(src.id);
    if (r >= 4) return `Because you rated ${src.name} ${r}/5`;
  }
  if (prefs.has(d.type)) return `Matches your interest in ${d.type} destinations`;
  if (filters.month) return `In season in ${MONTHS[filters.month - 1]}`;
  return "Consistently well rated by other travellers";
}

/**
 * Mirrors recsys.hybrid.HybridRecommender.recommend.
 * ratings: array of [destinationId, rating] in the order they were given.
 * filters: {type, state, month} with null/undefined for "any".
 */
export function recommend(model, { ratings = [], preferences = [], k = 5, filters = {} } = {}) {
  const { destinations, popularity, params } = model;
  const seen = new Set();
  ratings = ratings.filter(([d]) => model.itemIndex.has(d) && !seen.has(d) && seen.add(d));
  const ratingMap = new Map(ratings);
  const n = ratings.length;
  const alphaMax = params.alpha_max ?? 1;
  const alpha = alphaMax * Math.min(n, params.cf_full_weight_at) / params.cf_full_weight_at;

  const profile = userProfile(model, preferences, ratings);
  const other = profile
    ? model.features.map((f, j) => params.content_share * ((dot(f, profile) + 1) / 2) + (1 - params.content_share) * popularity[j])
    : popularity.slice();

  let pred = null, best = null, score;
  if (n) {
    ({ pred, best } = model.cf.kind === "mf" ? mfPredict(model, ratings) : itemKnnPredict(model, ratings));
    score = other.map((o, j) => alpha * (pred[j] - 1) / 4 + (1 - alpha) * o);
  } else {
    score = other;
  }
  const strategy = n === 0 ? (profile ? "content" : "popular") : alpha >= 1 ? "cf" : "hybrid";

  const allows = (d) => (!filters.type || d.type === filters.type)
    && (!filters.state || d.state === filters.state)
    && (!filters.month || d.best_months.includes(filters.month))
    && !ratingMap.has(d.id);

  const order = destinations.map((d, j) => j).filter((j) => allows(destinations[j]))
    .sort((a, b) => score[b] - score[a]).slice(0, k);
  const prefs = new Set(preferences);
  const items = order.map((j) => ({
    destination: destinations[j],
    score: Math.min(1, Math.max(0, score[j])),
    predicted_rating: pred ? Math.round(pred[j] * 100) / 100 : null,
    reason: reason(model, destinations[j], j, ratingMap, prefs, filters, alpha, best),
    source: strategy,
  }));
  return { strategy, alpha, items };
}

/** Turn the exported JSON into a model object with lookup tables. */
export function loadModel(json) {
  const m = json.model;
  return {
    ...m,
    types: json.meta.types,
    itemIndex: new Map(m.destinations.map((d, j) => [d.id, j])),
    meta: json.meta,
    metrics: json.metrics || null,
    personas: json.personas,
  };
}

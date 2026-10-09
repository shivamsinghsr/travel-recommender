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
  if (best && alpha >= 0.4 && best[j] >= 0) {
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
  const alpha = Math.min(n, params.cf_full_weight_at) / params.cf_full_weight_at;

  const profile = userProfile(model, preferences, ratings);
  const other = profile
    ? model.features.map((f, j) => params.content_share * ((dot(f, profile) + 1) / 2) + (1 - params.content_share) * popularity[j])
    : popularity.slice();

  let pred = null, best = null, score;
  if (n) {
    ({ pred, best } = itemKnnPredict(model, ratings));
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
    personas: json.personas,
  };
}

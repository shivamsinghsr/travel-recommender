// Runs the browser recommender on test cases written by tests/test_parity.py.
// Usage: node tests/js/parity_cli.mjs model.json cases.json out.json
import { readFileSync, writeFileSync } from "node:fs";
import { loadModel, recommend } from "../../web/js/recommender.js";

const [modelPath, casesPath, outPath] = process.argv.slice(2);
const model = loadModel(JSON.parse(readFileSync(modelPath, "utf8")));
const cases = JSON.parse(readFileSync(casesPath, "utf8"));

const out = cases.map((c) => {
  const res = recommend(model, { ratings: c.ratings, preferences: c.preferences, k: c.k, filters: c.filters });
  return {
    strategy: res.strategy,
    alpha: res.alpha,
    items: res.items.map((it) => ({
      destination_id: it.destination.id, score: it.score, predicted_rating: it.predicted_rating, reason: it.reason,
    })),
  };
});
writeFileSync(outPath, JSON.stringify(out));

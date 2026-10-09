// Two interchangeable backends with the same methods:
//   DemoBackend: runs the recommender in the browser (GitHub Pages).
//   ApiBackend:  talks to the FastAPI service (docker compose / any Python host).
import { loadModel, recommend } from "./recommender.js";

const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage blocked: keep in memory */ }
  },
  remove(key) {
    try { localStorage.removeItem(key); } catch { /* ignore */ }
  },
};

export class DemoBackend {
  kind = "demo";

  async init(modelUrl) {
    const res = await fetch(modelUrl, { cache: "no-cache" });
    if (!res.ok) throw new Error(`Couldn't load the model (${res.status}). Reload the page to try again.`);
    this.model = loadModel(await res.json());
    this.you = store.get("wn.you", { preferences: [], ratings: [] });
    this.personas = new Map(this.model.personas.map((p) => [String(p.id), {
      ...p, ratings: p.ratings.map((r) => [r.destination_id, r.rating]),
    }]));
    this.active = store.get("wn.active", "you");
    if (this.active !== "you" && !this.personas.has(this.active)) this.active = "you";
    return {
      destinations: this.model.destinations, types: this.model.types, meta: this.model.meta,
      metrics: this.model.metrics,
    };
  }

  choices() {
    return [
      { id: "you", label: "You", detail: "Starts empty. Saved on this device." },
      ...[...this.personas.values()].map((p) => ({
        id: String(p.id), label: p.name,
        detail: `Sample traveller: likes ${p.preferences.join(" and ").toLowerCase()}, rated ${p.ratings.length} places`,
      })),
    ];
  }

  get profile() { return this.active === "you" ? this.you : this.personas.get(this.active); }

  _save() { if (this.active === "you") store.set("wn.you", this.you); }

  async select(id) { this.active = id; store.set("wn.active", id); }
  async current() { return { name: this.active === "you" ? "You" : this.profile.name, ...this.profile }; }

  async setPreferences(prefs) { this.profile.preferences = prefs; this._save(); }

  async rate(id, rating) {
    const p = this.profile;
    const i = p.ratings.findIndex(([d]) => d === id);
    if (i >= 0) p.ratings[i] = [id, rating]; else p.ratings.push([id, rating]);
    this._save();
  }

  async unrate(id) { this.profile.ratings = this.profile.ratings.filter(([d]) => d !== id); this._save(); }
  async clearRatings() { this.profile.ratings = []; this._save(); }

  async recommend({ k, filters }) {
    const p = this.profile;
    return recommend(this.model, { ratings: p.ratings, preferences: p.preferences, k, filters });
  }
}

export class ApiBackend {
  kind = "api";

  constructor(base = "") { this.base = base; }

  async _req(path, { method = "GET", body } = {}) {
    const headers = {};
    if (body) headers["Content-Type"] = "application/json";
    if (this.token) headers.Authorization = `Bearer ${this.token}`;
    const res = await fetch(this.base + path, { method, headers, body: body ? JSON.stringify(body) : undefined });
    if (res.status === 204) return null;
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      if (res.status === 401 && this.token) this.signOut(); // expired or revoked session
      const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
      const err = new Error(detail || `Request failed (${res.status})`);
      err.status = res.status;
      throw err;
    }
    return data;
  }

  async init() {
    const [destinations, model] = await Promise.all([this._req("/api/destinations"), this._req("/api/model")]);
    const types = [...new Set(destinations.map((d) => d.type))];
    this.token = store.get("wn.token", null);
    this.user = null;
    if (this.token) {
      try { this.user = await this._req("/api/me"); } catch { this.signOut(); }
    }
    return {
      destinations, types,
      meta: {
        model_version: model.version,
        cf_model: model.config
          ? `hybrid ${model.config.cf.kind}, alpha_max ${model.config.alpha_max}, content ${model.config.content_share}`
          : "item-based model fitted from live ratings",
      },
      metrics: model.metrics,
    };
  }

  get signedIn() { return this.user != null; }

  _session(body) {
    this.token = body.access_token; this.user = body.user;
    store.set("wn.token", this.token);
  }

  async register({ name, email, password }) {
    this._session(await this._req("/api/auth/register", { method: "POST", body: { name, email, password, preferences: [] } }));
  }

  async login({ email, password }) {
    this._session(await this._req("/api/auth/login", { method: "POST", body: { email, password } }));
  }

  signOut() { this.token = null; this.user = null; store.remove("wn.token"); }

  async current() {
    const [user, ratings] = await Promise.all([
      this._req("/api/me"),
      this._req(`/api/users/${this.user.id}/ratings`),
    ]);
    return { ...user, ratings: ratings.map((r) => [r.destination_id, r.rating]) };
  }

  async setPreferences(preferences) {
    await this._req(`/api/users/${this.user.id}`, { method: "PATCH", body: { preferences } });
  }

  async rate(id, rating) {
    await this._req(`/api/users/${this.user.id}/ratings`, { method: "POST", body: { destination_id: id, rating } });
  }

  async unrate(id) { await this._req(`/api/users/${this.user.id}/ratings/${id}`, { method: "DELETE" }); }

  async clearRatings() {
    const { ratings } = await this.current();
    await Promise.all(ratings.map(([d]) => this.unrate(d)));
  }

  async recommend({ k, filters }) {
    const q = new URLSearchParams({ k });
    for (const [key, v] of Object.entries(filters)) if (v) q.set(key, v);
    return this._req(`/api/users/${this.user.id}/recommendations?${q}`);
  }
}

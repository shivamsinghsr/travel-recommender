# Travel Recommender

Personalised recommendations for 66 Indian destinations, served by a FastAPI JSON API.

> **Phase 1** of 3: fixes the original Flask prototype's bugs and exposes a JSON API.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m api.seed                 # creates travel.db (SQLite) from data/*.csv
uvicorn api.main:app --reload      # http://127.0.0.1:8000/docs
pytest
```

## What Phase 1 fixes

| Original prototype | Phase 1 |
|---|---|
| `user_similarity[user_id - 1]`: wrong user for IDs with gaps, crash for large IDs | explicit ID → row mapping; unknown users get a 404 |
| 5 real places repeated 200× | 66 real destinations with real states, types and seasons |
| Neighbours with zero similarity chosen arbitrarily | mean-centred cosine with overlap shrinkage, popularity fallback |
| Already-visited places recommended again | excluded |
| HTML-only Flask routes | JSON API with validation and OpenAPI docs |

## Data

`data/destinations.csv` lists real places. `data/users.csv` and `data/ratings.csv` are **simulated**
by `python -m recsys.simulate` (deterministic, seed 42): each simulated traveller has hidden tastes
that drive both what they visit and how they rate it.

## License

MIT

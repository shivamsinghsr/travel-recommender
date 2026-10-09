# Where next? A travel recommender for India

Personalised recommendations for 66 Indian destinations. New travellers get picks from the interests
they choose; as they rate places, a collaborative-filtering model takes over.

**Live demo:** https://shivamsinghsr.github.io/travel-recommender/ (runs entirely in your browser)

> **Phase 2** of 3: PostgreSQL, ratings write-back, a hybrid recommender and a web app.

## Run it

**Full stack with Docker** (PostgreSQL + API + web app):

```bash
docker compose up --build        # http://localhost:8000  (API docs at /docs)
```

**Without Docker** (SQLite):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m api.seed                # runs migrations, loads data/*.csv
uvicorn api.main:app --reload     # http://127.0.0.1:8000
pytest
```

## How recommendations work

```
alpha = min(number of ratings, 5) / 5
score = alpha × collaborative + (1 − alpha) × (0.75 × content + 0.25 × popularity)
```

- **Content-based**: each destination is a vector of its type, region and season; your profile is built
  from the interests you pick and the places you rate. Works with zero ratings.
- **Collaborative (item-based)**: "travellers who rated X highly also rated Y highly". Needs only your own
  ratings plus a precomputed item-similarity matrix, so it works from your first rating.
- **Filters first**: type, state and month are applied before scoring; rated places never come back.

The same algorithm runs in Python (API) and JavaScript (demo). `tests/test_parity.py` runs both on
84 cases and fails if their rankings differ.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/destinations?type=&state=&month=` | Browse the catalogue |
| POST | `/api/users` | Create a profile |
| PATCH | `/api/users/{id}` | Update name or interests |
| GET/POST | `/api/users/{id}/ratings` | List / add or update a rating |
| DELETE | `/api/users/{id}/ratings/{destination_id}` | Remove a rating |
| GET | `/api/users/{id}/recommendations?k=&type=&state=&month=` | Personalised picks |
| GET | `/api/health` | Database and model status |

## Data

`data/destinations.csv` lists real places. `data/users.csv` and `data/ratings.csv` are **simulated** by
`python -m recsys.simulate` (deterministic, seed 42): each simulated traveller has hidden tastes that drive
both what they visit and how they rate it.

## License

MIT

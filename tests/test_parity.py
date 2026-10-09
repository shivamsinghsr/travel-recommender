"""The browser (web/js/recommender.js) and Python must rank identically."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from recsys.catalog import TYPES
from recsys.export_web import build_hybrid, export
from recsys.hybrid import Filters

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


def _cases(hybrid, inter, n_random: int = 60) -> list[dict]:
    rng = np.random.default_rng(0)
    ids = [d.id for d in hybrid.destinations]
    states = sorted({d.state for d in hybrid.destinations})
    cases = [
        {"ratings": [], "preferences": [], "k": 5, "filters": {}},
        {"ratings": [], "preferences": ["Beach"], "k": 5, "filters": {}},
        {"ratings": [], "preferences": ["Beach", "Wildlife"], "k": 5, "filters": {"month": 12}},
        {"ratings": [[27, 5]], "preferences": ["Adventure"], "k": 8, "filters": {}},
    ]
    for uid in inter.user_ids[:20]:  # real histories, some longer than the k=20 neighbour cap
        cases.append({"ratings": [[d, r] for d, r in inter.user_ratings(int(uid)).items()],
                      "preferences": [], "k": 10, "filters": {}})
    for _ in range(n_random):
        n = int(rng.integers(0, 30))
        chosen = rng.choice(ids, size=n, replace=False)
        filters = {}
        if rng.random() < 0.4:
            filters["month"] = int(rng.integers(1, 13))
        if rng.random() < 0.3:
            filters["type"] = str(rng.choice(TYPES))
        if rng.random() < 0.15:
            filters["state"] = str(rng.choice(states))
        cases.append({
            "ratings": [[int(d), int(rng.integers(1, 6))] for d in chosen],
            "preferences": [str(t) for t in rng.choice(TYPES, size=int(rng.integers(0, 3)), replace=False)],
            "k": int(rng.integers(1, 12)),
            "filters": filters,
        })
    return cases


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_and_python_agree(tmp_path):
    hybrid, inter = build_hybrid()
    model_path = export(tmp_path, hybrid=hybrid)
    cases = _cases(hybrid, inter)
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    subprocess.run([NODE, str(ROOT / "tests/js/parity_cli.mjs"), str(model_path),
                    str(tmp_path / "cases.json"), str(tmp_path / "out.json")], check=True)
    js_results = json.loads((tmp_path / "out.json").read_text())

    for case, js in zip(cases, js_results, strict=True):
        py = hybrid.recommend(dict(case["ratings"]), case["preferences"], k=case["k"],
                              filters=Filters(**case["filters"]))
        assert js["strategy"] == py.strategy, case
        assert js["alpha"] == pytest.approx(py.alpha)
        assert [i["destination_id"] for i in js["items"]] == [r.destination_id for r in py.items], case
        for j, p in zip(js["items"], py.items, strict=True):
            assert j["score"] == pytest.approx(p.score, abs=1e-9)
            assert j["reason"] == p.reason
            if p.predicted_rating is None:
                assert j["predicted_rating"] is None
            else:
                assert j["predicted_rating"] == pytest.approx(p.predicted_rating, abs=0.011)

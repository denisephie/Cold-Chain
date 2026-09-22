# Cold-Chain

Predicts **silent cold-chain failures** — shipments that arrive looking fine but were
temperature-compromised in transit — using only information you have *before* the truck leaves.

The model deliberately uses four pre-shipment features, so it can be used to make a go/no-go
call rather than to explain a failure after the fact:

| Feature | Meaning |
| --- | --- |
| `transit_days` | Planned door-to-door transit time |
| `fill_ratio` | Product volume / container capacity |
| `leg_count` | Number of transport legs / hand-offs |
| `package_type` | Packaging class (0, 1, 2) |

There are two interfaces: a Python CLI (`predict_risk.py`) and a static web app (`index.html`)
that runs the same model in the browser with no backend.

---

## Quickstart

The trained models are pickles committed to `artifacts/`, and they must be loaded under the
library versions they were built with. Use a virtual environment on **Python 3.14**:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Score a shipment:

```bash
python predict_risk.py
```

It prompts for the four features, then prints a risk band, what-if scenarios (shorter transit,
fewer legs, different packaging), delay sensitivity, and an estimated door-opens range.

For the web app, open `index.html` directly, or serve the repo root to match how it deploys:

```bash
python -m http.server 8000   # then visit http://localhost:8000
```

---

## Environment

> **This is the most common way to get wrong answers from this repo.**

`artifacts/*.joblib` are pickled scikit-learn pipelines. Loading a pickle under a different
scikit-learn version can fail outright or — the dangerous case — score differently without
telling you. The versions each model was trained with are recorded in `artifacts/model_meta.json`
and `artifacts/door_meta.json`:

```
python 3.14.5   scikit-learn 1.9.0   xgboost 3.4.1   pandas 3.0.3
```

`predict_risk.py` checks this at startup and prints a warning naming any mismatch. If you see it,
your scores may not match the metrics below. Rebuild your environment, or retrain with `Model.py`
so the artifacts match your machine.

Note that `requirements.txt` is currently **unpinned**, so a fresh `pip install` will pull whatever
is latest that day and may not reproduce the versions above.

---

## Pipeline

```
shipment-sensor-dataset.csv
        │
        ├── EDA.py            exploratory plots (standalone, not part of the build)
        │
        ├── Model.py          trains 3 models  ->  artifacts/
        │
        └── export_web.py     artifacts/ + dataset  ->  data.js (browser bundle)
                                      │
                                index.html
```

### `Model.py`

Trains three models and writes them to `artifacts/` along with metadata and diagnostic plots.
Hyperparameters are tuned with `RandomizedSearchCV`; the decision threshold is chosen from a
cost ratio rather than a default 0.5.

1. **Risk model** — four pre-shipment features → failure probability.
2. **Door-opens model** — same features → expected door opens, with an 80% interval from a
   negative binomial (dispersion in `door_meta.json`).
3. **Risk given opens** — features + `door_opens` → failure probability. **What-if only**, since
   the real number of opens is not knowable before shipping.

### `export_web.py`

Compiles the XGBoost boosters into compact JSON trees so the browser can score without a backend.

### `advisory.py`

Separate context tool — weather along the route (Open-Meteo), GDACS hazard events, and FoodKeeper
storage guidance. **It does not feed the risk model**; it is background for a planned shipment.

```bash
python advisory.py "Rotterdam" "Milan" 2026-10-01 5 "chicken" chilled passive
```

Requires network access. Storage bands and product overrides in this file are hand-entered and
flagged in comments as needing verification — treat them as a starting point, not authority.

---

## Model performance

Held-out test set (20% of 8,000 shipments), from `artifacts/model_meta.json`:

| Metric | Value |
| --- | --- |
| ROC AUC | 0.845 |
| PR AUC | 0.569 |
| Brier score | 0.116 |
| Log loss | 0.363 |
| Base failure rate | 19.7% |

Door-opens model: R² 0.423, MAE 3.48 opens. Risk-given-opens AUC 0.873.

### Thresholds

| Band | Probability | Action |
| --- | --- | --- |
| LOW | < 0.23 | Standard handling |
| MEDIUM | 0.23 – 0.50 | Review before shipping |
| HIGH | ≥ 0.50 | Do not ship as planned |

The review threshold of 0.23 is not arbitrary: it comes from assuming a **missed failure costs
3.5× a false alarm** (`COST_FN` in `Model.py`). Lower that ratio and fewer shipments get flagged.

`COST_FN` does not affect training — it only selects the threshold, so it slides along a fixed
precision/recall curve and can never improve both at once. At 0.23 the model catches **78% of
failures at 42% precision**, flagging 36% of shipments; at the margin each additional failure
caught costs roughly 4–5 extra shipments flagged.

A sweep of the cost ratio on the held-out test set (2026-09-22) found F1 peaks at a ratio of 2.5
(0.560 vs 0.548), but bootstrapping put that gain at +0.012 with a 95% CI of [−0.009, +0.032] —
not a real improvement, so 3.5 was kept for its higher recall. Adding other booking-time features
(`carrier_id`, `origin_zone`, `dest_zone`) moved PR-AUC by +0.005, i.e. noise.

The upshot: this ratio should be set from **your own cost of a spoiled shipment versus an
unnecessary intervention**, not tuned for F1. See the comment above `COST_FN` in `Model.py`.

### Interpreting the output

PR AUC of 0.57 against a 19.7% base rate is a genuine signal, not a coin flip — but this is a
triage aid, not a guarantee. Two cautions:

- **Door opens are correlational.** They travel with inspections and delays in the logger data.
  The what-if numbers show association, not proof that opening a door causes a failure.
- **Out-of-range inputs are flagged, not blocked.** If a value falls outside the training
  ranges in `model_meta.json`, the model still scores it but labels the result an
  extrapolation. An unseen `package_type` is ignored by the model entirely. Read the notes
  printed under the score.

---

## Data

`shipment-sensor-dataset.csv` — 8,000 shipments, 24 columns, target `silent_failure`.

The file carries far more than the model uses (temperature, humidity, vibration, carrier and
zone IDs, sensor gaps). Those are outcome or in-transit signals and are excluded on purpose:
they are not available at the moment the go/no-go decision is made. `EDA.py` explores the
full set.

---

## Deployment

Static hosting on Vercel — `vercel.json` sets `cleanUrls` and an `X-Content-Type-Options` header.
There is no server-side component; `index.html` plus `data.js` are the whole deployed app.

---

## Known issues

- **`export_web.py` writes to the wrong path.** It writes `./web/data.js`, but `index.html` loads
  `data.js` from the repo root and no `web/` directory exists. Regenerating the bundle therefore
  leaves the live site on the old `data.js`. Until this is fixed, copy the file manually:
  `python export_web.py && cp web/data.js data.js`
- **`requirements.txt` is unpinned** — see [Environment](#environment).

## Layout

```
Model.py          training
predict_risk.py   CLI scorer
export_web.py     browser bundle builder
EDA.py            exploratory analysis
advisory.py       weather / hazard / storage context (standalone)
index.html        web app
data.js           exported models + dataset for the browser
artifacts/        trained models, metadata, plots
```

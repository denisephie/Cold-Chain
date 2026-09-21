"""
export_web.py - run AFTER Model.py. Converts ./artifacts (trained models) and the dataset into
web/data.js so the website can predict in the browser with no backend.

    python Model.py
    python export_web.py      # writes web/data.js
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

ART, OUT, CSV = "./artifacts", "./web", "./shipment-sensor-dataset.csv"
TARGET, OPENS = "silent_failure", "door_opens"
os.makedirs(OUT, exist_ok=True)

meta = json.load(open(f"{ART}/model_meta.json"))
dmeta = json.load(open(f"{ART}/door_meta.json"))
risk = joblib.load(f"{ART}/risk_pipeline.joblib")
door = joblib.load(f"{ART}/door_opens_model.joblib")
fail = joblib.load(f"{ART}/risk_given_opens.joblib")


# ---------- XGBoost -> compact JSON trees ----------
def nest(node, names):
    if "leaf" in node:
        return round(node["leaf"], 6)
    kids = {c["nodeid"]: c for c in node["children"]}
    return [names.index(node["split"]), round(node["split_condition"], 6),
            nest(kids[node["yes"]], names), nest(kids[node["no"]], names)]


def export_trees(pipe, link):
    booster = pipe.named_steps["model"].get_booster()
    names = list(booster.feature_names)
    raw = json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]
    bs = float(str(raw).strip("[]"))
    base = float(np.log(bs / (1 - bs)) if link == "logit" else np.log(bs))
    return {"n": names, "b": round(base, 6),
            "t": [nest(json.loads(t), names) for t in booster.get_dump(dump_format="json")]}, booster


def walk(tree, x):
    while isinstance(tree, list):
        tree = tree[2] if x[tree[0]] < tree[1] else tree[3]
    return tree


def run(m, x, link):
    z = m["b"] + sum(walk(t, x) for t in m["t"])
    return 1 / (1 + np.exp(-z)) if link == "logit" else float(np.exp(z))


risk_js, risk_booster = export_trees(risk, "logit")
door_js, _ = export_trees(door, "exp")
fail_js, _ = export_trees(fail, "logit")

# ---------- data + the same test split as Model.py ----------
df = pd.read_csv(CSV)
feats = meta["features"]
cols = list(dict.fromkeys(feats + dmeta["door_features"] + [OPENS, TARGET]))
data = df[cols].dropna(subset=[TARGET]).copy()
data[TARGET] = data[TARGET].astype(int)
train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data[TARGET])
y = test[TARGET].values
prob = risk.predict_proba(test[feats])[:, 1]
assert abs(roc_auc_score(y, prob) - meta["metrics_test"]["roc_auc"]) < 1e-9, "split differs from Model.py"


def vec(names, row):
    out = []
    for n in names:
        out.append(float(n.split("_")[-1] == str(int(row["package_type"]))) if n.startswith("package_type_")
                   else float(row[n]))
    return out


# check the browser-side maths against sklearn/xgboost
for _, r in test.head(50).iterrows():
    r = r.to_dict()
    assert abs(run(risk_js, vec(risk_js["n"], r), "logit")
               - risk.predict_proba(pd.DataFrame([r])[feats])[0, 1]) < 1e-4
    assert abs(run(door_js, vec(door_js["n"], r), "exp")
               - door.predict(pd.DataFrame([r])[dmeta["door_features"]])[0]) < 1e-3
    assert abs(run(fail_js, vec(fail_js["n"], r), "logit")
               - fail.predict_proba(pd.DataFrame([r])[dmeta["risk_features"]])[0, 1]) < 1e-4
print("Exported trees match the Python models.")


def thin(a, b, n=90):
    i = np.unique(np.linspace(0, len(a) - 1, n).astype(int))
    return [[round(float(a[j]), 4), round(float(b[j]), 4)] for j in i]


fpr, tpr, _ = roc_curve(y, prob)
prec, rec, _ = precision_recall_curve(y, prob)
o = np.argsort(rec)
cal_true, cal_pred = calibration_curve(y, prob, n_bins=10, strategy="quantile")
thr = meta["thresholds"]["review"]
tn, fp, fn, tp = confusion_matrix(y, (prob >= thr).astype(int)).ravel()
table = []
for t in (0.15, 0.20, thr, 0.30, 0.40, 0.50):
    f = prob >= t
    table.append([round(float(t), 2), round(float(f.mean()), 3),
                  round(float((f & (y == 1)).sum() / max(f.sum(), 1)), 3),
                  round(float((f & (y == 1)).sum() / (y == 1).sum()), 3)])

gain = risk_booster.get_score(importance_type="gain")
imp = {}
for k, v in gain.items():
    k = "package_type" if k.startswith("package_type") else k
    imp[k] = imp.get(k, 0) + v
tot = sum(imp.values())
imp = {k: round(v / tot, 4) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}

# ---------- dataset overview ----------
num = [c for c in df.columns if c not in ("shipment_id", TARGET)]
hist = {}
for c in ["transit_days", "door_opens", "temp_mean_c", "temp_max_c", "rh_mean",
          "vibration_index", "temp_std_c", "temp_recovery_rate"]:
    s = df[c].dropna()
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    edges = np.linspace(lo, hi, 21)
    d = {"edges": [round(float(e), 3) for e in edges]}
    for name, g in (("ok", df[df[TARGET] == 0][c]), ("fail", df[df[TARGET] == 1][c])):
        h, _ = np.histogram(g.dropna().clip(lo, hi), bins=edges)
        d[name] = [round(float(v), 4) for v in h / h.sum()]
    hist[c] = d


def rate(col, bins=None):
    g = df.groupby(pd.cut(df[col], bins) if bins else df[col], observed=True)[TARGET].agg(["mean", "size"])
    return [[str(k), round(float(r["mean"]), 4), int(r["size"])] for k, r in g.iterrows()]


groups = {"leg_count": rate("leg_count"), "package_type": rate("package_type"),
          "transit_days": rate("transit_days", [0, 2, 4, 6, 8, 20]),
          "carrier_id": rate("carrier_id"), "origin_zone": rate("origin_zone"), "dest_zone": rate("dest_zone")}
corr = df[num].corrwith(df[TARGET]).dropna()
corr = [[k, round(float(v), 3)] for k, v in corr.reindex(corr.abs().sort_values(ascending=False).index).items()]
colstats = [[c, round(float(df[c].mean()), 3), round(float(df[c].min()), 3), round(float(df[c].max()), 3),
             int(df[c].isna().sum())] for c in num]
sample = df.sample(200, random_state=1)
scols = ["shipment_id", "transit_days", "door_opens", "temp_mean_c", "temp_max_c", "temp_min_c", "rh_mean",
         "package_type", "fill_ratio", "leg_count", TARGET]

CC = {
    "meta": {k: meta[k] for k in ("created", "features", "categories", "training_ranges", "thresholds",
                                  "cost_ratio_fn_to_fp", "metrics_test", "best_params", "base_failure_rate",
                                  "versions")},
    "door": {k: dmeta[k] for k in ("nb_dispersion_k", "interval", "opens_training_range", "metrics_test")},
    "models": {"risk": risk_js, "door": door_js, "fail": fail_js},
    "test": {"n": int(len(y)), "pos": int(y.sum()), "cm": [int(tn), int(fp), int(fn), int(tp)],
             "roc": thin(fpr, tpr), "pr": thin(rec[o], prec[o]), "cal": thin(cal_pred, cal_true, 10),
             "table": table, "imp": imp, "train_n": int(len(train))},
    "data": {"rows": int(len(df)), "cols": int(df.shape[1]), "rate": round(float(df[TARGET].mean()), 4),
             "mean": {c: round(float(df[c].mean()), 3) for c in num}, "hist": hist, "groups": groups,
             "corr": corr, "colstats": colstats,
             "sample": {"cols": scols, "rows": [[(round(v, 3) if isinstance(v, float) else v) for v in r]
                                                for r in sample[scols].itertuples(index=False)]}},
}
with open(f"{OUT}/data.js", "w") as f:
    f.write("window.CC=" + json.dumps(CC, separators=(",", ":"), default=lambda o: o.item()) + ";")
print(f"Wrote {OUT}/data.js ({os.path.getsize(f'{OUT}/data.js') / 1024:.0f} KB)")

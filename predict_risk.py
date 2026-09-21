"""
predict_risk.py - score shipments before they leave.
"""

import json
import os
import re
import sys

import joblib
import pandas as pd

ART_DIR = "./artifacts"
PACKAGE_LABELS = {0: "Package type 0", 1: "Package type 1", 2: "Package type 2"}

ADVICE = {
    "LOW": "Standard handling.",
    "MEDIUM": "Review before shipping: shorten the route or transit time, reduce hand-offs, "
              "upgrade the packaging, and add a data logger.",
    "HIGH": "Do not ship as planned. Reduce transit time / legs or upgrade packaging first, "
            "and put a data logger with alarms on the shipment.",
}

# Load models
model = joblib.load(os.path.join(ART_DIR, "risk_pipeline.joblib"))
with open(os.path.join(ART_DIR, "model_meta.json")) as f:
    meta = json.load(f)

FEATURES = meta["features"]
NUMERIC = meta["numeric_features"]
CATEGORICAL = meta["categorical_features"]
CATEGORIES = meta["categories"]
RANGES = meta["training_ranges"]
REVIEW_T = meta["thresholds"]["review"]
HIGH_T = meta["thresholds"]["high"]

# The door-opens models are optional
try:
    from scipy.stats import nbinom
    door_model = joblib.load(os.path.join(ART_DIR, "door_opens_model.joblib"))
    risk_opens_model = joblib.load(os.path.join(ART_DIR, "risk_given_opens.joblib"))
    with open(os.path.join(ART_DIR, "door_meta.json")) as f:
        door_meta = json.load(f)
    HAS_DOOR = True
except Exception:
    HAS_DOOR = False


# Scoring
def band(p):
    if p >= HIGH_T:
        return "HIGH"
    if p >= REVIEW_T:
        return "MEDIUM"
    return "LOW"


def coerce(df):
    df = df.copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="raise")
    for col in CATEGORICAL:
        df[col] = df[col].astype(type(CATEGORIES[col][0]))
    return df[FEATURES]


def validate(row):
    # raises ValueError for impossible values, returns notes for unusual ones
    problems, notes = [], []
    if row["transit_days"] <= 0:
        problems.append("transit_days must be > 0")
    if not 0 < row["fill_ratio"] <= 1:
        problems.append("fill_ratio must be between 0 and 1")
    if row["leg_count"] < 1 or row["leg_count"] != int(row["leg_count"]):
        problems.append("leg_count must be a whole number >= 1")
    for col in CATEGORICAL:
        if row[col] not in CATEGORIES[col]:
            notes.append(f"{col}={row[col]} was never seen in training (known: {CATEGORIES[col]}); "
                         "the model ignores it, so treat the result as less reliable.")
    if problems:
        raise ValueError("; ".join(problems))
    for col in NUMERIC:
        lo, hi = RANGES[col]
        if not lo <= row[col] <= hi:
            notes.append(f"{col}={row[col]:g} is outside the training range ({lo:g} to {hi:g}); "
                         "the prediction is an extrapolation.")
    return notes


def predict_risk(row):
    frame = coerce(pd.DataFrame([row]))
    notes = validate({c: frame[c].iloc[0] for c in FEATURES})
    p = float(model.predict_proba(frame)[0, 1])
    return {"risk": p, "band": band(p), "advice": ADVICE[band(p)], "notes": notes}


def what_if(row):
    base = predict_risk(row)["risk"]
    scenarios = []
    for days in (1, 2):
        if row["transit_days"] - days >= RANGES["transit_days"][0]:
            scenarios.append((f"Transit {days} day(s) shorter",
                              {**row, "transit_days": row["transit_days"] - days}))
    if row["leg_count"] > 1:
        scenarios.append(("One fewer leg / hand-off", {**row, "leg_count": row["leg_count"] - 1}))
    for pkg in CATEGORIES["package_type"]:
        if pkg != row["package_type"]:
            scenarios.append((f"Switch to {PACKAGE_LABELS.get(pkg, pkg)}", {**row, "package_type": pkg}))
    results = []
    for label, r in scenarios:
        p = predict_risk(r)["risk"]
        results.append((label, p, p - base))
    return results


def delay_sensitivity(row):
    base = predict_risk(row)["risk"]
    results = []
    for extra in (1, 2):
        p = predict_risk({**row, "transit_days": row["transit_days"] + extra})["risk"]
        results.append((f"Trip runs {extra} day(s) late", p, p - base))
    return results


def score_csv(path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing required columns: {missing}")
    prob = model.predict_proba(coerce(df[FEATURES]))[:, 1]
    df["risk_probability"] = prob.round(4)
    df["risk_band"] = [band(p) for p in prob]
    out = os.path.splitext(path)[0] + "_scored.csv"
    df.to_csv(out, index=False)
    print(f"Scored {len(df)} shipments -> {out}")
    print(df["risk_band"].value_counts().to_string())


# Door opens
def risk_at_opens(row, n_opens):
    frame = coerce(pd.DataFrame([row]))[door_meta["door_features"]].copy()
    frame["door_opens"] = n_opens
    return float(risk_opens_model.predict_proba(frame[door_meta["risk_features"]])[0, 1])


def door_forecast(row):
    frame = coerce(pd.DataFrame([row]))[door_meta["door_features"]]
    mu = float(door_model.predict(frame)[0])
    k = door_meta["nb_dispersion_k"]
    q_lo, q_hi = door_meta["interval"]
    low = int(nbinom.ppf(q_lo, k, k / (k + mu)))
    high = int(nbinom.ppf(q_hi, k, k / (k + mu)))
    typical = int(round(mu))
    return {"expected": mu, "low": low, "high": high,
            "scenarios": [("Light handling", low), ("Typical handling", typical), ("Heavy handling", high)],
            "risks": [risk_at_opens(row, n) for n in (low, typical, high)]}


def door_section(row):
    if not HAS_DOOR:
        return
    try:
        base = predict_risk(row)["risk"]
        f = door_forecast(row)
    except Exception:
        return
    pct = round((door_meta["interval"][1] - door_meta["interval"][0]) * 100)
    print("\nHandling (door opens)")
    print(f"  Expected about {f['expected']:.0f} opens ({pct}% range {f['low']} to {f['high']})")
    for (label, n), p in zip(f["scenarios"], f["risks"]):
        print(f"  {label + f' ({n} opens)':<32} {p:>4.0%}")
    print("  Door opens travel with inspections and delays in the logger data; this shows association, "
          "not proof that opening a door causes failure.")
    raw = input("Known or planned door opens (customs check, etc.)? Number, or Enter to skip: ").strip()
    if raw.isdigit():
        n = int(raw)
        p = risk_at_opens(row, n)
        print(f"  With {n} opens: {p:.0%}   ({p - base:+.0%} vs the headline {base:.0%})")
        most_seen = door_meta["opens_training_range"][1]
        if n > most_seen:
            print(f"  ! More opens than any training shipment ({most_seen}); treat as an extrapolation.")


# Prompts
def ask_number(prompt, cast=float, lo=None, hi=None):
    while True:
        try:
            v = cast(input(prompt).strip())
        except ValueError:
            print("  Please enter a number.")
            continue
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            print(f"  Value must be between {lo} and {hi}.")
            continue
        return v


def ask_choice(prompt, options):
    codes = list(options)
    print(prompt)
    for i, code in enumerate(codes, 1):
        print(f"  {i}. {options[code]}")
    while True:
        try:
            return codes[int(input("  Choose number: ").strip()) - 1]
        except (ValueError, IndexError):
            print("  Invalid choice.")


def ask_fill_ratio():
    raw = input("Fill ratio 0-1 (product volume / container capacity), or Enter to calculate it: ").strip()
    if raw:
        try:
            v = float(raw)
            if 0 < v <= 1:
                return v
        except ValueError:
            pass
        print("  Invalid, let's calculate it instead.")
    volume = ask_number("  Product volume (L): ", lo=0.001)
    capacity = ask_number("  Container capacity (L): ", lo=0.001)
    ratio = min(volume / capacity, 1.0)
    print(f"  fill_ratio = {ratio:.3f}   (assumes fill_ratio = volume / capacity; confirm this matches your dataset)")
    return ratio


def get_input():
    print("\n================================")
    print("PRE-SHIPMENT RISK CHECK")
    print("================================")
    row = {}
    row["transit_days"] = ask_number("Planned transit time (days, door to door): ", lo=0.01)
    row["leg_count"] = ask_number("Number of transport legs / hand-offs (1, 2, 3...): ", cast=int, lo=1)
    labels = {c: PACKAGE_LABELS.get(c, str(c)) for c in CATEGORIES["package_type"]}
    row["package_type"] = ask_choice("Package type:", labels)
    row["fill_ratio"] = ask_fill_ratio()
    return row


def show(row):
    try:
        res = predict_risk(row)
    except ValueError as e:
        print(f"\nInvalid input: {e}")
        return
    print("\n--------------------------------")
    print(f"Failure risk: {res['risk']:.0%}   ->   {res['band']}")
    print(f"Base rate in training data: {meta['base_failure_rate']:.0%}")
    print(res["advice"])
    for n in res["notes"]:
        print(f"  ! {n}")
    options = what_if(row)
    if options:
        print("\nWhat if...")
        for label, p, delta in options:
            print(f"  {label:<32} {p:>4.0%}  ({delta:+.0%})")
    print("\nIf it slips...")
    for label, p, delta in delay_sensitivity(row):
        print(f"  {label:<32} {p:>4.0%}  ({delta:+.0%})")
    print("--------------------------------")


# Trip briefing (advisory.py)
RANGE_INPUT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)\s*$")


def ask_product(advisory):
    raw = input("Product (e.g. apples, milk, pharma, or a range like 2-8): ").strip()
    m = RANGE_INPUT.match(raw)
    if m:
        return advisory.resolve_product((float(m.group(1)), float(m.group(2))))
    if advisory.find_override(raw):
        return advisory.resolve_product(raw)
    if not advisory.foods_available():
        print("  foodkeeper_lookup.csv not found (run build_foodkeeper.py). Enter a range instead, e.g. 2-8.")
        return ask_product(advisory)
    matches = advisory.search_foods(raw)
    if not matches:
        print("  Not found. Try another name, or enter a range like 2-8.")
        return ask_product(advisory)
    food = matches[0]
    if len(matches) > 1:
        for i, r in enumerate(matches, 1):
            print(f"  {i}. {advisory.food_label(r)}")
        pick = input("  Which one? (Enter = 1): ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(matches):
            food = matches[int(pick) - 1]
    storage = input("  Ship it chilled / room / frozen? (Enter = automatic): ").strip().lower() or None
    return advisory.resolve_product(raw, storage=storage, food=food)


def trip_briefing(row):
    if input("\nAdd trip briefing (product, season, weather, alerts)? (y/n): ").strip().lower() != "y":
        return
    try:
        import advisory
        from datetime import date
        origin = input("Origin city: ").strip()
        dest = input("Destination city: ").strip()
        ship = date.fromisoformat(input("Ship date (YYYY-MM-DD): ").strip())
        product = ask_product(advisory)
        packaging = input("Packaging system: active / passive / hybrid (Enter to skip): ").strip().lower() or None
        advisory.print_briefing(advisory.briefing(origin, dest, ship, row["transit_days"], product,
                                                  packaging=packaging))
    except Exception as e:  # never let the briefing break the risk check
        print(f"  Briefing unavailable: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        score_csv(sys.argv[1])
    else:
        while True:
            row = get_input()
            show(row)
            door_section(row)
            trip_briefing(row)
            if input("\nAssess another shipment? (y/n): ").strip().lower() != "y":
                break

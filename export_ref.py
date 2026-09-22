"""Builds refdata.js: the static reference lists the web app needs.

Two independent things live here, and they stay independent on purpose:

  countries  country name + ISO codes + a lat/lon to query Open-Meteo with.
             Used by the Overview page (live weather) and by the shipment
             form's Country From / Country To pickers.
  food       the FoodKeeper storage-life table, grouped by category. Used by
             the shipment form's Food Type picker and the shelf-life check.

Nothing in here touches shipment-sensor-dataset.csv; the Overview page must
not carry any training-data figures.

    python export_ref.py
"""

import csv
import json
import pathlib
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).parent
FOODKEEPER = HERE / "foodkeeper_lookup.csv"
OUT = HERE / "refdata.js"
GEO_CACHE = HERE / "capital_coords.json"

# mledoze/countries, public domain. Fetched at build time only; the generated
# refdata.js is committed so the app never needs the network for this.
COUNTRIES_URL = (
    "https://raw.githubusercontent.com"
    "/mledoze/countries/master/dist/countries.json"
)
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


def get_json(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def geocode_capital(capital, iso2, cache):
    """Resolve a capital to lat/lon via Open-Meteo geocoding.

    The country centroid that ships with the country list is a bad weather
    proxy - Indonesia's lands in the Flores Sea - so the capital is looked up
    instead. Results are cached on disk; a miss falls back to the centroid.
    """
    key = f"{iso2}:{capital}"
    if key in cache:
        return cache[key]
    if not capital:
        cache[key] = None
        return None

    q = urllib.parse.urlencode(
        {"name": capital, "count": 10, "language": "en", "format": "json"})
    try:
        res = get_json(f"{GEOCODE_URL}?{q}", timeout=30).get("results") or []
    except Exception as exc:  # network hiccup -> fall back to the centroid
        print(f"  geocode failed for {capital} ({iso2}): {exc}")
        return None

    hits = [r for r in res if r.get("country_code") == iso2]
    if not hits:
        cache[key] = None
        return None
    # PPLC is the "capital of a political entity" feature code.
    hits.sort(key=lambda r: (r.get("feature_code") != "PPLC",
                             -(r.get("population") or 0)))
    best = hits[0]
    cache[key] = [round(best["latitude"], 4), round(best["longitude"], 4)]
    time.sleep(0.12)  # be polite to a free API
    return cache[key]


def load_countries():
    raw = get_json(COUNTRIES_URL)
    cache = json.loads(GEO_CACHE.read_text()) if GEO_CACHE.exists() else {}

    out = []
    for c in raw:
        if not c.get("independent") and c.get("cca3") not in ("HKG", "MAC", "PSE", "TWN"):
            continue
        centroid = c.get("latlng")
        if not centroid or len(centroid) != 2:
            continue
        cap = (c.get("capital") or [""])[0]
        ll = geocode_capital(cap, c["cca2"], cache) or [
            round(centroid[0], 4), round(centroid[1], 4)]
        out.append([
            c["name"]["common"],
            c["cca2"],
            c["cca3"],
            ll[0],
            ll[1],
            cap,
            c.get("region", ""),
        ])

    GEO_CACHE.write_text(json.dumps(cache, indent=0))
    out.sort(key=lambda r: r[0])
    return out


def num(v):
    v = (v or "").strip()
    if not v:
        return None
    try:
        return round(float(v), 1)
    except ValueError:
        return None


def load_food():
    with FOODKEEPER.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cats = sorted({r["category"].strip() for r in rows if r["category"].strip()})
    cat_ix = {c: i for i, c in enumerate(cats)}

    items = []
    for r in rows:
        cat = r["category"].strip()
        if cat not in cat_ix:
            continue
        name = r["name"].strip()
        sub = (r["subtitle"] or "").strip()
        label = f"{name} ({sub})" if sub else name
        items.append([
            label,
            cat_ix[cat],
            num(r["pantry_min_days"]), num(r["pantry_max_days"]),
            num(r["fridge_min_days"]), num(r["fridge_max_days"]),
            num(r["freezer_min_days"]), num(r["freezer_max_days"]),
            r["keywords"].strip().lower(),
        ])
    items.sort(key=lambda r: (r[1], r[0]))
    return cats, items


def main():
    countries = load_countries()
    cats, items = load_food()

    payload = {
        "source": {"countries": COUNTRIES_URL, "food": FOODKEEPER.name},
        # [name, iso2, iso3, lat, lon, capital, region]
        "countries": countries,
        "foodCategories": cats,
        # [label, categoryIndex, pantryMin, pantryMax, fridgeMin, fridgeMax,
        #  freezerMin, freezerMax, keywords]
        "food": items,
    }

    OUT.write_text(
        "/* Generated by export_ref.py - do not edit by hand. */\n"
        "window.REF=" + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    kb = OUT.stat().st_size / 1024
    print(f"{OUT.name}: {len(countries)} countries, {len(items)} food items, "
          f"{len(cats)} categories, {kb:.0f} KB")


if __name__ == "__main__":
    main()

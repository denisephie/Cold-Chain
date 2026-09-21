"""advisory.py - context for a planned shipment. It does not feed the risk model."""

import csv
import json
import math
import os
import statistics
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GDACS_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"     
FOODKEEPER_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foodkeeper_lookup.csv")

TIMEOUT_S = 10
FORECAST_HORIZON_DAYS = 15

# The risk model data looks like a 2-8 C chain (my reading of the CSV, please confirm)
MODEL_TEMP_RANGE = (2.0, 8.0)

# FoodKeeper storage types: fridge 4 C or below, freezer -18 C or below, pantry 10-21 C.
# The chilled lower bound (0) and the frozen lower bound (-30) are my choices.
STORAGE = {
    "chilled": {"lo": 0.0, "hi": 4.0, "fk": "fridge", "label": "refrigerated"},
    "room": {"lo": 10.0, "hi": 21.0, "fk": "pantry", "label": "room temperature (pantry)"},
    "frozen": {"lo": -30.0, "hi": -18.0, "fk": "freezer", "label": "frozen"},
}
STORAGE_ALIASES = {"fridge": "chilled", "refrigerated": "chilled", "cold": "chilled",
                   "pantry": "room", "ambient": "room", "freezer": "frozen"}

# FoodKeeper is home storage advice, so medicines and chill-sensitive fruit get their own range.
# Values are from memory, verify them.
OVERRIDES = {
    "pharma": {"lo": 2.0, "hi": 8.0, "aliases": ["vaccine", "vaccines", "medicine", "biologic", "insulin"],
               "note": "Refrigerated medicines (2-8 C). Freezing can ruin many of them."},
    "bananas": {"lo": 13.0, "hi": 14.0, "aliases": ["banana"],
                "note": "Chilling injury below ~12-13 C; too warm speeds ripening."},
    "mangoes": {"lo": 10.0, "hi": 13.0, "aliases": ["mango"],
                "note": "Chilling injury below ~10 C."},
}

# General descriptions. The shipper's spec sheet has the real hold time.
PACKAGING = {
    "active": "Powered (reefer, thermoelectric). Needs power or battery for the whole trip plus any delay.",
    "passive": "Insulation + coolant, no power. Typically holds 24-96 h.",
    "hybrid": "Passive coolant regulated by controls. PCM covers a power loss for a limited time.",
}
PASSIVE_HOLD_H = (24, 96)

# Rule-of-thumb thresholds, tune them
HEAT_C = 35.0
RAIN_HOURLY_MM = 10.0
RAIN_TOTAL_MM = 50.0
GUST_KMH = 60.0
EVENT_RADIUS_KM = {"TC": 500, "FL": 150, "EQ": 200, "VO": 100, "DR": 300, "WF": 100}
EVENT_NAMES = {"EQ": "Earthquake", "TC": "Tropical cyclone", "FL": "Flood",
               "VO": "Volcano", "DR": "Drought", "WF": "Wildfire"}
MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


# Helpers
def _get_json(url, params=None):
    if params:
        url = url + "?" + urllib.parse.urlencode(params, safe=";,")
    req = urllib.request.Request(url, headers={"User-Agent": "cold-chain-advisor/0.2"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.load(resp)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def to_vector(p):
    lat, lon = math.radians(p["lat"]), math.radians(p["lon"])
    return (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))


def route_points(origin, dest, step_km=400):
    total = haversine_km(origin["lat"], origin["lon"], dest["lat"], dest["lon"])
    if total < 1:
        return [origin]
    a, b, omega = to_vector(origin), to_vector(dest), total / 6371.0
    n = max(1, int(total // step_km))
    points = []
    for i in range(n + 1):
        f = i / n
        k1 = math.sin((1 - f) * omega) / math.sin(omega)
        k2 = math.sin(f * omega) / math.sin(omega)
        v = [k1 * a[j] + k2 * b[j] for j in range(3)]
        points.append({"lat": math.degrees(math.atan2(v[2], math.hypot(v[0], v[1]))),
                       "lon": math.degrees(math.atan2(v[1], v[0]))})
    return points


def fmt_days(days):
    if days < 21:
        return f"{days:g} d"
    if days < 120:
        return f"{days / 7:.0f} wk"
    return f"{days / 30:.0f} mo"


# Products
_foods = None


def load_foods():
    global _foods
    if _foods is None:
        _foods = []
        if os.path.exists(FOODKEEPER_CSV):
            with open(FOODKEEPER_CSV, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    for s in ("pantry", "fridge", "freezer"):
                        for k in ("min", "max"):
                            value = row[f"{s}_{k}_days"]
                            row[f"{s}_{k}_days"] = float(value) if value else None
                    _foods.append(row)
    return _foods


def foods_available():
    return bool(load_foods())


def food_label(row):
    return f"{row['name']} ({row['subtitle']})" if row["subtitle"] else row["name"]


def shelf_info(row, storage):
    fk = STORAGE[storage]["fk"]
    return {"storage": storage, "min_days": row[f"{fk}_min_days"], "max_days": row[f"{fk}_max_days"],
            "note": row[f"{fk}_note"]}


def has_shelf_info(shelf):
    return shelf["min_days"] is not None or shelf["note"] not in ("", "Not Recommended")


def search_foods(query, n=5):
    q = query.strip().lower()
    if not q:
        return []
    names = {q, q.rstrip("s")}
    scored = []
    for row in load_foods():
        name, keywords = row["name"].lower(), row["keywords"].lower()
        if name in names:
            score = 100
        elif any(name.startswith(x) for x in names):
            score = 80
        elif any(x in name for x in names):
            score = 60
        elif any(x in keywords for x in names):
            score = 40
        else:
            continue
        has_info = any(has_shelf_info(shelf_info(row, s)) for s in STORAGE)
        scored.append((-score, not has_info, len(name) + len(row["subtitle"]), row))
    scored.sort(key=lambda t: t[:3])
    return [t[-1] for t in scored[:n]]


def find_override(name):
    q = name.strip().lower()
    for key, o in OVERRIDES.items():
        if q == key or q in o["aliases"]:
            return key
    return None


def resolve_product(query, storage=None, food=None):
    # query is a name or a (lo, hi) tuple. Returns None if nothing matches.
    if isinstance(query, tuple):
        return {"label": "custom range", "lo": float(query[0]), "hi": float(query[1]),
                "note": "", "source": "your range", "shelf": None}
    key = find_override(query)
    if key:
        o = OVERRIDES[key]
        return {"label": key, "lo": o["lo"], "hi": o["hi"], "note": o["note"],
                "source": "built-in override (verify)", "shelf": None}
    row = food or next(iter(search_foods(query, 1)), None)
    if row is None:
        return None
    storage = (storage or "").lower()
    storage = STORAGE_ALIASES.get(storage, storage)
    if storage not in STORAGE:
        # coldest storage type that has usable FoodKeeper data
        storage = next((s for s in STORAGE if has_shelf_info(shelf_info(row, s))), "chilled")
    s = STORAGE[storage]
    return {"label": f"{food_label(row)} - {s['label']}", "lo": s["lo"], "hi": s["hi"],
            "note": "", "source": "USDA FoodKeeper", "shelf": shelf_info(row, storage)}


def product_flags(prod, transit_days):
    shelf = prod.get("shelf")
    if not shelf:
        return []
    label = STORAGE[shelf["storage"]]["label"]
    if shelf["min_days"] is None:
        if shelf["note"] == "Not Recommended":
            return [("watch", f"FoodKeeper does not recommend storing this {label}.")]
        if shelf["note"]:
            return [("note", f"FoodKeeper gives '{shelf['note']}' for this storage type, so there is no fixed "
                             f"shelf life to compare with the trip.")]
        return [("note", "FoodKeeper has no shelf-life figure for this storage type.")]
    lo, hi = shelf["min_days"], shelf["max_days"]
    span = fmt_days(lo) if lo == hi else f"{fmt_days(lo)} to {fmt_days(hi)}"
    base = (f"FoodKeeper shelf life when {label}: {span} (counted from purchase; "
            f"a rough guide, not a transport spec).")
    if transit_days >= lo:
        return [("watch", f"The {transit_days:g}-day trip is at or beyond the shortest shelf life. {base}")]
    if transit_days + 2 >= lo:
        return [("note", f"A 2-day slip would use up the shortest shelf life. {base}")]
    if transit_days >= 0.5 * lo:
        return [("note", f"The trip uses over half of the shortest shelf life. {base}")]
    return [("ok", base)]


def packaging_flags(kind, transit_days):
    kind = (kind or "").lower()
    if kind not in PACKAGING:
        return []
    flags = [("note", f"{kind.title()} packaging: {PACKAGING[kind]}")]
    if kind == "passive":
        hours = transit_days * 24
        lo, hi = PASSIVE_HOLD_H
        if hours > hi:
            flags.append(("watch", f"The trip is about {hours:.0f} h, beyond the typical {lo}-{hi} h a passive "
                                   f"shipper holds. Consider active/hybrid packaging or re-icing at a hub."))
        elif hours + 24 > hi:
            flags.append(("note", f"About {hours:.0f} h planned: a one-day slip would pass the typical {hi} h "
                                  f"upper limit."))
        elif hours > 48:
            flags.append(("note", f"About {hours:.0f} h planned: make sure the shipper is qualified for that "
                                  f"duration at the expected ambient temperatures."))
    elif kind == "active":
        flags.append(("note", "Ask what happens on power loss or a long delay; carry a data logger with alarms."))
    return flags


# Season (Open-Meteo history)
def geocode(place):
    data = _get_json(GEOCODE_URL, {"name": place, "count": 1, "language": "en"})
    hit = (data.get("results") or [None])[0]
    if not hit:
        raise LookupError(f"place not found: {place}")
    name = ", ".join(x for x in (hit.get("name"), hit.get("admin1"), hit.get("country")) if x)
    return {"name": name, "lat": hit["latitude"], "lon": hit["longitude"]}


def climate_by_month(loc):
    # last 3 full years of daily data
    year = date.today().year
    daily = _get_json(ARCHIVE_URL, {
        "latitude": loc["lat"], "longitude": loc["lon"],
        "start_date": f"{year - 3}-01-01", "end_date": f"{year - 1}-12-31",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum", "timezone": "auto",
    })["daily"]
    months = {m: {"hi": [], "lo": [], "rain": 0.0} for m in range(1, 13)}
    for t, hi, lo, rain in zip(daily["time"], daily["temperature_2m_max"],
                               daily["temperature_2m_min"], daily["precipitation_sum"]):
        m = months[int(t[5:7])]
        if hi is not None:
            m["hi"].append(hi)
        if lo is not None:
            m["lo"].append(lo)
        if rain is not None:
            m["rain"] += rain
    return months


def season_context(loc, when, prod):
    months = climate_by_month(loc)
    m = months[when.month]
    if not m["hi"] or not m["lo"]:
        raise ValueError("no historical data for that month")
    hi, lo = statistics.mean(m["hi"]), statistics.mean(m["lo"])
    rain = m["rain"] / 3
    avg_rain = sum(v["rain"] for v in months.values()) / 3 / 12
    ratio = rain / avg_rain if avg_rain > 1 else None

    if abs(loc["lat"]) < 23.5:  # tropics: wet/dry instead of warm/cold
        if ratio is None:
            label = "a dry climate"
        elif ratio >= 1.3:
            label = "wet season"
        elif ratio <= 0.6:
            label = "dry season"
        else:
            label = "a transition between wet and dry seasons"
    else:
        i = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[when.month]
        if loc["lat"] < 0:
            i = (i + 2) % 4
        label = ["winter", "spring", "summer", "autumn"][i]

    notes = []
    if ratio is not None and ratio >= 1.3:
        notes.append("More rain delays and wet handling than usual.")
    if hi > prod["hi"] + 10:
        notes.append(f"Usual daytime highs are far above the product's {prod['hi']:g} C limit, so time outside "
                     f"cold storage (docks, tarmac, customs) is the weak point.")
    if lo <= 0 and prod["lo"] >= 0:
        notes.append("Nights near or below freezing: a product that must not freeze needs protection "
                     "while waiting outdoors.")
    return {"place": loc["name"], "month": MONTHS[when.month], "label": label, "hi": hi, "lo": lo,
            "rain_mm": rain, "ratio": ratio, "notes": notes}


# Weather (Open-Meteo forecast)
def fetch_hourly(loc, start, end):
    if (end - date.today()).days > FORECAST_HORIZON_DAYS:
        raise ValueError(f"{end} is beyond the ~{FORECAST_HORIZON_DAYS}-day forecast horizon")
    return _get_json(FORECAST_URL, {
        "latitude": loc["lat"], "longitude": loc["lon"],
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_gusts_10m",
        "start_date": start.isoformat(), "end_date": end.isoformat(), "timezone": "auto",
    })["hourly"]


def pick(hourly, key, idx):
    return [hourly[key][i] for i in idx if hourly.get(key) and hourly[key][i] is not None]


def summarize(hourly, day_from, day_to, prod):
    idx = [i for i, t in enumerate(hourly["time"]) if day_from <= date.fromisoformat(t[:10]) <= day_to]
    temp = pick(hourly, "temperature_2m", idx)
    rh = pick(hourly, "relative_humidity_2m", idx)
    rain = pick(hourly, "precipitation", idx)
    gust = pick(hourly, "wind_gusts_10m", idx)
    if not temp:
        raise ValueError("no hourly data returned for the window")
    return {"hours": len(temp), "t_min": min(temp), "t_max": max(temp),
            "rh_max": max(rh) if rh else None,
            "rain_max_mm_h": max(rain) if rain else 0.0, "rain_total_mm": sum(rain) if rain else 0.0,
            "gust_max_kmh": max(gust) if gust else 0.0,
            "hours_above_hi": sum(t > prod["hi"] for t in temp),
            "hours_below_lo": sum(t < prod["lo"] for t in temp)}


def weather_flags(s, where, prod):
    flags = []
    if s["hours_above_hi"]:
        level = "watch" if s["t_max"] >= HEAT_C else "note"
        flags.append((level, f"{where}: air is above the product's {prod['hi']:g} C limit for "
                             f"{s['hours_above_hi']} of {s['hours']} h (peak {s['t_max']:.0f} C). Whenever the "
                             f"shipment sits outside cold storage, the packaging alone has to hold the range."))
    if s["hours_below_lo"] and prod["lo"] >= 0 and s["t_min"] <= 0:
        flags.append(("watch", f"{where}: air drops to {s['t_min']:.0f} C, at or below freezing, and this product "
                               f"must stay above {prod['lo']:g} C. Freezing risk if it waits outdoors."))
    elif s["hours_below_lo"]:
        flags.append(("note", f"{where}: air is below the product's {prod['lo']:g} C lower limit for "
                              f"{s['hours_below_lo']} h (low {s['t_min']:.0f} C)."))
    if s["rain_max_mm_h"] >= RAIN_HOURLY_MM or s["rain_total_mm"] >= RAIN_TOTAL_MM:
        flags.append(("watch", f"{where}: heavy rain ({s['rain_total_mm']:.0f} mm total, up to "
                               f"{s['rain_max_mm_h']:.0f} mm/h). Delays and wet handling are more likely."))
    if s["gust_max_kmh"] >= GUST_KMH:
        flags.append(("watch", f"{where}: gusts up to {s['gust_max_kmh']:.0f} km/h can delay flights, "
                               f"sea legs and loading."))
    return flags


# Disaster alerts (GDACS)
def event_point(geometry):
    if not geometry:
        return None
    c = geometry.get("coordinates")
    while isinstance(c, list) and c and isinstance(c[0], list):
        c = c[0]
    return (c[1], c[0]) if isinstance(c, list) and len(c) >= 2 else None


def gdacs_events(ship, arrive, points):
    data = _get_json(GDACS_URL, {
        "eventlist": "EQ;TC;FL;VO;DR;WF",
        "fromdate": (ship - timedelta(days=14)).isoformat(),  # events already under way
        "todate": (arrive + timedelta(days=1)).isoformat(),
        "alertlevel": "orange;red",
    })
    found = []
    for feature in data.get("features", []):
        p = feature.get("properties", {})
        loc = event_point(feature.get("geometry"))
        etype = p.get("eventtype")
        if not loc or etype not in EVENT_RADIUS_KM:
            continue
        end = str(p.get("todate", ""))[:10]
        if end and date.fromisoformat(end) < ship - timedelta(days=1):
            continue
        dists = [haversine_km(loc[0], loc[1], q["lat"], q["lon"]) for q in points]
        nearest = min(range(len(dists)), key=dists.__getitem__)
        if dists[nearest] > EVENT_RADIUS_KM[etype]:
            continue
        if nearest == 0:
            where = "near the origin"
        elif nearest == len(points) - 1:
            where = "near the destination"
        else:
            where = "along the route"
        found.append({"type": EVENT_NAMES[etype], "level": str(p.get("alertlevel", "")).title(),
                      "name": p.get("name") or p.get("eventname") or "", "country": p.get("country", ""),
                      "where": where, "km": round(dists[nearest]),
                      "severity": (p.get("severitydata") or {}).get("severitytext", ""),
                      "link": (p.get("url") or {}).get("report", "")})
    return sorted(found, key=lambda e: ({"Red": 0, "Orange": 1}.get(e["level"], 2), e["km"]))


# Briefing
def briefing(origin, destination, ship_date, transit_days, product, storage=None, packaging=None):
    # product: a resolve_product() result, a name, or a (lo, hi) tuple
    prod = product if isinstance(product, dict) else resolve_product(product, storage)
    if prod is None:
        raise KeyError(f"unknown product '{product}'. Try another name, or give a range like (2, 8)")
    arrive = ship_date + timedelta(days=max(1, math.ceil(transit_days)))
    out = {"product": prod, "ship": ship_date, "arrive": arrive,
           "checks": product_flags(prod, transit_days) + packaging_flags(packaging, transit_days)}
    lo, hi = MODEL_TEMP_RANGE
    out["model_applies"] = abs(prod["lo"] - lo) <= 1.5 and abs(prod["hi"] - hi) <= 1.5

    try:
        o, d = geocode(origin), geocode(destination)
        out["places"] = {"origin": o["name"], "destination": d["name"]}
    except Exception as e:
        error = {"ok": False, "error": str(e)}
        out.update(season=error, weather=error, events=error)
        return out

    try:
        season_o = season_context(o, ship_date, prod)
        same_place = (d["lat"], d["lon"]) == (o["lat"], o["lon"]) and arrive.month == ship_date.month
        items = [season_o] if same_place else [season_o, season_context(d, arrive, prod)]
        out["season"] = {"ok": True, "items": items}
    except Exception as e:
        out["season"] = {"ok": False, "error": str(e)}

    try:
        day_after = arrive + timedelta(days=1)  # arrival day plus one day of slip
        so = summarize(fetch_hourly(o, ship_date, ship_date), ship_date, ship_date, prod)
        sd = summarize(fetch_hourly(d, arrive, day_after), arrive, day_after, prod)
        flags = (weather_flags(so, f"Departure day at {o['name']}", prod)
                 + weather_flags(sd, f"Arrival at {d['name']} (and one day of slip)", prod))
        out["weather"] = {"ok": True, "origin": so, "destination": sd, "flags": flags}
    except Exception as e:
        out["weather"] = {"ok": False, "error": str(e)}

    try:
        out["events"] = {"ok": True, "items": gdacs_events(ship_date, arrive, route_points(o, d))}
    except Exception as e:
        out["events"] = {"ok": False, "error": str(e)}
    return out


def print_briefing(b):
    p = b["product"]
    print("\n================================")
    print("TRIP BRIEFING (context only, not part of the risk score)")
    print("================================")
    print(f"Product: {p['label']}  ->  keep at {p['lo']:g} to {p['hi']:g} C   [{p['source']}]")
    if p["note"]:
        print(f"  {p['note']}")
    if not b["model_applies"]:
        print(f"  ! The risk score was learned from data that looks like a {MODEL_TEMP_RANGE[0]:g}-"
              f"{MODEL_TEMP_RANGE[1]:g} C chain. For this product, read it as a 'long / complicated trip' "
              f"signal, not a calibrated failure rate.")
    if "places" in b:
        print(f"Route: {b['places']['origin']} -> {b['places']['destination']}  ({b['ship']} to about {b['arrive']})")
    for level, text in b["checks"]:
        print(f"  [{level}] {text}")

    print("\nSeason (Open-Meteo history, last 3 years):")
    season = b["season"]
    if not season["ok"]:
        print(f"  unavailable: {season['error']}")
    else:
        for item in season["items"]:
            extra = f", rain {item['ratio']:.1f}x the average month" if item["ratio"] else ""
            print(f"  {item['place']}: {item['month']} is usually {item['label']} (high {item['hi']:.0f} C, "
                  f"low {item['lo']:.0f} C, about {item['rain_mm']:.0f} mm rain{extra}).")
            for note in item["notes"]:
                print(f"     - {note}")

    print("\nWeather (Open-Meteo forecast):")
    weather = b["weather"]
    if not weather["ok"]:
        print(f"  unavailable: {weather['error']}")
    else:
        for name in ("origin", "destination"):
            w = weather[name]
            print(f"  {name:<12} {w['t_min']:.0f} to {w['t_max']:.0f} C, rain {w['rain_total_mm']:.0f} mm, "
                  f"gusts {w['gust_max_kmh']:.0f} km/h")
        for level, text in weather["flags"] or [("ok", "Nothing unusual in the forecast at either end.")]:
            print(f"  [{level}] {text}")

    print("\nDisaster alerts (GDACS, orange/red only):")
    events = b["events"]
    if not events["ok"]:
        print(f"  unavailable: {events['error']}")
    elif not events["items"]:
        print("  None near the straight-line route. (Real routing via hubs may differ.)")
    else:
        for e in events["items"]:
            print(f"  [{e['level']}] {e['type']} {e['name']} {e['country']}: {e['km']} km from the route, "
                  f"{e['where']}. {e['severity']}")
            if e["link"]:
                print(f"      {e['link']}")
    print("\nSources: USDA FoodKeeper; Open-Meteo.com; Global Disaster Alert and Coordination System (GDACS).")
    print("Automated alerts can be wrong or late; check local authorities before acting.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 5:
        sys.exit('usage: python advisory.py "Origin" "Destination" YYYY-MM-DD transit_days product '
                 '[chilled|room|frozen] [active|passive|hybrid]')
    print_briefing(briefing(args[0], args[1], date.fromisoformat(args[2]), float(args[3]), args[4],
                            storage=args[5] if len(args) > 5 else None,
                            packaging=args[6] if len(args) > 6 else None))

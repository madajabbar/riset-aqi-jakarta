"""OpenAQ v3 station-observation access for the Jakarta window.

Two modes:
  --scan : inventory PM2.5 sensors within 25 km of Monas + observation counts
  --dump : fetch hourly observations per sensor -> our_data/openaq_raw.csv

Auth: env OPENAQ_API_KEY (x-api-key header). API key -> station metadata +
hourly series via /v3/sensors/{id}/measurements/hourly (datetime_from/_to,
UTC timestamps; local = UTC+7).
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

LAT, LON = -6.2088, 106.8456  # Monas
RADIUS_M = 25000
WIN_FROM = "2022-08-01T00:00:00Z"
WIN_TO = "2026-10-02T00:00:00Z"
BASE = "https://api.openaq.org/v3"
OUT = Path(__file__).resolve().parent.parent / "our_data"
KEY = os.environ.get("OPENAQ_API_KEY", "")


def get(path, **params):
    r = requests.get(BASE + path, headers={"x-api-key": KEY},
                     params=params, timeout=60)
    if r.status_code == 429:
        wait = int(r.headers.get("Retry-After", "30"))
        print(f"  429, sleep {wait}s", file=sys.stderr)
        time.sleep(wait + 1)
        return get(path, **params)
    r.raise_for_status()
    return r.json()


def pm25_sensors():
    """(loc_id, loc_name, sensor_id) for all PM2.5 sensors in the radius."""
    out = []
    d = get("/locations", coordinates=f"{LAT},{LON}", radius=RADIUS_M,
            limit=100)
    for loc in d["results"]:
        sid = get(f"/locations/{loc['id']}/sensors", limit=100)
        for s in sid["results"]:
            p = (s.get("parameter") or {}).get("name", "")
            if p == "pm25":
                out.append((loc["id"], loc.get("name", "?"), s["id"]))
        time.sleep(0.3)
    return out


def scan():
    print(f"{'loc':>8} {'sensor':>9} {'n_1h':>7}  name")
    for lid, name, sid in pm25_sensors():
        try:
            d = get(f"/sensors/{sid}/measurements/hourly",
                    datetime_from=WIN_FROM, datetime_to=WIN_TO, limit=1)
            n = int(str(d["meta"]["found"]).replace('>', ''))
        except requests.HTTPError as e:
            n = -1
            print("ERR", e)
        print(f"{lid:>8} {sid:>9} {n:>7}  {name[:44]}")
        time.sleep(0.3)


def dump(sensor_ids=None):
    rows = {}
    for lid, name, sid in pm25_sensors():
        if sensor_ids and sid not in sensor_ids:
            continue
        page = 1
        while True:
            d = get(f"/sensors/{sid}/measurements/hourly",
                    datetime_from="2026-06-01T00:00:00Z", datetime_to="2026-08-01T00:00:00Z",
                    limit=1000, page=page)
            res = d["results"]
            for m in res:
                t = m["period"]["datetimeFrom"]["utc"]
                v = m.get("value")
                qual = m.get("isValidQuality")
                rows.setdefault(t, []).append((name, v, qual))
            total = int(str(d["meta"]["found"]).replace('>', ''))
            got = page * 1000
            print(f"  sensor {sid} {name[:28]:28} {min(got,total)}/{total}",
                  file=sys.stderr)
            if got >= total or not res:
                break
            page += 1
            time.sleep(0.5)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "openaq_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utc_from", "n_obs", "values"])
        for t in sorted(rows):
            obs = rows[t]
            vals = ";".join(f"{n}={v}" for n, v, q in obs)
            w.writerow([t, len(obs), vals])
    print(f"wrote {len(rows)} hourly rows -> our_data/openaq_raw.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", action="store_true")
    g.add_argument("--dump", action="store_true")
    ap.add_argument("--sensors", default="",
                    help="comma-separated sensor ids to dump (default all)")
    args = ap.parse_args()
    if not KEY:
        sys.exit("set OPENAQ_API_KEY")
    if args.scan:
        scan()
    else:
        ids = {int(x) for x in args.sensors.split(",") if x}
        dump(ids or None)

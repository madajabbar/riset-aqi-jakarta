"""Fetch Jakarta hourly air quality (Open-Meteo / CAMS) + weather (Open-Meteo / ERA5).

Anonymous, reproducible, keyless. Outputs into ../our_data:
  jakarta_aqi.csv : datetime, us_aqi, pm2_5, pm10, co, no2, o3, so2
  jakarta_met.csv : datetime, temp, rh, wind_speed, wind_dir, pressure, precip, swr

Swap-in note: if OpenAQ station obs (needs free API key) become available later,
produce a jakarta_aqi.csv with the same columns and the rest of the pipeline is
unchanged (observation-based target instead of CAMS reanalysis).
"""
import datetime as dt
import sys
import time
from pathlib import Path

import requests

LAT, LON = -6.2088, 106.8456  # central Jakarta
TZ = "Asia/Jakarta"
OUT = Path(__file__).resolve().parent.parent / "our_data"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MET_URL = "https://archive-api.open-meteo.com/v1/archive"
AQ_VARS = "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
MET_VARS = ("temperature_2m,relative_humidity_2m,wind_speed_10m,"
            "wind_direction_10m,surface_pressure,precipitation,shortwave_radiation")
S = requests.Session()
S.headers["User-Agent"] = "riset-aqi-jakarta/1.0"


def chunks(start, end, days=360):
    d = start
    while d <= end:
        yield d, min(d + dt.timedelta(days=days), end)
        d = min(d + dt.timedelta(days=days + 1), end + dt.timedelta(days=1))


def write_csv(path, head_times, keys, grid, col_names):
    lines = [",".join(["datetime"] + col_names)]
    for t in head_times:
        lines.append(t + "," + ",".join("" if v is None else str(v) for v in grid[t]))
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(head_times)} rows)")


def run(start, end):
    for label, url, params, cols, fname in [
        # NOTE: air-quality-api returns nulls before ~2022-08 for this region;
        # it only accepts past_days (max data depth), not deep start_date.
        ("AQ", AQ_URL, {"hourly": AQ_VARS, "past_days": 3700},
         ["us_aqi", "pm2_5", "pm10", "co", "no2", "o3", "so2"], "jakarta_aqi.csv"),
        ("MET", MET_URL, {"hourly": MET_VARS, "models": "era5"},
         ["temp", "rh", "wind_speed", "wind_dir", "pressure", "precip", "swr"], "jakarta_met.csv"),
    ]:
        print(f"[{label}] fetching {start}..{end}")
        p = dict(params)
        grid = {}
        if "past_days" in p:
            pp = dict(p, latitude=LAT, longitude=LON, timezone=TZ)
            r = S.get(url, params=pp, timeout=300)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
            h = r.json()["hourly"]
            keys = [k for k in h if k != "time"]
            for i, t in enumerate(h["time"]):
                grid[t] = [h[k][i] for k in keys]
            times = sorted(grid)
            write_csv(OUT / fname, times, keys, grid, cols)
            continue
        for a, b in chunks(start, end):
            pp = dict(p, start_date=a.isoformat(), end_date=b.isoformat(),
                      latitude=LAT, longitude=LON, timezone=TZ)
            for attempt in range(4):
                try:
                    r = S.get(url, params=pp, timeout=180)
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
                    h = r.json()["hourly"]
                    keys = [k for k in h if k != "time"]
                    for i, t in enumerate(h["time"]):
                        grid[t] = [h[k][i] for k in keys]
                    break
                except Exception as e:
                    print("  retry:", e)
                    if attempt == 3:
                        raise
                    time.sleep(10 * (attempt + 1))
            print(f"  {a}..{b} ok ({len(grid)} rows so far)", flush=True)
        # keys order is insertion-stable per API docs (matches param order)
        times = sorted(grid)
        assert len(times) == len(grid)
        write_csv(OUT / fname, times, keys, grid, cols)


if __name__ == "__main__":
    start = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date(2015, 1, 1)
    end = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date.today() - dt.timedelta(days=2)
    OUT.mkdir(exist_ok=True)
    run(start, end)

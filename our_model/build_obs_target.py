"""Aggregate OpenAQ station observations -> our_data_obs/jakarta_aqi.csv.

Input : ../our_data/openaq_raw.csv  (utc_from, n_obs, "Station=value;...")
Output: ../our_data_obs/jakarta_aqi.csv (datetime WIB, us_aqi, pm2_5, n_stations)

Method (station-based observation target, replaces CAMS reanalysis):
 - per UTC hour pool all station pm2.5 values; drop invalid (<0, sentinel -999)
 - city value = MEDIAN across stations (robust to one bad low-cost sensor)
 - local datetime = UTC + 7 (WIB), matching jakarta_met.csv index
 - us_aqi derived from pm2.5 via EPA 2024 breakpoints (kept only for the
   pipeline's notna filter; experiment target is pm2_5)
"""
import statistics
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "our_data" / "openaq_raw.csv"
OUTDIR = HERE.parent / "our_data_obs"
BP = [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
      (55.5, 125.4, 151, 200), (125.5, 150.4, 201, 250),
      (150.5, 250.4, 251, 300)]  # EPA 2024 pm2.5 24h (approx for hourly use)


def aqi(c):
    for cl, ch, il, ih in BP:
        if c <= ch:
            return round((ih - il) / (ch - cl) * (c - cl) + il)
    return 500


def main():
    rows = []
    with open(RAW) as f:
        header = f.readline()
        for line in f:
            t, _, vals = line.rstrip("\n").split(",", 2)
            obs = []
            for part in vals.split(";"):
                _, _, v = part.rpartition("=")
                try:
                    x = float(v)
                except ValueError:
                    continue
                if 0 <= x < 1000:
                    obs.append(x)
            if obs:
                rows.append((t, round(statistics.median(obs), 2), len(obs)))
    df = pd.DataFrame(rows, columns=["utc", "pm2_5", "n_stations"])
    df["datetime"] = (pd.to_datetime(df["utc"]).dt.tz_convert("Asia/Jakarta")
                      .dt.tz_localize(None).dt.floor("h"))
    df = df.groupby("datetime").agg(
        pm2_5=("pm2_5", "mean"), n_stations=("n_stations", "max")).reset_index()
    df["us_aqi"] = df["pm2_5"].map(aqi)
    df = df[["datetime", "us_aqi", "pm2_5", "n_stations"]]
    OUTDIR.mkdir(exist_ok=True)
    df.to_csv(OUTDIR / "jakarta_aqi.csv", index=False)
    (OUTDIR / "jakarta_met.csv").symlink_to(
        (HERE.parent / "our_data" / "jakarta_met.csv").resolve())
    print(f"{len(df)} hourly obs rows, "
          f"{df['datetime'].min()} .. {df['datetime'].max()} -> {OUTDIR}")
    print(df.describe().loc[["mean", "50%", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()

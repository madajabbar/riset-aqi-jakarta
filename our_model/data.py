"""Data loading & tensor prep for EMD-Transformer-BiLSTM + weather on Jakarta.

Mirrors the baseline protocol (Dong et al. 2024, air_patna.csv):
 - MinMaxScaler(-1,1) on the whole target series, 75/25 chronological split
   (same reproducible-but-optimistic protocol as the reference impl; kept so
   ablation comparisons are apples-to-apples)
 - sliding windows of `input_window` steps, one-step/h-step ahead labels
Our addition: exogenous weather channels aligned per timestamp.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_merged(data_dir):
    aqi = pd.read_csv(data_dir + "/jakarta_aqi.csv", parse_dates=["datetime"]).set_index("datetime")
    met = pd.read_csv(data_dir + "/jakarta_met.csv", parse_dates=["datetime"]).set_index("datetime")
    df = aqi.join(met, how="left")
    # trim to span where AQ columns actually exist (Open-Meteo obs window)
    df = df[df[["us_aqi", "pm2_5"]].notna().any(axis=1)]
    # fill full hourly grid, interpolate small gaps
    idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(idx).interpolate(limit_area="inside")
    # wind direction -> sin/cos (circular-safe), drop raw
    wd = np.deg2rad(df["wind_dir"].ffill().bfill())
    df["wind_sin"], df["wind_cos"] = np.sin(wd), np.cos(wd)
    df = df.drop(columns=["wind_dir"])
    df = df.dropna(subset=["us_aqi"])
    return df


WEATHER_COLS = ["temp", "rh", "wind_speed", "pressure", "precip", "swr", "wind_sin", "wind_cos"]


def make_series(df, start=None, end=None, target="us_aqi"):
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index < end]
    y = df[target].values.astype(np.float64)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
    w = df[WEATHER_COLS].values.astype(np.float64)
    ok = ~np.isnan(w).any(1)
    mu, sd = w[ok].mean(0), w[ok].std(0) + 1e-8
    w = (w - mu) / sd
    keep = ok
    return y[keep], w[keep], scaler, df[keep]


def make_windows(y, w, input_window, horizon, train_frac=0.75):
    """Returns X (N, tw, 1+K), Y (N, tw, 1), split index at train_frac."""
    n = len(y) - input_window - horizon + 1
    X = np.empty((n, input_window, 1 + w.shape[1]), np.float32)
    Y = np.empty((n, input_window, 1), np.float32)
    for i in range(n):
        X[i, :, 0] = y[i:i + input_window]
        X[i, :, 1:] = w[i:i + input_window]
        Y[i, :, 0] = y[i + horizon: i + input_window + horizon]
    split = int(round(len(X) * train_frac))
    return X, Y, split

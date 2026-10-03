# our_model — EMD-Transformer-BiLSTM + exogenous weather (Jakarta)

Adaptasi `baseline_emd-transformer-bilstm.py` (Dong et al. 2024) sesuai plan:
1. **Data**: Jakarta, hourly, 2015→sekarang — `our_data/jakarta_aqi.csv` (CAMS/US-AQI + 6 polutan, Open-Meteo) + `jakarta_met.csv` (ERA5: suhu, RH, angin, tekanan, hujan, radiasi). Keyless & reproducible. Kalau nanti dapat API key OpenAQ (gratis), ganti `jakarta_aqi.csv` dengan data stasiun observasi — kolom sama, pipeline tidak berubah.
2. **Novelty** (bukan arsitektur baru): channel eksogen meteorologi masuk ke (a) Transformer per-IMF (`TransAm(n_exog=8)`, Linear embedding menggantikan broadcast-hack rujukan) dan (b) BiLSTM fusi. Ablation = RQ2.
3. **Domain**: Jakarta tropis vs Patna/Dong.

## Pakai
```bash
python our_model/fetch_data.py 2015-01-01 2026-10-01   # refresh data
python our_model/train.py --weather                     # our model
python our_model/train.py                               # ablation: no weather (base)
python our_model/train.py --smoke                       # quick sanity (3 ep)
```
Colab T4: `git clone` repo → `!pip install EMD-signal scikit-learn pandas matplotlib` → jalankan `train.py --weather --epochs 100` (EMD jalan di CPU, ±10–20 mnt untuk 100k jam).

## Protokol (disengaja identik dgn rujukan, biar fair)
- MinMaxScaler(-1,1) pada seluruh series + EMD seluruh series (split 75/25 kronologis) — sama seperti baseline (split terjadi setelah dekomposisi). Dicatat di paper sebagai protokol reproduksi rujukan.
- Per-IMF: Transformer-BiLSTM vs LinearRegression, yang RMSE-nya menang dipakai (mekanisme proteksi rujukan dipertahankan).
- Target default `us_aqi`, horizon 1 jam (`--horizon` untuk 5–24 jam).

## Output
`our_model/results/jakarta-{target}-{base|weather}-win{W}-h{H}.csv` (+ .png), metrik RMSE/MAE/MAPE diprint.

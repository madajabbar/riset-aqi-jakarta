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

## Protokol
- Split **tiga fase kronologis 70/15/15** (train/validasi/test); `TRAIN_FRAC`/`VAL_FRAC` di `train.py`.
- Per-IMF: input = window **IMF itu sendiri** (+ kanal cuaca), target = window IMF yang sama. Transformer-BiLSTM vs LinearRegression rujukan diadu, pemenang dipilih dari **blok validasi** saja.
- Fusi BiLSTM dilatih atas **prediksi out-of-sample blok validasi** (+ cuaca) → AQI, lalu diterapkan ke blok test — distribusi input train & test setara.
- Metrik RMSE/MAE/MAPE hanya dari blok test.
- Bawaan rujukan yang **tidak** diubah (ditulis sebagai limitasi di paper): MinMaxScaler(-1,1) + EMD dihitung atas seluruh seri sebelum split; statistik cuaca (mean/std) + interpolasi juga atas seluruh periode. Varian bebas-leakage = pekerjaan RQ1 berikutnya.
- Target default `us_aqi`, horizon 1 jam (`--horizon` untuk 5–24 jam).

## Riwayat perbaikan 10 Okt 2026 (audit eksternal, diverifikasi langsung di kode)
1. Loop per-IMF dulu bertarget seri AQI penuh → Transformer tidak pernah melihat IMF (ciri di log: loss mentok ±0,08295 berulang, `-> lin` 13/13). Diperbaiki.
2. Pemilihan tf-vs-lin dulu dihitung di data test (test-set selection bias) → sekarang blok validasi.
3. Fusi dulu dilatih atas IMF asli, diuji atas IMF prediksi (exposure bias) → sekarang dilatih atas prediksi OOS validasi.
4. Checkpoint pindah ke `results/checkpoints/<lengan>-v2/`; file format lama (tanpa `pred_va`) otomatis dilatih ulang, bukan crash.
→ Semua angka sebelum commit ini (weather 2,3601 GPU / base 2,3484 lokal) **tidak layak** dikutip di paper.

## Output
`our_model/results/jakarta-{target}-{base|weather}-win{W}-h{H}.csv` (+ .png), metrik RMSE/MAE/MAPE diprint.

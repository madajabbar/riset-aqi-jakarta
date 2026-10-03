# our_data — Jakarta hourly, WIB (Asia/Jakarta)

- `jakarta_aqi.csv` — kolom: `datetime, us_aqi, pm2_5, pm10, co, no2, o3, so2`
  Sumber: Open-Meteo Air Quality API (CAMS), tanpa key. Window berisi data:
  **2022-08 → kini** (~36,600 jam valid; API mengembalikan null sebelum 2022-08
  untuk region ini).
- `jakarta_met.csv` — kolom: `datetime, temp, rh, wind_speed, wind_dir, pressure, precip, swr`
  Sumber: Open-Meteo Historical Weather API (ERA5), 2015 → kini.

Refresh: `python our_model/fetch_data.py START END`.

Catatan penting:
- Target = reanalysis/observasi-blend CAMS versi Open-Meteo, bukan sensor stasiun lokal.
  Upgrade path (lebih kuat utk reviewer): API key gratis OpenAQ v3 → ganti
  `jakarta_aqi.csv` dengan observasi stasiun (schema kolom sama, pipeline tak berubah).
- Koordinat titik: -6.2088, 106.8456 (Jakarta pusat).

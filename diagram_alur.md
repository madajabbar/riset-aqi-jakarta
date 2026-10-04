# Diagram Alur — EMD–Transformer–BiLSTM + Weather (Jakarta)

Render: paste ke https://mermaid.live → export PNG/SVG, atau `mmdc -i diagram_alur.md -o flowchart.png`.

```mermaid
flowchart TD
    A["Data Mentah<br/>Open-Meteo (CAMS)<br/>AQI + 6 polutan"] --> B["Data Meteorologi<br/>ERA5: suhu, RH, angin,<br/>tekanan, hujan, radiasi"]

    A --> C["Preprocessing<br/>merge & imputasi"]
    B --> C

    C --> D["Scaling<br/>MinMaxScaler(-1, 1)"]
    D --> E["Windowing<br/>window = 5 · horizon = 1"]

    E --> F["Dekomposisi EMD<br/>(12 IMF)"]
    F --> G["IMF₁ … IMF₁₂"]

    G --> H["Per-IMF<br/>Transformer–BiLSTM<br/>(+ channel weather)"]
    B -.-> H
    H --> I["Fallback per-IMF<br/>LinearRegression"]
    I --> J{"Pilih RMSE<br/>terkecil?"}
    J -->|Transformer menang| K["Prediksi IMF<br/>(Transformer)"]
    J -->|Linear menang| L["Prediksi IMF<br/>(Linear)"]

    K --> M["Fusi BiLSTM<br/>(IMF prediksi + weather)"]
    L --> M
    B -.-> M

    M --> N["Inverse scaling"]
    N --> O["Metrik<br/>RMSE · MAE · MAPE"]

    subgraph ABLASI["Ablation (RQ2)"]
        P["Base<br/>tanpa weather"]
        Q["Our model<br/>dengan weather"]
    end
    O --> ABLASI
```

## Ringkasan alur

1. **Data**: `fetch_data.py` (Open-Meteo CAMS + ERA5) → `our_data/`.
2. **Preprocessing**: `data.py` — merge, MinMaxScaler(-1,1), windowing (win=5, h=1).
3. **Dekomposisi**: `EMD(max_imf=12)` → 12 IMF.
4. **Per-IMF**: `model.TransAm(n_exog)` (Transformer–BiLSTM) vs `LinearRegression`; RMSE terkecil dipakai (proteksi rujukan).
5. **Fusi**: `model.LstmRNN` (BiLSTM) atas IMF prediksi + weather → AQI.
6. **Evaluasi**: inverse-scale → RMSE / MAE / MAPE.
7. **Ablation**: `--weather` vs base (`train.py`).
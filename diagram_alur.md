# Sistem Prediksi AQI Jakarta — EMD–Transformer–BiLSTM + Weather (RQ2)

**Tujuan:** Ekstensi arsitektur Dong et al. (2024) dengan integrasi data meteorologi exogenous (ERA5/CAMS) pada pipeline *forecasting*.  
**Dataset:** Open-Meteo API (historical air quality + weather), region DKI Jakarta 2022–2026.  
**Arketipe:** Ensemble Decomposition → Per-Component Deep Learning → Fusion Stage.

---

## Arsitektur Sistem

Render online: [Mermaid.live](https://mermaid.live) (`Paste code di tab Source → Export PNG/SVG`).  
Render CLI: `npx @mermaid-js/mermaid-cli -i diagram_alur.md -o flowchart.png`

```mermaid
flowchart TD
    %% Subgraph: Data Acquisition
    subgraph DATA ["Fase 1: Akuisisi & Preprocessing"]
        direction TB
        AQ[("Data Air Quality<br/><i>CAMS Reanalysis</i><br/>US-AQI • PM2.5 • PM10<br/>CO • NO₂ • O₃ • SO₂")] --> MERGE["Merge & Alignment<br/>(hourly grid)]
        MET[("Data Meteorologi<br/><i>ERA5 Reanalysis</i><br/>Temp • RH • Wind<br/>Pressure • Precip<br/>Radiasi (SWR)")] --> MERGE
        
        MERGE --> IMPOUT[Imputasi Gap<br/>(interpolation limit_area)]
        IMPOUT --> WIND[Wind Encoding<br/>sin/cos circular]
        WIND --> SCALE[Feature Scaling<br/>MinMaxScaler ∈ [-1, 1]]
    end

    %% Subgraph: Windowing
    subgraph WIN["Fase 2: Konstruksi Tensor"]
        SCALE --> WINDOW[Sliding Window<br/>Input: 5 langkah | Horizon: 1]
        WINDOW --> SPLIT{Chronological Split<br/>Train/Test = 75% / 25%}
    end

    %% Subgraph: Decomposition & Per-IMF Training
    subgraph EMD_STAGE ["Fase 3: Dekomposisi & Komponen-Independen"]
        SPLIT --> DECOMP[Empirical Mode Decomposition<br/>EMD(max_imf=12)]
        DECOMP --> IMF_LIST[12 Instan Mode Fungsi (IMFs)]
        
        IMF_LIST --> TRANS_PER_IMF["Per-IMF Modeling<br/>TransAm(n_exog=<b>K</b>)"]
        TRANS_PER_IMF --> TF_LSTM[Transformer Encoder<br/>+ BiLSTM Decoder<br/>+ Linear Head]
        
        TRANS_PER_IMF -.-> LIN_PER_IMF["Baseline Fallback<br/>LinearRegression<br/>(reference safeguard)"]
        
        TF_LSTM --> COMPARE{"Pilih Best Predictor<br/>(RMSE test)"}
        LIN_PER_IMF --> COMPARE
        
        COMPARE -->|Transformer Win| PRED_TF[Prediksi IMF<br/>(Transformer-BiLSTM)]
        COMPARE -->|Linear Regression Win| PRED_LIN[Prediksi IMF<br/>(Linear Fallback)]
    end

    %% Subgraph: Fusion Stage
    subgraph FUSE ["Fase 4: Fusi Multi-Komponen"]
        PRED_TF --> FUSION_INPUT["Stacked Predictions<br/>[Pred_ΔIMF₁ … Pred_ΔIMF₁₂] + Weather"]
        PRED_LIN --> FUSION_INPUT
        
        FUSION_INPUT --> LSTM_FUSE[LstmRNN (Fusion Network)<br/>Bidirectional LSTM<br/>Input: 12+K features]
        LSTM_FUSE --> RAW_PRED["Final Prediction<br/>AQI (scaled domain)"]
    end

    %% Subgraph: Evaluation
    subgraph EVAL ["Fase 5: Evaluasi Metrik"]
        RAW_PRED --> INVSCALE[Inverse Transformation<br/>(scaler.inverse_transform)]
        INVSCALE --> REAL_PRED["Predicted AQI (real-scale)"]
        REAL_PRED --> METRICS[{"Evaluasi Statistik<br/>RMSE • MAE • MAPE"}]
    end

    %% Ablation Study RQ2
    subgraph ABLATION ["Ablasi Studi (RQ2 Comparison)"]
        BASE_ONLY[Base Model<br/>(AQI-only, no exog)] --> MEASURE_A
        WEATHER_ADD[Our Model<br/>(+Weather Exog)] --> MEASURE_B
        MEASURE_A[Metrics Phase 1] --> COMP_ABALSION{Statistical<br/>Comparison}
        MEASURE_B[Metrics Phase 2] --> COMP_ABALSION
    end

    %% Connections between main stages
    DATA --- WIN
    WIN --- EMD_STAGE
    EMD_STAGE --- FUSE
    FUSE --- EVAL
    
    %% Dashed connections for cross-dependencies
    MET -.->|Weather Channels K| TRANS_PER_IMF
    MET -.->|Exogenous Features| FUSION_INPUT
    
    style DATA fill:#e8f4f8,stroke:#06B6D4,stroke-width:2px
    style WIN fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style EMD_STAGE fill:#dbeafe,stroke:#2563eb,stroke-width:2px
    style FUSE fill:#f3e8ff,stroke:#9333ea,stroke-width:2px
    style EVAL fill:#ecfdf5,stroke:#059669,stroke-width:2px
    style ABLATION fill:#fee2e2,stroke:#dc2626,stroke-width:2px
    style COMPARE fill:#ffffff,stroke:#1e293b,stroke-width:3px
    style COMP_ABALSION fill:#ffffff,stroke:#1e293b,stroke-width:3px
    style DECISIONS stroke-width:2px
```

---

## Metodologi Implementasi

### 1. Ingest & Preparation (`fetch_data.py`)
* **Sumber:** Open-Meteo API (CAMS reanalysis untuk air quality; ERA5 untuk meteorologi).
* **Output:** CSV harian → merge menjadi `jakarta_aqi.csv` + `jakarta_met.csv`.
* **Validasi:** Interpolasi gap kecil via `limit_area='inside'`, encoding arah angin sin/cos.

### 2. Transformasi Numerik (`data.py`)
* **Scaling:** `MinMaxScaler(feature_range=(-1, 1))` pada seluruh target series (protocol sesuai referensi).
* **Windowing:** Sliding window `input_window = 5` → `horizon = 1`.
* **Split:** Chronological 75:25 (train:test).

### 3. Dekomposisi Adaptif (`PyEMD`)
* Memecah signal non-stasioner AQI menjadi **12 IMF** (Intrinsic Mode Functions).
* Setiap IMF mewakili frekuensi/osilasi berbeda dari pola polusi.

### 4. Pelatihan Independent (Per-IMF)
Setiap IMF dilatih secara independen melalui dua arsitektur paralel:
1. **TransAm (Extended):** Transformer Encoder → BiLSTM Decoder → Linear Output. Mendukung channel cuaca exogenous via `nn.Linear(1+K, d_model)`.
2. **Linear Regression:** Regresi linier klasik sebagai *sanity check & fallback* (safeguard sesuai protokol referensi).
3. **Seleksi Otomatis:** Pilih prediktor dengan RMSE terkecil pada subset uji setiap IMF.

### 5. Fusion Network
Prediksi dari semua komponen yang telah terpilih digabungkan ke dalam satu jaringan **LstmRNN** (BiLSTM dengan input multi-feature) untuk menghasilkan output AQI final tunggal.

### 6. Evaluasi & Metrik
* **Inverse Scale:** Kembalikan prediksi ke skala asli (unit AQI riil).
* **Metric:** RMSE, MAE, MAPE dihitung terhadap ground truth real-scale.
* **Visualisasi:** Plot comparison prediction vs truth (`plot_result.py`).

---

## Ablasi Studi (Research Question 2)

| Variabel | Eksperimen | Tujuan Penelitian |
| :--- | :--- | :--- |
| **Base Model** | Training tanpa fitur meteorologi (AQI-only) | Establish baseline performa arsitektur murni |
| **Our Model** | Training dengan fitur cuaca (temp, wind, pressure, dll.) | Mengukur kontribusi exogenous variables terhadap akurasi |
| **Statistical Test** | Compare RMSE/MAE/MAPE antar fase | Validasi signifikansi penambahan weather input |

Parameter default mengacu protocol paper referensi:
* **Phase 1 (Per-IMF Transformer-BiLSTM):** `--epochs 100`
* **Phase 2 (Fusion BiLSTM):** `--fusion_epochs 1000`
* **EMD Components:** `--imfs 12`

---

## Spesifikasi Teknis

| Komponen | Library | Keterangan |
| :--- | :--- | :--- |
| **Framework DL** | PyTorch 2.x + CUDA | Autograd, Dynamic Graph, GPU Acceleration |
| **Decomposition** | PyEMD | Empirical Mode Decomposition |
| **Preprocessing** | scikit-learn | MinMaxScaler, StandardScaler |
| **Data Handling** | Pandas + NumPy | DataFrame alignment, ndarray tensor |
| **Visualization** | Matplotlib Agg | Rendering offline / headless environment |

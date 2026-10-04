"""Generate diagram_alur_v2.docx for Yang Mulia's report."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# Styles
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

# Title
title = doc.add_heading('Sistem Prediksi AQI Jakarta — EMD–Transformer–BiLSTM + Weather', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Metadata/Subtitle
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Ekstensi Arsitektur Dong et al. (2024) • Research Question 2")
run.font.color.rgb = RGBColor(6, 182, 212) # Cyan accent
run.font.size = Pt(10)
run.italic = True

doc.add_paragraph("\n" "Tujuan: Integrasi data meteorologi exogenous (ERA5/CAMS) pada pipeline forecasting.")
doc.add_paragraph("Dataset: Open-Meteo API (historical air quality + weather), region DKI Jakarta 2022–2026.\n" "Arketipe: Ensemble Decomposition → Per-Component Deep Learning → Fusion Stage.\n")

# Section 1: Diagram Code
doc.add_heading('Diagram Alur Sistem (Mermaid Source)', level=1)
doc.add_paragraph("Salin kode berikut ke https://mermaid.live untuk menghasilkan visualisasi diagram profesional.")

code_text = """flowchart TD
    %% Subgraph: Data Acquisition
    subgraph DATA ["Fase 1: Akuisisi & Preprocessing"]
        direction TB
        AQ(("Data Air Quality<br/>CAMS Reanalysis<br/>US-AQI • PM2.5 • PM10<br/>CO • NO₂ • O₃ • SO₂")) --> MERGE("Merge & Alignment<br/>(hourly grid)")
        MET(("Data Meteorologi<br/>ERA5 Reanalysis<br/>Temp • RH • Wind<br/>Pressure • Precip<br/>Radiasi (SWR)")) --> MERGE
        
        MERGE --> IMPOUT("Imputasi Gap<br/>(interpolation limit_area)")
        IMPOUT --> WIND("Wind Encoding<br/>sin/cos circular")
        WIND --> SCALE["Feature Scaling<br/>MinMaxScaler ∈ [-1, 1]"]
    end

    %% Subgraph: Windowing
    subgraph WIN ["Fase 2: Konstruksi Tensor"]
        SCALE --> WINDOW("Sliding Window<br/>Input: 5 langkah | Horizon: 1")
        WINDOW --> SPLIT{"Chronological Split<br/>Train/Test = 75% / 25%"}
    end

    %% Subgraph: Decomposition & Per-IMF Training
    subgraph EMD_STAGE ["Fase 3: Dekomposisi & Komponen-Independen"]
        SPLIT --> DECOMP("Empirical Mode Decomposition<br/>EMD(max_imf=12)")
        DECOMP --> IMF_LIST("12 Instan Mode Fungsi (IMFs)")
        
        IMF_LIST --> TRANS_PER_IMF("Per-IMF Modeling<br/>TransAm(n_exog=<b>K</b>)")
        TRANS_PER_IMF --> TF_LSTM["Transformer Encoder<br/>+ BiLSTM Decoder<br/>+ Linear Head"]
        
        TRANS_PER_IMF -.-> LIN_PER_IMF["Baseline Fallback<br/>LinearRegression<br/>(reference safeguard)"]
        
        TF_LSTM --> COMPARE{"Pilih Best Predictor<br/>(RMSE test)"}
        LIN_PER_IMF --> COMPARE
        
        COMPARE -->|"Transformer Win"| PRED_TF("Prediksi IMF<br/>(Transformer-BiLSTM)")
        COMPARE -->|"Linear Regression Win"| PRED_LIN("Prediksi IMF<br/>(Linear Fallback)")
    end

    %% Subgraph: Fusion Stage
    subgraph FUSE ["Fase 4: Fusi Multi-Komponen"]
        PRED_TF --> FUSION_INPUT("Stacked Predictions<br/>[Pred_ΔIMF₁ … Pred_ΔIMF₁₂] + Weather")
        PRED_LIN --> FUSION_INPUT
        
        FUSION_INPUT --> LSTM_FUSE["LstmRNN (Fusion Network)<br/>Bidirectional LSTM<br/>Input: 12+K features"]
        LSTM_FUSE --> RAW_PRED("Final Prediction<br/>AQI (scaled domain)")
    end

    %% Subgraph: Evaluation
    subgraph EVAL ["Fase 5: Evaluasi Metrik"]
        RAW_PRED --> INVSCALE("Inverse Transformation<br/>(scaler.inverse_transform)")
        INVSCALE --> REAL_PRED("Predicted AQI (real-scale)")
        REAL_PRED --> METRICS("Evaluasi Statistik<br/>RMSE • MAE • MAPE")
    end

    %% Ablation Study RQ2
    subgraph ABLATION ["Ablasi Studi (RQ2 Comparison)"]
        BASE_ONLY("Base Model<br/>(AQI-only, no exog)") --> MEASURE_A["Metrics Phase 1"]
        WEATHER_ADD("Our Model<br/>(+Weather Exog)") --> MEASURE_B["Metrics Phase 2"]
        MEASURE_A --> COMP_ABALSION{"Statistical<br/>Comparison"}
        MEASURE_B --> COMP_ABALSION
    end

    %% Connections between main stages
    DATA --- WIN
    WIN --- EMD_STAGE
    EMD_STAGE --- FUSE
    FUSE --- EVAL
    
    %% Dashed connections for cross-dependencies
    MET -.-> "Weather Channels K" -.- TRANS_PER_IMF
    MET -.-> "Exogenous Features" -.- FUSION_INPUT
"""

p_code = doc.add_paragraph()
run_code = p_code.add_run(code_text)
run_code.font.name = 'Courier New'
run_code.font.size = Pt(9)
run_code.font.color.rgb = RGBColor(0, 100, 0)

# Section 2: Metodologi
doc.add_heading('Metodologi Implementasi', level=1)

methods = [
    ("1. Ingest & Preparation", "fetch_data.py", "Sumber: Open-Meteo API (CAMS reanalysis untuk air quality; ERA5 untuk meteorologi).\nOutput: CSV harian → merge menjadi jakarta_aqi.csv + jakarta_met.csv.\nValidasi: Interpolasi gap kecil via limit_area='inside', encoding arah angin sin/cos."),
    ("2. Transformasi Numerik", "data.py", "Scaling: MinMaxScaler(feature_range=(-1, 1)) pada seluruh target series (protocol sesuai referensi).\nWindowing: Sliding window input_window = 5 → horizon = 1.\nSplit: Chronological 75:25 (train:test)."),
    ("3. Dekomposisi Adaptif", "PyEMD", "Memecah signal non-stasioner AQI menjadi 12 IMF (Intrinsic Mode Functions).\nSetiap IMF mewakili frekuensi/osilasi berbeda dari pola polusi."),
    ("4. Pelatihan Independent (Per-IMF)", "model.py", "Setiap IMF dilatih secara independen melalui dua arsitektur paralel:\n\n• TransAm (Extended): Transformer Encoder → BiLSTM Decoder → Linear Output. Mendukung channel cuaca exogenous via nn.Linear(1+K, d_model).\n• Linear Regression: Regresi linier klasik sebagai sanity check & fallback (safeguard sesuai protokol referensi).\n• Seleksi Otomatis: Pilih prediktor dengan RMSE terkecil pada subset uji setiap IMF."),
    ("5. Fusion Network", "model.py", "Prediksi dari semua komponen yang telah terpilih digabungkan ke dalam satu jaringan LstmRNN (BiLSTM dengan input multi-feature) untuk menghasilkan output AQI final tunggal."),
    ("6. Evaluasi & Metrik", "-", "Inverse Scale: Kembalikan prediksi ke skala asli (unit AQI riil).\nMetric: RMSE, MAE, MAPE dihitung terhadap ground truth real-scale.\nVisualisasi: Plot comparison prediction vs truth (plot_result.py).")
]

for title, filename, desc in methods:
    p = doc.add_paragraph()
    run_t = p.add_run(f"{title} ({filename})")
    run_t.bold = True
    doc.add_paragraph(desc)

# Section 3: Tabel Ablasi
doc.add_heading('Ablasi Studi (Research Question 2)', level=1)

table = doc.add_table(rows=4, cols=4) # Removed style arg
hdr_cells = table.rows[0].cells
headers = ['Variabel', 'Eksperimen', 'Tujuan Penelitian', 'Implementasi']
for i, h in enumerate(headers):
    hdr_cells[i].text = h

data = [
    ('Base Model', 'Training tanpa fitur meteorologi (AQI-only)', 'Establish baseline performa arsitektur murni', '--epochs 100 --imfs 12'),
    ('Our Model', 'Training dengan fitur cuaca (temp, wind, pressure, dll.)', 'Mengukur kontribusi exogenous variables terhadap akurasi', '--weather --epochs 100 --imfs 12'),
    ('Statistical Test', 'Compare RMSE/MAE/MAPE antar fase', 'Validasi signifikansi penambahan weather input', 'Python stats module')
]

for row_data in data:
    row_cells = table.add_row().cells
    for i, val in enumerate(row_data):
        row_cells[i].text = val

# Section 4: Spesifikasi Teknis
doc.add_heading('Spesifikasi Teknis', level=1)
tech_table = doc.add_table(rows=6, cols=2) # Removed style arg
t_hdr = tech_table.rows[0].cells
t_hdr[0].text = "Komponen"
t_hdr[1].text = "Library/Tool"

specs = [
    ('Framework DL', 'PyTorch 2.x + CUDA'),
    ('Decomposition', 'PyEMD'),
    ('Preprocessing', 'scikit-learn'),
    ('Data Handling', 'Pandas + NumPy'),
    ('Visualization', 'Matplotlib Agg')
]

for comp, lib in specs:
    row = tech_table.add_row().cells
    row[0].text = comp
    row[1].text = lib

doc.save('diagram_alur_v2.docx')
print("DOCX saved successfully.")

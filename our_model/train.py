"""Run EMD-Transformer-BiLSTM (+ weather exogenous) on Jakarta air quality.

Pipeline (adapted from baseline_emd-transformer-bilstm.py, diperbaiki 10 Okt 2026):
  1. scale AQI, EMD -> IMFs
  2. split tiga fase kronologis 70/15/15: train / validasi / test
  3. per IMF: Transformer-BiLSTM DAN LinearRegression rujukan sama-sama dilatih
     pada IMF ITU SENDIRI (input = window IMF [+ cuaca], target = window IMF);
     pemenangnya dipilih HANYA dari blok validasi
  4. fusi BiLSTM dilatih atas prediksi out-of-sample blok validasi (+ cuaca) -> AQI,
     lalu diterapkan ke blok test (distribusi input train & test setara)
  5. metrik RMSE/MAE/MAPE pada blok test setelah inverse-scale

Yang diperbaiki di revisi ini (audit 10 Okt 2026):
  - target loop per-IMF dulu seri AQI penuh -> Transformer tidak pernah melihat
    IMF (loss mentok ±0,08295, "lin" menang 13/13)
  - seleksi tf-vs-lin dulu memakai data test (test-set selection bias)
  - fusi dulu dilatih atas IMF asli tapi diuji atas IMF prediksi (exposure bias)
Sengaja TIDAK diubah (bawaan rujukan, ditulis sebagai limitasi): MinMaxScaler +
EMD atas seluruh seri sebelum split.

Ablation for RQ2:  python train.py --weather  vs  python train.py
Colab/Kaggle:      !python our_model/train.py --weather --epochs 100
"""
import argparse
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, TensorDataset

import data as datamod
import model as modelmod

# ponytail: fraksi split dipaku di sini, bukan argumen CLI — jadikan opsi kalau
# nanti perlu sweep protokol
TRAIN_FRAC, VAL_FRAC = 0.70, 0.85


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None)
    p.add_argument("--target", default="us_aqi")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--horizon", type=int, default=1, help="steps ahead (1 = like baseline)")
    p.add_argument("--imfs", type=int, default=12)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--fusion-epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weather", action="store_true", help="enable exogenous weather channels (our model)")
    p.add_argument("--start", default=None, help="ISO date, e.g. 2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None, help="output name suffix")
    return p.parse_args()


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def rmse(a, b):
    return metrics.mean_squared_error(a, b) ** 0.5


def main():
    a = parse_args()
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from pathlib import Path

    data_dir = str(Path(a.data_dir or Path(__file__).resolve().parent.parent / "our_data"))
    df = datamod.load_merged(data_dir)
    y, w, scaler, dfk = datamod.make_series(df, a.start, a.end, a.target)

    K = w.shape[1] if a.weather else 0
    w_use = w if a.weather else np.zeros((len(w), 0), np.float32)
    tw, ow = a.window, a.horizon
    n_samp = len(y) - tw - ow + 1
    n_tr, n_va = int(round(n_samp * TRAIN_FRAC)), int(round(n_samp * VAL_FRAC))
    # indeks (koordinat y) dari target langkah terakhir tiap blok
    va_start, te_start = n_tr + tw + ow - 1, n_va + tw + ow - 1
    print(f"split: train {n_tr} | val {n_va - n_tr} | test {n_samp - n_va} windows")

    # 1. EMD on the scaled series (same protocol as reference: whole series)
    from PyEMD import EMD
    t0 = time.time()
    imfs = EMD().emd(y, max_imf=a.imfs)
    print(f"EMD: {len(imfs)} IMFs in {time.time()-t0:.0f}s")
    n_imf = len(imfs)

    def loaders(series, lo, hi):
        X, Y, _ = datamod.make_windows(series, w_use, tw, ow)
        ds = TensorDataset(torch.tensor(X[lo:hi]), torch.tensor(Y[lo:hi]))
        return DataLoader(ds, min(a.batch_size, max(1, hi - lo)), shuffle=False)

    def predict(m, ld):
        m.eval()
        with torch.no_grad():
            return torch.cat([m(bx.transpose(0, 1).to(device))[-1].view(-1).cpu() for bx, _ in ld]).numpy()

    def linear_fallback(series):
        """LinearRegression per-IMF (mekanisme proteksi rujukan), dilatih di blok train.

        Return prediksi untuk SEMUA sample setelah blok train, yaitu sample
        n_tr..n_samp (blok validasi di depan, blok test di belakang).
        """
        n = len(series) - tw - ow + 1
        Z = np.stack([series[i:i + tw] for i in range(n)])
        yy = np.array([series[i + ow: i + tw + ow] for i in range(n)])
        m = LinearRegression().fit(Z[:n_tr].reshape(n_tr, -1), yy[:n_tr])
        return m.predict(Z[n_tr:].reshape(n - n_tr, -1))[:, -1]  # langkah terakhir

    criterion = nn.MSELoss()

    # --- CHECKPOINT SYSTEM (per-arm + v2: format npz lama tidak punya pred_va) ---
    tag = a.tag or ("weather" if a.weather else "base")
    CKPT_DIR = Path(__file__).resolve().parent / "results" / "checkpoints" / f"{tag}-v2"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    preds_te = [None] * n_imf
    preds_va = [None] * n_imf

    for j in range(n_imf):
        pth = CKPT_DIR / f"imf_{j}_info.npz"
        if pth.exists():
            try:
                d = np.load(str(pth), allow_pickle=True)
                preds_te[j], preds_va[j] = d["pred"], d["pred_va"]
                print(f"[RESUMED] Loaded completed IMF {j+1} (saved mode: {d['mode']})")
            except Exception as e:
                print(f"[WARNING] Could not load IMF {j+1} checkpoint ({e}), will retrain.")
                preds_te[j] = preds_va[j] = None

    # --- PER-IMF TRAINING LOOP (target = IMF itu sendiri) ---
    for j, imf in enumerate(imfs):
        if preds_te[j] is not None:
            print(f"Skipping already trained IMF {j+1}")
            continue

        ld_tr = loaders(imf, 0, n_tr)
        ld_va = loaders(imf, n_tr, n_va)
        ld_te = loaders(imf, n_va, n_samp)

        m = modelmod.TransAm(n_exog=K).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.95)
        ep = 3 if a.smoke else a.epochs

        print(f"Training IMF {j+1}/{n_imf} (target = IMF {j+1}) ...")
        for epoch in range(1, ep + 1):
            m.train()
            tot, t_ep = 0, time.time()
            for bx, by in ld_tr:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = criterion(m(bx.transpose(0, 1)), by.transpose(0, 1))
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                tot += loss.item()
            sch.step()
            if epoch % max(1, ep // 5) == 0:
                print(f"IMF{j+1} ep{epoch} loss {tot/len(ld_tr):.5f} {time.time()-t_ep:.0f}s")

        tf_va, tf_te = predict(m, ld_va), predict(m, ld_te)
        lin = linear_fallback(imf)
        lin_va, lin_te = lin[:n_va - n_tr], lin[n_va - n_tr:]
        truth_va = imf[va_start: va_start + len(tf_va)]

        r_tf = rmse(tf_va, truth_va)
        r_lin = rmse(lin_va[:len(truth_va)], truth_va)
        # ponytail: seleksi memakai blok VALIDASI. Kalau butuh lebih ketat, ganti
        # jadi cross-fitting di blok train (butuh refit per fold).
        if r_tf <= r_lin:
            mode, va_pred, te_pred = "tf", tf_va, tf_te
        else:
            mode, va_pred, te_pred = "lin", lin_va[:len(tf_va)], lin_te[:len(tf_te)]

        print(f"IMF{j+1}: val transformer {r_tf:.4f} | linear {r_lin:.4f} -> {mode}")
        np.savez(str(CKPT_DIR / f"imf_{j}_info.npz"),
                 pred=te_pred, pred_va=va_pred,
                 truth=imf[te_start: te_start + len(te_pred)],
                 rmse_trans=float(r_tf), rmse_lin=float(r_lin), mode=mode)
        print(f"[CHECKPOINT SAVED] IMF {j+1} state saved to disk.")

        preds_te[j], preds_va[j] = te_pred, va_pred

    # 3. fusi: dilatih atas prediksi OOS blok validasi, diuji atas prediksi blok test
    assert len(preds_va[0]) == n_va - n_tr, "panjang prediksi validasi tidak sejajar"
    assert len(preds_te[0]) == n_samp - n_va, "panjang prediksi test tidak sejajar"
    arr_va = np.stack(preds_va, 1)
    arr_te = np.stack(preds_te, 1)
    fx_tr = np.concatenate([arr_va, w_use[va_start: va_start + len(arr_va)]], 1) if K else arr_va
    fx_te = np.concatenate([arr_te, w_use[te_start: te_start + len(arr_te)]], 1) if K else arr_te
    fy_tr = y[va_start: va_start + len(arr_va)].reshape(-1, 1)
    print(f"fusion training input: {fx_tr.shape} (out-of-sample validasi)")
    fx_tr = torch.tensor(fx_tr.astype(np.float32)).unsqueeze(1).to(device)
    fy_tr = torch.tensor(fy_tr.astype(np.float32)).unsqueeze(1).to(device)
    fx_te = torch.tensor(fx_te.astype(np.float32)).unsqueeze(1).to(device)

    FUSION_CKPT = CKPT_DIR / "fusion_best.pt"

    if FUSION_CKPT.exists():
        print("\n[RESUMING] Loading saved Fusion Model...")
        fusion = modelmod.LstmRNN(n_imf + K).to(device)
        fusion.load_state_dict(torch.load(FUSION_CKPT, map_location=device))
    else:
        print("\nTraining Fusion Layer from scratch...")
        fusion = modelmod.LstmRNN(n_imf + K).to(device)
        fopt = torch.optim.Adam(fusion.parameters(), lr=1e-2)
        ep = 50 if a.smoke else a.fusion_epochs
        for e in range(ep):
            out = fusion(fx_tr)
            loss = criterion(out, fy_tr)
            fopt.zero_grad(); loss.backward(); fopt.step()
            if (e + 1) % max(1, ep // 5) == 0:
                print(f"fusion ep{e+1} loss {loss.item():.5f}")
        torch.save(fusion.state_dict(), FUSION_CKPT)
        print("Fusion model saved to checkpoint.")

    fusion.eval()
    with torch.no_grad():
        pre = fusion(fx_te).cpu().numpy().flatten()

    true = y[te_start: te_start + len(pre)].reshape(-1, 1)
    pre_inv = scaler.inverse_transform(pre.reshape(-1, 1))
    true_inv = scaler.inverse_transform(true)
    rmse_v = metrics.mean_squared_error(pre_inv, true_inv) ** 0.5
    mae = metrics.mean_absolute_error(true_inv, pre_inv)
    mape = metrics.mean_absolute_percentage_error(true_inv, pre_inv)
    if a.smoke:
        tag += "_smoke"
    out_csv = Path(__file__).resolve().parent.parent / "our_model" / "results" / f"jakarta-{a.target}-{tag}-win{tw}-h{ow}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    pd.DataFrame({"pred": pre_inv.flatten(), "truth": true_inv.flatten()},
                 index=dfk.index[te_start: te_start + len(pre)]).to_csv(out_csv)
    print(f"\nRMSE {rmse_v:.4f} | MAE {mae:.4f} | MAPE {mape*100:.2f}% -> {out_csv}")

    if not a.smoke:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(pre_inv, color="red", label="Prediction")
        plt.plot(true_inv, color="blue", label="Truth")
        plt.legend(); plt.grid(True); plt.title(f"EMD-Transformer-BiLSTM ({tag}) Jakarta {a.target}")
        plt.tight_layout()
        plt.savefig(out_csv.with_suffix(".png"), dpi=150)
        print("plot saved:", out_csv.with_suffix(".png"))


if __name__ == "__main__":
    main()
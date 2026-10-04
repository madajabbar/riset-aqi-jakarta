"""Run EMD-Transformer-BiLSTM (+ weather exogenous) on Jakarta air quality.

Pipeline (faithful to baseline_emd-transformer-bilstm.py, extended with --weather):
  1. scale AQI, EMD -> IMFs
  2. per IMF: Transformer-BiLSTM on window (value [+ weather]); keep the
     reference's LinearRegression per-IMF fallback (pick whichever RMSE wins)
  3. fuse predicted IMFs (+ weather) with a BiLSTM -> AQI
  4. metrics RMSE/MAE/MAPE on inverse-scaled test split

Ablation for RQ2:  python train.py --weather  vs  python train.py
Colab:            !python our_model/train.py --weather --epochs 100
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, TensorDataset

import data as datamod
import model as modelmod


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
    X, Y, split = datamod.make_windows(y, w_use, tw, ow)

    # 1. EMD on the scaled series (same protocol as reference: whole series)
    from PyEMD import EMD
    t0 = time.time()
    imfs = EMD().emd(y, max_imf=a.imfs)
    print(f"EMD: {len(imfs)} IMFs in {time.time()-t0:.0f}s")
    n_imf = len(imfs)

    # tensors sliced by chronological split
    tr_X, tr_Y = torch.tensor(X[:split]), torch.tensor(Y[:split])
    te_X, te_Y = torch.tensor(X[split:]), torch.tensor(Y[split:])
    tr_ld = DataLoader(TensorDataset(tr_X, tr_Y), a.batch_size, shuffle=False)
    te_ld = DataLoader(TensorDataset(te_X, te_Y), 1000, shuffle=False)

    def linear_fallback(imf):
        """Reference's per-IMF LinearRegression on the window (no weather)."""
        arr = imf
        n = len(arr) - tw - ow + 1
        Z = np.stack([arr[i:i + tw] for i in range(n)])
        yy = np.array([arr[i + ow: i + tw + ow] for i in range(n)])
        m = LinearRegression().fit(Z[:split].reshape(split, -1), yy[:split])
        pred = m.predict(Z[split:].reshape(len(Z) - split, -1))
        return pred  # (n_te, tw)

    criterion = nn.MSELoss()
    te_start = split + tw + ow - 1  # absolute y-index of first test-pred target step
    preds_te = []
    for j, imf in enumerate(imfs):
        m = modelmod.TransAm(n_exog=K).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.95)
        ep = 3 if a.smoke else a.epochs
        for epoch in range(1, ep + 1):
            m.train()
            tot, t_ep = 0, time.time()
            for bx, by in tr_ld:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = criterion(m(bx.transpose(0, 1)), by.transpose(0, 1))
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                tot += loss.item()
            sch.step()
            if epoch % max(1, ep // 5) == 0:
                print(f"IMF{j+1} ep{epoch} loss {tot/len(tr_ld):.5f} {time.time()-t_ep:.0f}s")
        m.eval()
        with torch.no_grad():
            pred = torch.cat([m(bx.transpose(0, 1).to(device))[-1].view(-1).cpu() for bx, _ in te_ld]).numpy()
        # linear fallback comparison on test part of this IMF (aligned)
        lin = linear_fallback(imf)[:, -1]
        truth_te = imf[te_start: te_start + len(pred)]
        rmse_t = metrics.mean_squared_error(pred, truth_te) ** 0.5
        rmse_l = metrics.mean_squared_error(lin[:len(pred)], truth_te[:len(lin)]) ** 0.5
        chosen = pred if rmse_t <= rmse_l else lin[:len(pred)]
        print(f"IMF{j+1}: transformer {rmse_t:.4f} | linear {rmse_l:.4f} -> {'tf' if rmse_t<=rmse_l else 'lin'}")
        preds_te.append(chosen)

    # 3. fusion: train on true train-split IMFs (+weather), test on predicted IMFs
    #    (same protocol as the reference: position-aligned, no shift games)
    tr_imf_stack = np.stack(imfs, 1)[:split]
    test_imf_stack = np.stack(preds_te, 1)  # (n_te_pred, n_imf)
    fx_tr = np.concatenate([tr_imf_stack, w_use[:split]], 1) if K else tr_imf_stack
    te_w = w_use[te_start: te_start + len(test_imf_stack)] if K else None
    fx_te = np.concatenate([test_imf_stack, te_w], 1) if K else test_imf_stack
    fy_tr = y[:split].reshape(-1, 1)
    fx_tr = torch.tensor(fx_tr.astype(np.float32)).unsqueeze(1).to(device)
    fy_tr = torch.tensor(fy_tr.astype(np.float32)).unsqueeze(1).to(device)
    fx_te = torch.tensor(fx_te.astype(np.float32)).unsqueeze(1).to(device)
    fusion = modelmod.LstmRNN(n_imf + K).to(device)
    fopt = torch.optim.Adam(fusion.parameters(), lr=1e-2)
    ep = 50 if a.smoke else a.fusion_epochs
    for e in range(ep):
        out = fusion(fx_tr)
        loss = criterion(out, fy_tr)
        fopt.zero_grad(); loss.backward(); fopt.step()
        if (e + 1) % max(1, ep // 5) == 0:
            print(f"fusion ep{e+1} loss {loss.item():.5f}")
    fusion.eval()
    with torch.no_grad():
        pre = fusion(fx_te).cpu().numpy().flatten()

    true = y[te_start: te_start + len(pre)].reshape(-1, 1)
    pre_inv = scaler.inverse_transform(pre.reshape(-1, 1))
    true_inv = scaler.inverse_transform(true)
    rmse = metrics.mean_squared_error(pre_inv, true_inv) ** 0.5
    mae = metrics.mean_absolute_error(true_inv, pre_inv)
    mape = metrics.mean_absolute_percentage_error(true_inv, pre_inv)
    tag = a.tag or ("weather" if a.weather else "base")
    if a.smoke:
        tag += "_smoke"
    out_csv = Path(__file__).resolve().parent.parent / "our_model" / "results" / f"jakarta-{a.target}-{tag}-win{tw}-h{ow}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    pd.DataFrame({"pred": pre_inv.flatten(), "truth": true_inv.flatten()},
                 index=dfk.index[te_start: te_start + len(pre)]).to_csv(out_csv)
    print(f"\nRMSE {rmse:.4f} | MAE {mae:.4f} | MAPE {mape*100:.2f}% -> {out_csv}")

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

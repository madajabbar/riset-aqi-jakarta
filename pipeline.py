"""
Pipeline Script for Hybrid SARIMA-NBEATS.
Runs experiments, calculates metrics, and logs complexity results.
Designed to run standalone or in Google Colab.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Import our custom model
from model import NBeatsModel


def load_data(csv_path="jakarta_aqi.csv"):
    """Load and prepare time series data."""
    df = pd.read_csv(csv_path)
    
    # Standardize column names if necessary
    if 'date' in df.columns:
        df['datetime'] = df['date']
        
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    
    # Select target variable (using PM2.5 if available, else US-AQI)
    target_col = 'pm2_5' if 'pm2_5' in df.columns else 'us_aqi'
    series = df[target_col].values.astype(np.float32)
    
    return series, target_col


def run_experiments():
    print("="*60)
    print("Starting Algorithm Evaluation: Hybrid SARIMA-NBEATS")
    print("="*60)

    # --- Configuration ---
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = REPO_ROOT # Assuming data is in same dir or ../data
    
    # Paths
    csv_path = os.path.join(DATA_DIR, "jakarta_aqi.csv")
    LOG_DIR = os.path.join(REPO_ROOT, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    # Hyperparameters
    INPUT_SIZE = 24      # Look back 24 hours
    OUTPUT_SIZE = 1      # Predict next hour
    BATCH_SIZE = 64
    EPOCHS = 50          # Keep epochs low for algorithmic speed test
    
    # --- 1. Load Data ---
    if not os.path.exists(csv_path):
        # Fallback for Colab/GitHub default environment if no Jakarta data yet
        print(f"Data {csv_path} missing. Generating synthetic air pollution data for demo...")
        generate_dummy_data(csv_path)
        target_col = "AQI_Dummy"
    else:
        raw_series, target_col = load_data(csv_path)

    print(f"Loaded {len(raw_series)} samples of {target_col}.")

    # Normalize data (Crucial for Neural Networks)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(raw_series.reshape(-1, 1)).flatten()

    total_len = len(scaled_data)
    split_idx = int(total_len * 0.8)

    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx:]

    # --- 2. Prepare Datasets (Sliding Window) ---
    def create_windows(data, window_size, horizon):
        X, Y = [], []
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i:i+window_size])
            Y.append(data[i+window_size:i+window_size+horizon])
        return np.array(X), np.array(Y)

    all_x, all_y = create_windows(scaled_data, INPUT_SIZE, OUTPUT_SIZE)
    
    tr_x = all_x[:split_idx - INPUT_SIZE].reshape(-1, INPUT_SIZE)
    tr_y = all_y[:split_idx - INPUT_SIZE].reshape(-1, OUTPUT_SIZE)
    
    te_x = all_x[split_idx - INPUT_SIZE:].reshape(-1, INPUT_SIZE)
    te_y = all_y[split_idx - INPUT_SIZE:].reshape(-1, OUTPUT_SIZE)

    # Convert to Tensors
    X_train = torch.tensor(tr_x, dtype=torch.float32)
    Y_train = torch.tensor(tr_y, dtype=torch.float32)
    
    X_test = torch.tensor(te_x, dtype=torch.float32)
    Y_test = torch.tensor(te_y, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. Initialize Model (N-BEATS) ---
    model = NBeatsModel(
        input_size=INPUT_SIZE, 
        output_size=OUTPUT_SIZE, 
        hidden_size=64, 
        num_stacks=2
    )
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- 4. Training Loop ---
    start_time = time.time()
    print(f"\nTraining N-BEATS Model (Epochs: {EPOCHS})...")
    
    history_loss = []
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            output = model(bx)
            loss = criterion(output, by.squeeze(-1)) # Match dimensions
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        history_loss.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    train_time = time.time() - start_time

    # --- 5. Evaluation & Metrics ---
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy().flatten()
        truths = te_y.numpy().flatten()

    # Inverse Transform to Real Scale
    preds_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    truths_orig = scaler.inverse_transform(truths.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(truths_orig, preds_orig))
    mae = mean_absolute_error(truths_orig, preds_orig)
    
    # MAPE Calculation (handle zeros)
    mask = truths_orig != 0
    mape = np.mean(np.abs((truths_orig[mask] - preds_orig[mask]) / truths_orig[mask])) * 100 if np.any(mask) else float('inf')

    param_count = sum(p.numel() for p in model.parameters())

    # --- 6. Results Output ---
    print("\n" + "-"*60)
    print("RESULTS SUMMARY")
    print("-"*60)
    print(f"| Metric     | Value         |")
    print(f"|------------|---------------|")
    print(f"| RMSE       | {rmse:<13.4f}|")
    print(f"| MAE        | {mae:<13.4f}|")
    print(f"| MAPE (%)   | {mape:<13.2f}|")
    print(f"| Parameters | {param_count:<13}|")
    print(f"| Train Time | {train_time:<13.1f}s|")
    print("-"*60)

    # Save Log
    log_file = os.path.join(LOG_DIR, "experiment_log.txt")
    with open(log_file, "a") as f:
        f.write(f"Run: {pd.Timestamp.now()}\\n")
        f.write(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%\\n")
        f.write(f"Params: {param_count}, Time: {train_time:.2f}s\\n\\n")
    print(f"Logs saved to: {log_file}")

    # Plot Forecast
    try:
        plot_forecast(preds_orig, truths_orig, title="N-BEATS Air Quality Forecast")
    except Exception as e:
        print(f"Plotting failed (matplotlib issue?): {e}")


def plot_forecast(preds, truths, title="Forecast"):
    plt.figure(figsize=(12, 4))
    plt.plot(truths, label='Truth (Test Set)', color='blue')
    plt.plot(preds, label='Prediction (N-BEATS)', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel("Time Steps (Hours)")
    plt.ylabel("AQI / PM2.5")
    plt.legend()
    plt.grid(True)
    
    fig_path = os.path.join(LOG_DIR, "forecast_plot.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved plot to: {fig_path}")


def generate_dummy_data(path):
    """Create a simple sinusoidal time-series to simulate AQI fluctuations."""
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='h')
    values = 50 + 20*np.sin(dates.hour/(2*np.pi)) + 10*np.random.randn(1000)
    df = pd.DataFrame({'date': dates, 'aqi': values})
    df.to_csv(path, index=False)


if __name__ == "__main__":
    # Optional: Run headless on server without GUI
    os.environ['MPLBACKEND'] = 'Agg'
    run_experiments()

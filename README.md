# Hybrid SARIMA-NBEATS - Algorithm & Complexity Project

## Overview
Implementation of a Neural Basis Expansion Analysis (N-BEATS) model to replace BiLSTM in a hybrid air pollution prediction pipeline.
Designed to compare accuracy vs. complexity (training time, parameters) against traditional baselines.

## Files Structure
- `model.py`: PyTorch implementation of N-BEATS blocks (Generic/Trend & Seasonality).
- `pipeline.py`: Main orchestrator script for training, evaluation, and metric logging.
- `requirements.txt`: Dependencies for the environment.

## Running on Google Colab
1. Upload `jakarta_aqi.csv` (or generate dummy data via pipeline).
2. `!pip install -r requirements.txt`
3. `!python pipeline.py`

## Key Features
- **Modular Architecture**: Easy replacement of components for ablation studies.
- **Complexity Logging**: Automatically logs parameter count and training time.
- **Colab Optimized**: Includes dummy data generator if dataset is missing.

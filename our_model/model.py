"""EMD-Transformer-BiLSTM (Dong et al. 2024 reference architecture) extended
with exogenous weather inputs.

Per IMF component: TransformerEncoder -> BiLSTM -> linear, input channels =
1 (IMF value) + K (z-scored weather). The reference impl fakes the embedding
by broadcasting scalars against the positional encoding; we do the honest
Linear(1+K, d_model) instead (upgrade in-place, same shape contract).
Fusion stage: BiLSTM over [predicted IMFs, weather] -> AQI.
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransAm(nn.Module):
    """src: (seq_len, batch, 1+K) -> (seq_len, batch, 1)"""

    def __init__(self, feature_size=250, num_layers=1, dropout=0.1, n_exog=0):
        super().__init__()
        self.n_exog = n_exog
        self.embedding = nn.Linear(1 + n_exog, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.hidden_size = 2
        self.decoder = nn.LSTM(feature_size, self.hidden_size, num_layers=2, bias=True, bidirectional=True)
        self.linear1 = nn.Linear(self.hidden_size * 2, 1)
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-0.1, 0.1)
        for name, param in self.decoder.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, src):
        x = self.embedding(src) + self.pos_encoder.pe[: src.size(0), :]
        # causal mask like the reference (each step sees its window prefix)
        mask = torch.triu(torch.ones(src.size(0), src.size(0), device=src.device), diagonal=1).bool()
        x = self.transformer_encoder(x, mask=mask)
        x, _ = self.decoder(x)
        return self.linear1(x)


class LstmRNN(nn.Module):
    """Fusion: (seq_len, batch, input_size) -> (seq_len, batch, 1)"""

    def __init__(self, input_size, hidden_size=20, output_size=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True, bidirectional=True)
        self.linear1 = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear1(x)

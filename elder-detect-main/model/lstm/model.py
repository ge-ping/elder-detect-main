"""
LSTM 三分类模型 — 论文5 简化版
"""
import torch
import torch.nn as nn


class FallLSTM(nn.Module):
    """2层 LSTM + Dropout + 3分类输出

    input:  (N, 35, 29)
    output: (N, 3) logits
    """

    def __init__(
        self,
        input_dim: int = 29,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)          # (N, 35, H)
        out = out[:, -1, :]            # 取最后一帧 hidden state
        out = self.dropout(out)
        return self.fc(out)            # (N, 3)

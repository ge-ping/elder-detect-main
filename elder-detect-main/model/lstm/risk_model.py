"""
RiskRegressor — 连续 risk 回归模型（model/lstm/risk_evolution.md）

与 FallLSTM 共享同款 2 层 LSTM 骨干，仅把输出头从
「Linear(64,3)+softmax（3 类）」换成「Linear(64,1)+sigmoid（单连续 risk）」。

    input:  (N, 35, 29)
    output: (N, 1) ∈ [0,1]   ← 直接输出跌倒风险，取代原概率合成公式

可选 RiskMultiTask（本期 MULTITASK=False 不开）：
    共享骨干 + 分类头(CE) + 风险头(加权MSE)，推理时同时给类别与连续 risk。
"""
import torch
import torch.nn as nn


class RiskRegressor(nn.Module):
    """2 层 LSTM + Dropout + 1 维 sigmoid 风险回归头

    input:  (N, W, input_dim)
    output: (N, 1) 经 sigmoid ∈ [0,1]
    """

    def __init__(
        self,
        input_dim: int = 29,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)            # (N, W, H)
        out = out[:, -1, :]                  # 取最后一帧 hidden state
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out))   # (N, 1) ∈ [0,1]


class RiskMultiTask(nn.Module):
    """可选：共享 LSTM 骨干 + 分类头 + 风险头（联合训练）

    仅在 MULTITASK=True 时使用，本期默认关闭。
        head_cls  : Linear(64,3) + softmax  → 分类（CE 监督 labels）
        head_risk : Linear(64,1) + sigmoid → 风险（加权MSE 监督 risk）
        Loss = λ·CE + (1−λ)·weighted_mse
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
        self.head_cls = nn.Linear(hidden_dim, num_classes)
        self.head_risk = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        logits = self.head_cls(out)                 # (N, 3)
        risk = torch.sigmoid(self.head_risk(out))     # (N, 1)
        return logits, risk

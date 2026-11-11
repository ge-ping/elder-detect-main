"""
model/stgcn/stgcn.py — ST-GCN（Spatial-Temporal Graph Convolutional Network）

独立模块，替换 LSTM 作为 B 模型（跌倒风险回归）。
输入：(N, C=3, T, V)  — C=3 (x,y,z), T=窗口帧数, V=关节数(17)
输出：(N, 1) ∈ [0,1] — 连续跌倒风险（与 RiskRegressor 同接口，demo 门控一行不改）

设计对齐经典 ST-GCN (Yan et al. 2018)：
  - 每个 STGCNBlock = 空间图卷积（按 A 的 3 个分区聚合）+ 时序卷积 + BN + 残差
  - 多层后全局均值池化 → 回归头(sigmoid)

load_stgcn(path) 工厂读 model_info.json 构造网络并载入权重，供 demo 自动切换。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.stgcn.graph import (
    FALL_JOINTS_17, FALL_EDGES, FALL_JOINT_SET_NAME,
    COCO13_JOINTS, COCO13_EDGES, COCO13_JOINT_SET_NAME,
    build_spatial_graph, NUM_FALL_JOINTS, select_joint_set,
)


class STGCNBlock(nn.Module):
    """单个 ST-GCN 块：空间图卷积 + 时序卷积 + 残差。"""

    def __init__(self, in_c: int, out_c: int, A: np.ndarray,
                 temp_kernel: int = 9, stride: int = 1, dropout: float = 0.3,
                 residual: bool = True):
        super().__init__()
        self.K = A.shape[0]  # 分区数（=3: self/in/out）
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        # 每个分区一个 1x1 通道映射
        self.conv_a = nn.ModuleList(
            [nn.Conv2d(in_c, out_c, 1, bias=False) for _ in range(self.K)]
        )
        # 残差连接
        if not residual:
            self.residual = lambda x: 0
        elif in_c == out_c and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_c),
            )
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        # 时序卷积（沿 T 维），kernel 须为奇数
        pad = (temp_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (temp_kernel, 1), (stride, 1), (pad, 0)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout, inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        res = self.residual(x)
        agg = None
        for k in range(self.K):
            # 空间聚合：对 V 维做邻接加权求和，再通道映射
            x_agg = torch.einsum("nctv,vw->nctw", x, self.A[k])  # (N,C,T,V)
            x_agg = self.conv_a[k](x_agg)                        # (N,out_c,T,V)
            agg = x_agg if agg is None else agg + x_agg
        x = self.bn(agg) + res
        x = self.relu(x)
        x = self.tcn(x)
        return x


class STGCN(nn.Module):
    """ST-GCN 跌倒风险回归网络。

    input:  (N, in_channels=3, T=window, V=num_joints)
    output: (N, 1) sigmoid ∈ [0,1]
    """

    def __init__(self, num_joints: int = NUM_FALL_JOINTS, in_channels: int = 3,
                 layers: Optional[list[int]] = None, window: int = 35,
                 temp_kernel: int = 9, dropout: float = 0.3,
                 joint_set: Optional[str] = None):
        super().__init__()
        if layers is None:
            layers = [64, 64, 128, 128, 256]
        # 按关节集选图（fall17 / coco13），保证邻接 V 维与数据一致
        joints, edges, self.joint_set = select_joint_set(num_joints, joint_set)
        assert len(joints) == num_joints, (
            f"关节集 {self.joint_set} 节点数 {len(joints)} "
            f"与 num_joints={num_joints} 不一致")
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.window = window
        self.layers = list(layers)
        self.temp_kernel = temp_kernel
        self.dropout_p = dropout

        A = build_spatial_graph(joints, edges)
        # 输入投影（首块，无残差）
        self.input_block = STGCNBlock(
            in_channels, layers[0], A, temp_kernel, 1, dropout, residual=False)
        # 中间块
        self.blocks = nn.ModuleList(
            [STGCNBlock(layers[i], layers[i + 1], A, temp_kernel, 1, dropout)
             for i in range(len(layers) - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        x = self.input_block(x)
        for b in self.blocks:
            x = b(x)
        # 全局均值池化（T 与 V）
        x = x.mean(dim=-1).mean(dim=-1)   # (N, C_out)
        x = self.dropout(x)
        x = self.fc(x)                    # (N, 1)
        return self.sigmoid(x)            # (N, 1) ∈ [0,1]


def load_stgcn(path: str) -> STGCN:
    """读 model_info.json 构造网络并载入权重。供 demo 自动切换。"""
    p = Path(path)
    info_path = p.parent / "model_info.json"
    info = {}
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    model = STGCN(
        num_joints=info.get("num_joints", NUM_FALL_JOINTS),
        in_channels=info.get("input_channel", 3),
        layers=info.get("layers", [64, 64, 128, 128, 256]),
        window=info.get("window_size", 35),
        temp_kernel=info.get("temp_kernel", 9),
        dropout=info.get("dropout", 0.3),
        joint_set=info.get("joint_set"),
    )
    state = torch.load(str(p), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

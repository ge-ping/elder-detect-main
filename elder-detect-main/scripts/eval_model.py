#!/usr/bin/env python
"""
eval_model.py — 加载已训练模型，生成 test_report.txt + 混淆矩阵图

用法:
    python scripts/eval_model.py model/lstm/p1_smooth/s005/run_01
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.lstm.model import FallLSTM

DATA_DIR = PROJECT_ROOT / "datasets/experiment"
DATASETS = ["Le2i", "UR-FALL", "CAUCAFall", "UP-FALL", "Pre-VFall"]
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_test_data(data_dir: Path):
    Xl, yl = [], []
    for name in DATASETS:
        d = data_dir / name
        xp, yp = d / "X_test.npy", d / "y_test.npy"
        if xp.exists():
            Xl.append(torch.from_numpy(np.load(xp)).float())
            yl.append(torch.from_numpy(np.load(yp)).long())
    return torch.cat(Xl, dim=0), torch.cat(yl, dim=0)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/eval_model.py <run_dir> [data_dir]")
        return
    run_dir = Path(sys.argv[1])
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else (DATA_DIR / "v5_w25")

    # 读 config
    cfg = {}
    cfg_path = run_dir / "config.txt"
    if cfg_path.exists():
        for line in cfg_path.read_text().splitlines():
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip()

    input_dim = int(cfg.get("input_dim", 29))
    print(f"Model: {run_dir}, input_dim={input_dim}")
    print(f"Data: {data_dir}")

    # 加载数据
    X_test, y_test = load_test_data(data_dir)
    print(f"Test samples: {X_test.shape[0]}")

    # 加载模型
    model = FallLSTM(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(str(run_dir / "best_model.pt"), map_location=DEVICE, weights_only=True))
    model.eval()

    # 推理
    test_dl = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_dl:
            Xb = Xb.to(DEVICE)
            all_preds.append(model(Xb).argmax(1).cpu())
            all_labels.append(yb)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    # 分类报告
    try:
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        test_f1 = f1_score(y_true, y_pred, average="macro")
        report = classification_report(y_true, y_pred,
                                       target_names=["Normal", "Unstable", "Fall"], digits=4)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\nTest macro F1: {test_f1:.4f}")
        print(report)
        print("Confusion Matrix:")
        print(f"{'':>10} {'Pred N':>8} {'Pred U':>8} {'Pred F':>8}")
        for i, name in enumerate(["True N", "True U", "True F"]):
            print(f"{name:>10} {cm[i,0]:8d} {cm[i,1]:8d} {cm[i,2]:8d}")

        (run_dir / "test_report.txt").write_text(
            f"Test macro F1: {test_f1:.4f}\n\n{report}\nConfusion Matrix:\n"
            + f"{'':>10} {'Pred N':>8} {'Pred U':>8} {'Pred F':>8}\n"
            + "\n".join(f"{n:>10} {cm[i,0]:8d} {cm[i,1]:8d} {cm[i,2]:8d}"
                        for i, n in enumerate(["True N", "True U", "True F"])))
        print(f"\nSaved: {run_dir}/test_report.txt")
    except ImportError:
        print("(sklearn not available)")

    # 画图
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        try:
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Normal","Unstable","Fall"],
                        yticklabels=["Normal","Unstable","Fall"], ax=ax)
        except ImportError:
            ax.imshow(cm, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=14,
                            color="white" if cm[i,j]>cm.max()/2 else "black")
            ax.set_xticks([0,1,2]); ax.set_xticklabels(["N","U","F"])
            ax.set_yticks([0,1,2]); ax.set_yticklabels(["N","U","F"])
        ax.set_title(f"Confusion Matrix — Macro F1={test_f1:.3f}")
        fig.tight_layout()
        fig.savefig(str(run_dir / "test_cm.png"), dpi=120)
        print(f"Saved: {run_dir}/test_cm.png")
    except ImportError:
        print("(matplotlib not available)")


if __name__ == "__main__":
    main()

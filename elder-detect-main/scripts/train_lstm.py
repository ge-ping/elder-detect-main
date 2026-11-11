#!/usr/bin/env python
"""
train_lstm.py — 合并四个数据集，加权 CrossEntropyLoss 训练 LSTM

用法:
    python scripts/train_lstm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.lstm.model import FallLSTM


# ==================================================================
# 配置
# ==================================================================

DATASETS   = ["Le2i", "UR-FALL", "CAUCAFall", "UP-FALL", "Pre-VFall"]
EPOCHS     = 100
BATCH_SIZE = 64
LR         = 1e-3
VAL_RATIO  = 0.15
PATIENCE   = 15
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================================
# 加载数据
# ==================================================================

def load_datasets(data_dir: Path, dataset_names: list[str]) -> tuple[torch.Tensor, ...]:
    X_train_list, y_train_list = [], []
    X_test_list,  y_test_list  = [], []

    for name in dataset_names:
        d = data_dir / name
        for split, xl, yl in [("train", X_train_list, y_train_list),
                               ("test",  X_test_list,  y_test_list)]:
            xp, yp = d / f"X_{split}.npy", d / f"y_{split}.npy"
            if xp.exists() and yp.exists():
                X = np.load(xp)
                y = np.load(yp)
                xl.append(torch.from_numpy(X).float())
                yl.append(torch.from_numpy(y).long())
                print(f"  {name} {split}: {X.shape}")

    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    X_test  = torch.cat(X_test_list,  dim=0)
    y_test  = torch.cat(y_test_list,  dim=0)

    return X_train, y_train, X_test, y_test


# ==================================================================
# 训练
# ==================================================================

def compute_class_weights(y: torch.Tensor, scheme: str) -> torch.Tensor:
    """类别权重

    scheme:
      "sqrt"     — sqrt(N/ni)  温和版
      "balanced" — N/ni         激进版
      "none"     — [1,1,1]     无加权
      "1,5,2"    — 自定义 (逗号分隔)
    """
    counts = torch.bincount(y)
    total   = y.shape[0]
    print(f"  Class counts: {counts.tolist()}")

    if "," in scheme:
        weights = torch.tensor([float(x) for x in scheme.split(",")])
    elif scheme == "sqrt":
        weights = torch.sqrt(total / counts.float())
    elif scheme == "balanced":
        weights = total / counts.float()
    elif scheme == "none":
        weights = torch.ones(3)
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")

    print(f"  Class weights ({scheme}): {weights.tolist()}")
    return weights


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--version", "-v", type=str, default=None,
                   help="快捷: v1|v2|v2.2|v3 (自动设置 dim+data_dir+run_dir)")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--input_dim", type=int, default=None)
    p.add_argument("--weight_scheme", type=str, default="1,5,3",
                   help="类别权重: sqrt|balanced|none|1,5,2 (自定义)")
    p.add_argument("--f1_weights", type=str, default="1,1,1",
                   help="F1 加权权重: w_N,w_U,w_F (默认 1,1,1 即 macro)")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing (0=off, 建议 0.05)")
    args = p.parse_args()

    # --version 快捷映射
    VERSION_MAP = {
        "v1":   (29, "datasets/experiment/v1",  "model/lstm/v1"),
        "v2":   (55, "datasets/experiment/v2",  "model/lstm/v2"),
        "v2.2": (55, "datasets/experiment/v2_2","model/lstm/v2_2"),
        "v3":   (81, "datasets/experiment/v3",  "model/lstm/v3"),
        "v4":   (36, "datasets/experiment/v4",  "model/lstm/v4"),
    }
    if args.version:
        v = args.version.lower()
        if v not in VERSION_MAP:
            print(f"Unknown version: {v}. Choose: {list(VERSION_MAP.keys())}")
            return
        dim, ddir, rdir = VERSION_MAP[v]
        args.input_dim = args.input_dim or dim
        args.data_dir  = args.data_dir  or ddir
        args.run_dir   = args.run_dir   or rdir

    data_dir  = Path(args.data_dir or str(PROJECT_ROOT / "datasets"))
    input_dim = args.input_dim or 29
    f1_w = [float(x) for x in args.f1_weights.split(",")]

    print(f"Device: {DEVICE}")
    print(f"Data: {data_dir}, input_dim={input_dim}\n")
    print("Loading datasets...")
    X_train_all, y_train_all, X_test, y_test = load_datasets(data_dir, DATASETS)

    # ---- 分验证集 ----
    total_train = X_train_all.shape[0]
    val_size    = int(total_train * VAL_RATIO)
    train_size  = total_train - val_size
    train_ds_full = TensorDataset(X_train_all, y_train_all)
    train_ds, val_ds = random_split(
        train_ds_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"\n  Train: {train_size}, Val: {val_size}, Test: {X_test.shape[0]}")

    # 类别权重（仅基于训练集）
    y_train_only = torch.tensor([train_ds_full[i][1].item() for i in train_ds.indices])
    print("\nClass weights (on train only):")
    class_weights = compute_class_weights(y_train_only, args.weight_scheme).to(DEVICE)

    # ---- 自动递增 run 目录 ----
    base_dir = Path(args.run_dir) if args.run_dir else (PROJECT_ROOT / "model" / "lstm")
    run_id = 1
    while (base_dir / f"run_{run_id:02d}").exists():
        run_id += 1
    save_dir = base_dir / f"run_{run_id:02d}"
    save_dir.mkdir(parents=True)
    print(f"\nRun dir: {save_dir}")

    # ---- DataLoader ----
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # ---- 模型 ----
    model = FallLSTM(input_dim=input_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5,
    )

    # ---- 超参记录 ----
    with open(save_dir / "config.txt", "w") as f:
        f.write(f"input_dim={input_dim}\nepochs={EPOCHS}\nbatch_size={BATCH_SIZE}\nlr={LR}\n")
        f.write(f"val_ratio={VAL_RATIO}\npatience={PATIENCE}\n")
        f.write(f"datasets={DATASETS}\n")
        f.write(f"weight_scheme={args.weight_scheme}\nf1_weights={args.f1_weights}\n")
        f.write(f"label_smoothing={args.label_smoothing}\n")
        f.write(f"class_weights={class_weights.tolist()}\n")
        f.write(f"device={DEVICE}\n")

    # ---- 训练 ----
    print(f"\nTraining (patience={PATIENCE}, F1 weights={args.f1_weights})...")
    print(f"{'Epoch':>5} {'Train':>8} {'Val':>8} {'U_P':>6} {'U_R':>6} {'U_F1':>6} {'wF1':>6} {'LR':>8}")
    print(f"{'':>5} {'Loss':>8} {'Loss':>8} {'':>6} {'':>6} {'':>6} {'':>6} {'':>8}")
    print("-" * 58)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * Xb.size(0)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total += Xb.size(0)

        # ---- Val ----
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        v_preds, v_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * Xb.size(0)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += Xb.size(0)
                v_preds.append(logits.argmax(1).cpu())
                v_labels.append(yb.cpu())

        tl, ta = t_loss / t_total, t_correct / t_total
        vl, va = v_loss / v_total, v_correct / v_total
        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(vl);   history["val_acc"].append(va)

        # 加权 F1（默认 1:1:1 = macro）
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
            yv_true = torch.cat(v_labels).numpy()
            yv_pred = torch.cat(v_preds).numpy()
            per_class_f1 = f1_score(yv_true, yv_pred, average=None, zero_division=0)
            # 加权: w_N*f1_N + w_U*f1_U + w_F*f1_F
            w_sum = sum(f1_w)
            v_f1  = sum(w * f for w, f in zip(f1_w, per_class_f1)) / w_sum
            u_p   = precision_score(yv_true, yv_pred, labels=[1], average=None, zero_division=0)[0]
            u_r   = recall_score(yv_true, yv_pred, labels=[1], average=None, zero_division=0)[0]
            u_f1  = per_class_f1[1]
            f1_str = f"N={per_class_f1[0]:.3f} U={per_class_f1[1]:.3f} F={per_class_f1[2]:.3f}"
        except ImportError:
            v_f1 = va; u_p = 0; u_r = 0; u_f1 = 0; f1_str = "N/A"

        scheduler.step(v_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:5d} {tl:8.4f} {vl:8.4f} {u_p:6.4f} {u_r:6.4f} {u_f1:6.4f} {v_f1:6.4f} {current_lr:8.6f}")

        # ---- 保存最佳 ----
        if v_f1 > best_f1:
            best_f1 = v_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), str(save_dir / "best_model.pt"))
        else:
            no_improve += 1

        # ---- Early stopping ----
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    print(f"\nBest: epoch {best_epoch}, val wF1 = {best_f1:.4f}", flush=True)

    # 保存元信息供 demo 自动适配
    import json
    (save_dir / "model_info.json").write_text(json.dumps({
        "input_dim": input_dim,
        "window_size": X_train_all.shape[1] if X_train_all.shape[1] > 0 else None,
        "version": args.version or f"custom_dim{input_dim}",
        "weight_scheme": args.weight_scheme,
        "f1_weights": args.f1_weights,
        "label_smoothing": args.label_smoothing,
    }, indent=2))

    # ---- 最终 Test 评估 ----
    model.load_state_dict(torch.load(str(save_dir / "best_model.pt"), map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_dl:
            Xb = Xb.to(DEVICE)
            all_preds.append(model(Xb).argmax(1).cpu())
            all_labels.append(yb)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    try:
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        test_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"\nTest macro F1: {test_f1:.4f}")
        print("\n" + "=" * 50)
        print("Classification Report (best model — Test set)")
        print("=" * 50)
        print(classification_report(y_true, y_pred,
              target_names=["Normal", "Unstable", "Fall"], digits=4))

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(f"{'':>10} {'Pred N':>8} {'Pred U':>8} {'Pred F':>8}")
        for i, name in enumerate(["True N", "True U", "True F"]):
            print(f"{name:>10} {cm[i,0]:8d} {cm[i,1]:8d} {cm[i,2]:8d}")

        with open(save_dir / "test_report.txt", "w") as f:
            f.write(f"Best epoch: {best_epoch}\nTest macro F1: {test_f1:.4f}\n\n")
            f.write(classification_report(y_true, y_pred,
                    target_names=["Normal", "Unstable", "Fall"], digits=4))
            f.write("\nConfusion Matrix:\n")
            f.write(f"{'':>10} {'Pred N':>8} {'Pred U':>8} {'Pred F':>8}\n")
            for i, name in enumerate(["True N", "True U", "True F"]):
                f.write(f"{name:>10} {cm[i,0]:8d} {cm[i,1]:8d} {cm[i,2]:8d}\n")
    except ImportError:
        print("(scikit-learn not available)")

    # ---- 画图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(18, 5))

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(history["train_loss"], label="Train")
        ax1.plot(history["val_loss"], label="Val")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(history["train_acc"], label="Train")
        ax2.plot(history["val_acc"], label="Val")
        ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(1, 3, 3)
        try:
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Normal","Unstable","Fall"],
                        yticklabels=["Normal","Unstable","Fall"], ax=ax3)
            ax3.set_xlabel("Predicted"); ax3.set_ylabel("True")
        except ImportError:
            ax3.imshow(cm, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    ax3.text(j, i, str(cm[i,j]), ha="center", va="center",
                             fontsize=14, color="white" if cm[i,j]>cm.max()/2 else "black")
            ax3.set_xticks([0,1,2]); ax3.set_xticklabels(["N","U","F"])
            ax3.set_yticks([0,1,2]); ax3.set_yticklabels(["N","U","F"])
        ax3.set_title(f"Confusion Matrix — Test F1={test_f1:.3f}")

        fig.suptitle(f"FallLSTM — Best epoch {best_epoch}, val F1 {best_f1:.4f}")
        fig.tight_layout()
        fig.savefig(str(save_dir / "training_curve.png"), dpi=120, bbox_inches="tight")
        print(f"\nSaved: {save_dir}/training_curve.png")
    except ImportError:
        print("(matplotlib not available)")

    print("\nDone.")


if __name__ == "__main__":
    main()

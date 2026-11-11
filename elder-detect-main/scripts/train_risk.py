#!/usr/bin/env python
"""
train_risk.py — 合并数据集，加权 MSE 训练 RiskRegressor

标签 = 连续 risk∈[0,1]（来自 datasets/experiment/risk_w35/ 的 y_risk_*.npy）。
Loss（默认 class_weighted，对齐旧分类类权 [1,5,3]）：
        mean( w(y) · (pred − y)² )，其中
        y<0.3→Normal(w_n=1) ; 0.3≤y<0.6→Unstable(w_u=5,最高) ; y≥0.6→Fall(w_f=3)
        旧方案 weighted_mse=(1+β·y) 为单调加权，把 Fall 拉到 4x、Unstable 中段不足
        （Run02 Unstable F1 卡 0.40），故 Run03 起改用类加权。
        仍可用 --loss weighted_mse 复现 Run01/02。

用法:
    python scripts/train_risk.py --data_dir datasets/experiment/risk_w35 --input_dim 29
    python scripts/train_risk.py --run_dir model/lstm/risk_v1/run_01
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.lstm.risk_model import RiskRegressor
from scripts.build_config import (
    RISK_EPOCHS, RISK_BATCH_SIZE, RISK_LR, RISK_VAL_RATIO,
    RISK_PATIENCE, RISK_RUN_ROOT, REG_BETA, REG_LOSS_MODE,
    REG_W_NORMAL, REG_W_UNSTABLE, REG_W_FALL,
    REG_MIDBAND_CENTER, REG_MIDBAND_WIDTH, REG_MIDBAND_PEAK,
    RAMP_LO, RAMP_HI, TAU_U, TAU_ABN, RISK_MODE,
)

DATASETS = ["Le2i", "UR-FALL", "CAUCAFall", "UP-FALL", "Pre-VFall"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================================
# 加载数据（回归：读 y_risk，float）
# ==================================================================

def load_datasets(data_dir: Path, dataset_names: list[str]) -> tuple[torch.Tensor, ...]:
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for name in dataset_names:
        d = data_dir / name
        for split, xl, yl in [("train", X_train_list, y_train_list),
                               ("test", X_test_list, y_test_list)]:
            xp = d / f"X_{split}.npy"
            yp = d / f"y_risk_{split}.npy"
            if xp.exists() and yp.exists():
                X = np.load(xp)
                y = np.load(yp).astype(np.float32)
                xl.append(torch.from_numpy(X).float())
                yl.append(torch.from_numpy(y).float().unsqueeze(1))
                print(f"  {name} {split}: X{X.shape}, y_risk{y.shape}")

    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    X_test = torch.cat(X_test_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)
    return X_train, y_train, X_test, y_test


# ==================================================================
# 损失函数
# ==================================================================

# 派生分类阈值（与 eval_risk.py / demo 保持一致）
THR_WARN = 0.3
THR_DANGER = 0.6


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, beta: float = REG_BETA):
    """旧方案：mean( (1 + β·y) · (pred − y)² )，随 y 单调加权，Fall 权重最高"""
    weight = 1.0 + beta * target
    return torch.mean(weight * (pred - target) ** 2)


def class_weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                        w_n: float = REG_W_NORMAL, w_u: float = REG_W_UNSTABLE,
                        w_f: float = REG_W_FALL,
                        thr_lo: float = THR_WARN, thr_hi: float = THR_DANGER):
    """类加权 MSE，对齐旧分类类权 [1,5,3]：
       y<0.3→Normal(w_n) ; 0.3≤y<0.6→Unstable(w_u, 最高) ; y≥0.6→Fall(w_f)
       解决旧 (1+β·y) 单调加权把 Fall 拉到 4x、Unstable 中段监督不足的问题
       （Run02 中段 [0.3,0.6) MAE≈0.19 最差，Unstable F1 卡 0.40）。
    """
    w = torch.ones_like(target) * w_n
    w[(target >= thr_lo) & (target < thr_hi)] = w_u
    w[target >= thr_hi] = w_f
    return torch.mean(w * (pred - target) ** 2)


def midband_mse(pred: torch.Tensor, target: torch.Tensor,
                 center: float = REG_MIDBAND_CENTER,
                 width: float = REG_MIDBAND_WIDTH,
                 peak: float = REG_MIDBAND_PEAK):
    """连续中段加权 MSE（Run04 起）：
       权重在 Unstable 中段 y≈center 处高斯达峰，直接针对 band [0.3,0.6)
       MAE 最差处强化监督，比 flat w_u=5 横跨整段更精准。
       weight = 1 + peak·exp(-((y-center)/width)²)
         y=center     → 1+peak   (≈7)
         y=0.3/0.6   → 1+peak·exp(-((0.15)/0.18)²)≈1+0.43·peak (~3.6)
         y=0 /0.7/1  → ≈1 (几乎无额外加权)
    """
    bump = torch.exp(-((target - center) / width) ** 2)
    weight = 1.0 + peak * bump
    return torch.mean(weight * (pred - target) ** 2)


def get_criterion(args) -> callable:
    """按 args.loss 返回 (pred, target)→scalar 的闭包，统一训练/验证/Test 调用。"""
    if args.loss == "weighted_mse":
        return lambda p, t: weighted_mse(p, t, beta=args.beta)
    if args.loss == "midband":
        return lambda p, t: midband_mse(
            p, t, center=args.midband_center,
            width=args.midband_width, peak=args.midband_peak)
    # 默认 class_weighted
    return lambda p, t: class_weighted_mse(
        p, t, w_n=args.w_normal, w_u=args.w_unstable, w_f=args.w_fall)


# ==================================================================
# 训练
# ==================================================================

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--input_dim", type=int, default=None)
    p.add_argument("--epochs", type=int, default=RISK_EPOCHS)
    p.add_argument("--batch_size", type=int, default=RISK_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=RISK_LR)
    p.add_argument("--val_ratio", type=float, default=RISK_VAL_RATIO)
    p.add_argument("--patience", type=int, default=RISK_PATIENCE)
    p.add_argument("--beta", type=float, default=REG_BETA,
                   help="(旧) weighted_mse 权重 (1+β·y)")
    p.add_argument("--loss", type=str, default=REG_LOSS_MODE,
                   choices=["class_weighted", "weighted_mse", "midband"],
                   help="损失: class_weighted=对齐旧类权[1,5,3]; "
                        "weighted_mse=旧(1+β·y); midband=连续中段加权(Run04)")
    p.add_argument("--w_normal", type=float, default=REG_W_NORMAL)
    p.add_argument("--w_unstable", type=float, default=REG_W_UNSTABLE)
    p.add_argument("--w_fall", type=float, default=REG_W_FALL)
    p.add_argument("--midband_center", type=float, default=REG_MIDBAND_CENTER)
    p.add_argument("--midband_width", type=float, default=REG_MIDBAND_WIDTH)
    p.add_argument("--midband_peak", type=float, default=REG_MIDBAND_PEAK)
    args = p.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else (PROJECT_ROOT / "datasets" / "experiment" / "risk_w35")
    input_dim = args.input_dim or 29

    print(f"Device: {DEVICE}")
    print(f"Data: {data_dir}, input_dim={input_dim}, loss={args.loss}"
          f"{f', beta={args.beta}' if args.loss=='weighted_mse' else ''}\n")
    print("Loading datasets...")
    X_train_all, y_train_all, X_test, y_test = load_datasets(data_dir, DATASETS)

    # ---- 分验证集 ----
    total_train = X_train_all.shape[0]
    val_size = int(total_train * args.val_ratio)
    train_size = total_train - val_size
    train_ds_full = TensorDataset(X_train_all, y_train_all)
    train_ds, val_ds = random_split(
        train_ds_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"\n  Train: {train_size}, Val: {val_size}, Test: {X_test.shape[0]}")

    # ---- 自动递增 run 目录 ----
    base_dir = Path(args.run_dir) if args.run_dir else (PROJECT_ROOT / RISK_RUN_ROOT)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = 1
    while (base_dir / f"run_{run_id:02d}").exists():
        run_id += 1
    save_dir = base_dir / f"run_{run_id:02d}"
    save_dir.mkdir(parents=True)
    print(f"\nRun dir: {save_dir}")

    # ---- DataLoader ----
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)
    test_dl = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size)

    # ---- 模型 ----
    model = RiskRegressor(input_dim=input_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: RiskRegressor, params={n_params}")
    criterion = get_criterion(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    # ---- 超参记录 ----
    with open(save_dir / "config.txt", "w") as f:
        f.write(f"task=regression\n")
        f.write(f"input_dim={input_dim}\nepochs={args.epochs}\nbatch_size={args.batch_size}\n")
        f.write(f"lr={args.lr}\nval_ratio={args.val_ratio}\npatience={args.patience}\n")
        f.write(f"loss={args.loss}\n")
        if args.loss == "weighted_mse":
            f.write(f"beta={args.beta}\n")
        elif args.loss == "midband":
            f.write(f"midband_center={args.midband_center}\n"
                    f"midband_width={args.midband_width}\n"
                    f"midband_peak={args.midband_peak}\n")
        else:
            f.write(f"w_normal={args.w_normal}\nw_unstable={args.w_unstable}\nw_fall={args.w_fall}\n")
        f.write(f"risk_mode={RISK_MODE}\nramp_lo={RAMP_LO}\nramp_hi={RAMP_HI}\n")
        f.write(f"tau_u={TAU_U}\ntau_abn={TAU_ABN}\n")
        f.write(f"datasets={DATASETS}\ndevice={DEVICE}\n")

    # ---- 训练 ----
    print(f"\nTraining (patience={args.patience}, loss={args.loss})...")
    print(f"{'Epoch':>5} {'Train':>9} {'Val':>9} {'ValR²':>7} {'LR':>9}")
    print("-" * 42)

    history = {"train_loss": [], "val_loss": [], "val_r2": []}
    best_mse = float("inf")
    best_epoch = 0
    no_improve = 0

    # 验证集 R² 计算
    def r2_score(yt, yp):
        yt = yt.view(-1).cpu().numpy()
        yp = yp.view(-1).cpu().numpy()
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        t_loss, t_total = 0.0, 0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * Xb.size(0)
            t_total += Xb.size(0)

        # ---- Val ----
        model.eval()
        v_loss, v_total = 0.0, 0
        v_preds, v_trues = [], []
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb)
                loss = criterion(pred, yb)
                v_loss += loss.item() * Xb.size(0)
                v_total += Xb.size(0)
                v_preds.append(pred.cpu())
                v_trues.append(yb.cpu())

        tl = t_loss / t_total
        vl = v_loss / v_total
        vr2 = r2_score(torch.cat(v_trues), torch.cat(v_preds))
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_r2"].append(vr2)

        scheduler.step(vl)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:5d} {tl:9.4f} {vl:9.4f} {vr2:7.4f} {current_lr:9.6f}")

        # ---- 保存最佳（val MSE 最小）----
        if vl < best_mse:
            best_mse = vl
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), str(save_dir / "best_model.pt"))
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\nBest: epoch {best_epoch}, val MSE = {best_mse:.4f}", flush=True)

    # 保存元信息供 demo 自动适配
    (save_dir / "model_info.json").write_text(json.dumps({
        "task": "regression",
        "input_dim": input_dim,
        "window_size": int(X_train_all.shape[1]) if X_train_all.shape[1] > 0 else None,
        "risk_mode": RISK_MODE,
        "ramp_lo": RAMP_LO,
        "ramp_hi": RAMP_HI,
        "tau_u": TAU_U,
        "tau_abn": TAU_ABN,
        "loss": args.loss,
        "beta": args.beta,
        "midband_center": args.midband_center,
        "midband_width": args.midband_width,
        "midband_peak": args.midband_peak,
        "w_normal": args.w_normal,
        "w_unstable": args.w_unstable,
        "w_fall": args.w_fall,
        "multitask": False,
    }, indent=2))

    # ---- 最终 Test 评估 ----
    model.load_state_dict(torch.load(str(save_dir / "best_model.pt"), map_location=DEVICE))
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for Xb, yb in test_dl:
            Xb = Xb.to(DEVICE)
            all_preds.append(model(Xb).cpu())
            all_trues.append(yb)
    y_pred = torch.cat(all_preds).view(-1).numpy()
    y_true = torch.cat(all_trues).view(-1).numpy()

    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"\nTest MSE:  {mse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²:   {r2:.4f}")

    # ---- 画图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_loss"], label="Train")
        axes[0].plot(history["val_loss"], label="Val")
        axes[0].set_title("Weighted MSE Loss")
        axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(history["val_r2"], label="Val R²", color="green")
        axes[1].set_title("Val R²")
        axes[1].set_xlabel("Epoch"); axes[1].grid(True, alpha=0.3)
        fig.suptitle(f"RiskRegressor — Best epoch {best_epoch}, val MSE {best_mse:.4f}")
        fig.tight_layout()
        fig.savefig(str(save_dir / "training_curve.png"), dpi=120, bbox_inches="tight")
        print(f"\nSaved: {save_dir}/training_curve.png")
    except ImportError:
        print("(matplotlib not available)")

    print("\nDone.")


if __name__ == "__main__":
    main()

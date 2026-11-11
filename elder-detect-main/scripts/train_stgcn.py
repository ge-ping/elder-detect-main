#!/usr/bin/env python
"""
train_stgcn.py — 合并数据集，加权 MSE 训练 ST-GCN（model/stgcn/stgcn.py）

标签 = 连续 risk∈[0,1]（来自 datasets/experiment/skeleton_wXX/ 的 y_risk_*.npy）。
输入 = (N, 3, T, V)：每样本是 (T, 51) 的扁平骨架，喂模型前 reshape→(T,17,3)→(3,T,17)。
损失 / 优化器 / 早停 与 train_risk.py (LSTM) 完全一致 → A/B 公平。

用法:
    python scripts/train_stgcn.py --data_dir datasets/experiment/skeleton_w35
    python scripts/train_stgcn.py --data_dir datasets/experiment/skeleton_w25 --run_dir model/stgcn/stgcn_v1/run_cmp25
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

from model.stgcn.stgcn import STGCN
from model.stgcn.graph import NUM_FALL_JOINTS, FALL_JOINT_SET_NAME
# 复用 LSTM 训练的损失函数，保证 A/B 对比一致
from scripts.train_risk import class_weighted_mse, midband_mse, weighted_mse
from scripts.build_config import (
    RISK_EPOCHS, RISK_BATCH_SIZE, RISK_LR, RISK_VAL_RATIO, RISK_PATIENCE,
    REG_BETA, REG_LOSS_MODE, REG_W_NORMAL, REG_W_UNSTABLE, REG_W_FALL,
    REG_MIDBAND_CENTER, REG_MIDBAND_WIDTH, REG_MIDBAND_PEAK,
    RAMP_LO, RAMP_HI, TAU_U, TAU_ABN, RISK_MODE,
    STGCN_RUN_ROOT, STGCN_NUM_JOINTS, STGCN_INPUT_CHANNEL,
    STGCN_LAYERS, STGCN_TEMP_KERNEL, STGCN_DROPOUT, STGCN_EPOCHS,
    STGCN_BATCH_SIZE, STGCN_LR, STGCN_VAL_RATIO, STGCN_PATIENCE,
    STGCN_JOINT_SET,
)

DATASETS = ["Le2i", "UR-FALL", "CAUCAFall", "UP-FALL", "Pre-VFall"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 派生分类阈值（与 eval_risk.py / demo 保持一致）
THR_WARN = 0.3
THR_DANGER = 0.6


def get_criterion(args) -> callable:
    if args.loss == "weighted_mse":
        return lambda p, t: weighted_mse(p, t, beta=args.beta)
    if args.loss == "midband":
        return lambda p, t: midband_mse(
            p, t, center=args.midband_center,
            width=args.midband_width, peak=args.midband_peak)
    return lambda p, t: class_weighted_mse(
        p, t, w_n=args.w_normal, w_u=args.w_unstable, w_f=args.w_fall)


# ==================================================================
# 加载骨骼数据（回归：读 y_risk，float）
# X 形状 (N, window, num_joints*3)；喂模型前 reshape→(N,3,window,num_joints)
# ==================================================================

def load_datasets(data_dir: Path, dataset_names: list[str], num_joints: int, input_channel: int):
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for name in dataset_names:
        d = data_dir / name
        for split, xl, yl in [("train", X_train_list, y_train_list),
                               ("test", X_test_list, y_test_list)]:
            xp = d / f"X_{split}.npy"
            yp = d / f"y_risk_{split}.npy"
            if xp.exists() and yp.exists():
                X = np.load(xp).astype(np.float32)   # (N, window, K=51)
                y = np.load(yp).astype(np.float32)
                xl.append(torch.from_numpy(X))
                yl.append(torch.from_numpy(y).float().unsqueeze(1))
                print(f"  {name} {split}: X{X.shape}, y_risk{y.shape}")

    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    X_test = torch.cat(X_test_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)
    return X_train, y_train, X_test, y_test


def to_model_input(X: torch.Tensor, num_joints: int, input_channel: int) -> torch.Tensor:
    """(N, window, num_joints*3) → (N, 3, window, num_joints)"""
    N, T, _ = X.shape
    x = X.view(N, T, num_joints, input_channel)
    x = x.permute(0, 3, 1, 2)   # (N, C, T, V)
    return x


# ==================================================================
# 训练
# ==================================================================

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--window", type=int, default=None, help="窗口帧数（默认从数据推断）")
    p.add_argument("--num_joints", type=int, default=STGCN_NUM_JOINTS)
    p.add_argument("--input_channel", type=int, default=STGCN_INPUT_CHANNEL)
    p.add_argument("--epochs", type=int, default=STGCN_EPOCHS)
    p.add_argument("--batch_size", type=int, default=STGCN_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=STGCN_LR)
    p.add_argument("--val_ratio", type=float, default=STGCN_VAL_RATIO)
    p.add_argument("--patience", type=int, default=STGCN_PATIENCE)
    p.add_argument("--beta", type=float, default=REG_BETA)
    p.add_argument("--loss", type=str, default=REG_LOSS_MODE,
                   choices=["class_weighted", "weighted_mse", "midband"])
    p.add_argument("--w_normal", type=float, default=REG_W_NORMAL)
    p.add_argument("--w_unstable", type=float, default=REG_W_UNSTABLE)
    p.add_argument("--w_fall", type=float, default=REG_W_FALL)
    p.add_argument("--midband_center", type=float, default=REG_MIDBAND_CENTER)
    p.add_argument("--midband_width", type=float, default=REG_MIDBAND_WIDTH)
    p.add_argument("--midband_peak", type=float, default=REG_MIDBAND_PEAK)
    p.add_argument("--layers", type=str, default=None,
                   help="逗号分隔通道列表，如 64,64,128,128,256")
    args = p.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else (
        PROJECT_ROOT / "datasets" / "experiment" / "skeleton_w35")
    num_joints = args.num_joints
    input_channel = args.input_channel
    # 关节集：13 关节 → coco13（YOLO 2D），否则 fall17（MediaPipe 3D）
    joint_set_name = "coco13" if num_joints == 13 else "fall17"
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(STGCN_LAYERS)

    print(f"Device: {DEVICE}")
    print(f"Data: {data_dir}, num_joints={num_joints}, C={input_channel}, layers={layers}")
    print("Loading datasets...")
    X_train_all, y_train_all, X_test, y_test = load_datasets(
        data_dir, DATASETS, num_joints, input_channel)

    # 窗口长度：优先 CLI，否则从数据推断
    if args.window:
        window = args.window
    else:
        window = int(X_train_all.shape[1])
    print(f"Window: {window}\n")

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
    base_dir = Path(args.run_dir) if args.run_dir else (PROJECT_ROOT / STGCN_RUN_ROOT)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = 1
    while (base_dir / f"run_{run_id:02d}").exists():
        run_id += 1
    save_dir = base_dir / f"run_{run_id:02d}"
    save_dir.mkdir(parents=True)
    print(f"\nRun dir: {save_dir}")

    def collate_reshape(batch):
        Xb, yb = torch.utils.data.default_collate(batch)
        return to_model_input(Xb, num_joints, input_channel), yb

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_reshape)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_reshape)
    test_ds = TensorDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_reshape)

    # ---- 模型 ----
    model = STGCN(num_joints=num_joints, in_channels=input_channel,
                  layers=layers, window=window,
                  temp_kernel=STGCN_TEMP_KERNEL, dropout=STGCN_DROPOUT,
                  joint_set=joint_set_name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: STGCN, params={n_params}")
    criterion = get_criterion(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    # ---- 超参记录 ----
    with open(save_dir / "config.txt", "w") as f:
        f.write(f"task=regression\nmodel=stgcn\n")
        f.write(f"num_joints={num_joints}\ninput_channel={input_channel}\n")
        f.write(f"window={window}\nlayers={layers}\ntemp_kernel={STGCN_TEMP_KERNEL}\n")
        f.write(f"epochs={args.epochs}\nbatch_size={args.batch_size}\nlr={args.lr}\n")
        f.write(f"val_ratio={args.val_ratio}\npatience={args.patience}\nloss={args.loss}\n")
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

    def r2_score(yt, yp):
        yt = yt.view(-1).cpu().numpy()
        yp = yp.view(-1).cpu().numpy()
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    for epoch in range(1, args.epochs + 1):
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

    print(f"\nBest: epoch {best_epoch}, val MSE = {best_mse:.4f}")

    # 保存元信息供 demo 自动适配（关键：model_type=stgcn）
    # 读取训练数据集 meta 中的坐标滤波参数 → 透传，使 demo 能自动按"模型是否平滑训练"开关坐标滤波（零漂移）
    ls_info: dict = {}
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        try:
            _meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ls_info = _meta.get("landmark_smoothing", {}) or {}
        except Exception:
            ls_info = {}
    (save_dir / "model_info.json").write_text(json.dumps({
        "model_type": "stgcn",
        "task": "regression",
        "num_joints": num_joints,
        "input_channel": input_channel,
        "joint_set": joint_set_name,
        "window_size": window,
        "layers": layers,
        "temp_kernel": STGCN_TEMP_KERNEL,
        "dropout": STGCN_DROPOUT,
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
        "landmark_smoothing": ls_info,
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
        fig.suptitle(f"ST-GCN — Best epoch {best_epoch}, val MSE {best_mse:.4f}")
        fig.tight_layout()
        fig.savefig(str(save_dir / "training_curve.png"), dpi=120, bbox_inches="tight")
        print(f"\nSaved: {save_dir}/training_curve.png")
    except ImportError:
        print("(matplotlib not available)")

    print("\nDone.")


if __name__ == "__main__":
    main()

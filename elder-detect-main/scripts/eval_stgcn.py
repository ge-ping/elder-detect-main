#!/usr/bin/env python
"""
eval_stgcn.py — 加载已训练的 ST-GCN，做「五维」评判（与 eval_risk.py 完全同口径，保证 A/B 公平）

  指标同 eval_risk.py:
  (1) 回归：MSE / MAE / RMSE / R² + 分风险带 MAE
  (2) 派生分类 vs 旧基线：pred risk 过 0.3/0.6 还原 N/U/F，算 P/R/F1 + 混淆矩阵 + Unstable F1
      按数据集拆分
  (3) 预判提前量：每个跌倒序列，risk 首次越过 0.3/0.6 距跌倒帧多少帧 → 秒
  (4) ROC / PR + AUC + 阈值扫描（F1 vs 阈值）
  (5) 平滑性：相邻窗口 |Δrisk| 均值（目标 ≤0.05/帧）

输入 = (N, T, V*C) 扁平骨架，喂模型前 reshape → (N, C, T, V)（与 train_stgcn.py 一致）。

用法:
    python scripts/eval_stgcn.py model/stgcn/stgcn_v1/run_01 datasets/experiment/skeleton_w35
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.stgcn.stgcn import STGCN

DATASETS = ["Le2i", "UR-FALL", "CAUCAFall", "UP-FALL", "Pre-VFall"]
FPS_ASSUMED = 25.0          # 标签/缓存默认帧率
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THR_WARN = 0.3
THR_DANGER = 0.6
THR_FALL = 0.6


# ==================================================================
# 骨骼数据 reshape（与 train_stgcn.py 一致）
# ==================================================================
def to_model_input(X: torch.Tensor, num_joints: int, input_channel: int) -> torch.Tensor:
    """(N, T, V*C) → (N, C, T, V)"""
    N, T, _ = X.shape
    x = X.view(N, T, num_joints, input_channel)
    return x.permute(0, 3, 1, 2)


# ==================================================================
# 数据加载
# ==================================================================
def load_test_combined(data_dir: Path):
    Xl, yrl, ycl = [], [], []
    for name in DATASETS:
        d = data_dir / name
        xp = d / "X_test.npy"
        yrp = d / "y_risk_test.npy"
        ycp = d / "y_cls_test.npy"
        if xp.exists() and yrp.exists() and ycp.exists():
            Xl.append(np.load(xp).astype(np.float32))
            yrl.append(np.load(yrp).astype(np.float32))
            ycl.append(np.load(ycp).astype(np.int64))
    return (np.concatenate(Xl), np.concatenate(yrl), np.concatenate(ycl))


def load_test_per_ds(data_dir: Path) -> dict:
    out = {}
    for name in DATASETS:
        d = data_dir / name
        xp, yrp, ycp = d / "X_test.npy", d / "y_risk_test.npy", d / "y_cls_test.npy"
        if xp.exists() and yrp.exists() and ycp.exists():
            out[name] = (np.load(xp).astype(np.float32),
                         np.load(yrp).astype(np.float32),
                         np.load(ycp).astype(np.int64))
    return out


def load_video_windows(data_dir: Path):
    items = []
    for name in DATASETS:
        vd = data_dir / name / "videos"
        if not vd.exists():
            continue
        for xf in sorted(vd.glob("*_X.npy")):
            stem = xf.stem[:-len("_X")]
            yrf = vd / f"{stem}_y_risk.npy"
            ycf = vd / f"{stem}_y_cls.npy"
            if not (yrf.exists() and ycf.exists()):
                continue
            items.append({
                "ds": name, "vid": stem,
                "X": np.load(xf).astype(np.float32),
                "y_risk": np.load(yrf).astype(np.float32),
                "y_cls": np.load(ycf).astype(np.int64),
            })
    return items


# ==================================================================
# 派生分类
# ==================================================================
def derive_cls(risk: np.ndarray) -> np.ndarray:
    c = np.zeros_like(risk, dtype=np.int64)
    c[risk >= THR_WARN] = 1
    c[risk >= THR_FALL] = 2
    return c


# ==================================================================
# 模型加载
# ==================================================================
def load_stgcn(run_dir: Path):
    info = {}
    info_path = run_dir / "model_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
    num_joints = int(info.get("num_joints", 17))
    in_channels = int(info.get("input_channel", 3))
    layers = list(info.get("layers", [64, 64, 128, 128, 256]))
    window = int(info.get("window_size", 35))
    temp_kernel = int(info.get("temp_kernel", 9))
    dropout = float(info.get("dropout", 0.3))
    model = STGCN(num_joints=num_joints, in_channels=in_channels,
                  layers=layers, window=window, temp_kernel=temp_kernel,
                  dropout=dropout).to(DEVICE)
    model.load_state_dict(torch.load(str(run_dir / "best_model.pt"),
                                     map_location=DEVICE, weights_only=True))
    model.eval()
    return model, num_joints, in_channels, window


def predict(model, X_np, V, C, W):
    """X_np: (N, T, V*C) → pred (N,)"""
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_np).float()),
        batch_size=BATCH_SIZE)
    out = []
    with torch.no_grad():
        for (Xb,) in dl:
            Xi = to_model_input(Xb.to(DEVICE), V, C)
            out.append(model(Xi).cpu())
    return torch.cat(out).view(-1).numpy().astype(np.float32)


# ==================================================================
# 主流程
# ==================================================================
def main():
    import argparse
    global THR_WARN, THR_DANGER, THR_FALL
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    p.add_argument("data_dir", nargs="?", default="datasets/experiment/skeleton_w35")
    p.add_argument("--thr_warn", type=float, default=THR_WARN)
    p.add_argument("--thr_danger", type=float, default=THR_DANGER)
    p.add_argument("--thr_fall", type=float, default=THR_FALL)
    args = p.parse_args()
    THR_WARN, THR_DANGER, THR_FALL = args.thr_warn, args.thr_danger, args.thr_fall

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    model, V, C, W = load_stgcn(run_dir)
    print(f"Model: {run_dir}  joints={V}  C={C}  window={W}")
    print(f"Data:  {data_dir}")

    X_test, y_risk, y_cls = load_test_combined(data_dir)
    print(f"Test windows: {X_test.shape[0]},  frame-level risk: {y_risk.shape[0]}")

    pred = predict(model, X_test, V, C, W)
    y_true = y_risk
    y_cls_pred = derive_cls(pred)

    metrics: dict = {}

    # ---- (1) 回归指标 ----
    mse = float(np.mean((pred - y_true) ** 2))
    mae = float(np.mean(np.abs(pred - y_true)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    metrics["regression"] = {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}

    bands = {"[0,0.3)": (0.0, 0.3), "[0.3,0.6)": (0.3, 0.6), "[0.6,1.0]": (0.6, 1.01)}
    band_mae = {}
    for name, (lo, hi) in bands.items():
        m = (y_true >= lo) & (y_true < hi)
        band_mae[name] = float(np.mean(np.abs(pred[m] - y_true[m]))) if m.sum() > 0 else None
    metrics["regression"]["band_mae"] = band_mae
    print(f"\n[1] Regression: MSE={mse:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    for k, v in band_mae.items():
        print(f"      band {k} MAE={v:.4f}" if v is not None else f"      band {k} MAE=N/A")

    # ---- (2) 派生分类 ----
    try:
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        report = classification_report(y_cls, y_cls_pred,
                                       target_names=["Normal", "Unstable", "Fall"], digits=4,
                                       zero_division=0)
        cm = confusion_matrix(y_cls, y_cls_pred, labels=[0, 1, 2])
        u_f1 = float(f1_score(y_cls, y_cls_pred, labels=[1], average=None, zero_division=0)[0])
        macro_f1 = float(f1_score(y_cls, y_cls_pred, average="macro", zero_division=0))
        metrics["derived_classification"] = {
            "unstable_f1": u_f1, "macro_f1": macro_f1,
            "confusion": cm.tolist(),
        }
        print(f"\n[2] Derived classification (risk→{THR_WARN}/{THR_FALL}):")
        print(report)
        print("Confusion (rows=true, cols=pred):")
        print(f"{'':>10} {'Pred N':>8} {'Pred U':>8} {'Pred F':>8}")
        for i, nm in enumerate(["True N", "True U", "True F"]):
            print(f"{nm:>10} {cm[i,0]:8d} {cm[i,1]:8d} {cm[i,2]:8d}")
        print(f"  >> Unstable F1 = {u_f1:.4f}  (LSTM 旧最佳基线 0.664)")

        per_ds = {}
        per_ds_data = load_test_per_ds(data_dir)
        for ds, (Xds, yrd, ycd) in per_ds_data.items():
            pred_ds = predict(model, Xds, V, C, W)
            yc_pred_ds = derive_cls(pred_ds)
            uf = float(f1_score(ycd, yc_pred_ds, labels=[1], average=None, zero_division=0)[0])
            per_ds[ds] = {"unstable_f1": uf, "n": int(len(ycd))}
            print(f"      {ds:>10}: Unstable F1={uf:.4f}  (n={len(ycd)})")
        metrics["derived_classification"]["per_dataset"] = per_ds
    except ImportError:
        print("(sklearn not available — skip derived classification)")

    # ---- (3) 预判提前量 + (5) 平滑性 + 抽样曲线 ----
    items = load_video_windows(data_dir)
    lead03, lead06, lead06_frac, smoothness = [], [], [], []
    sample_curves = []
    for it in items:
        Xv, yrv, ycv = it["X"], it["y_risk"], it["y_cls"]
        n_w = Xv.shape[0]
        if n_w == 0:
            continue
        pred_v = predict(model, Xv, V, C, W)  # 长度 n_w，对应 frame f=j+W-1

        fall_w = np.where(ycv == 2)[0]
        if fall_w.size > 0:
            f_fall_frame = fall_w[0] + (W - 1)
            cross03 = np.where(pred_v >= THR_WARN)[0]
            cross06 = np.where(pred_v >= THR_DANGER)[0]
            if cross03.size > 0:
                lead03.append((f_fall_frame - (cross03[0] + (W - 1))) / FPS_ASSUMED)
            if cross06.size > 0:
                lead06.append((f_fall_frame - (cross06[0] + (W - 1))) / FPS_ASSUMED)
                lead06_frac.append(1.0)
            else:
                lead06_frac.append(0.0)

        if n_w > 1:
            smoothness.append(float(np.mean(np.abs(np.diff(pred_v)))))

        if len(sample_curves) < 4 and fall_w.size > 0:
            T = (W - 1) + n_w
            frame_pred = np.zeros(T, dtype=np.float32)
            frame_true = np.zeros(T, dtype=np.float32)
            for j in range(n_w):
                f = j + (W - 1)
                frame_pred[f] = pred_v[j]
                frame_true[f] = yrv[j]
            sample_curves.append({"ds": it["ds"], "vid": it["vid"],
                                 "frame_pred": frame_pred, "frame_true": frame_true})

    metrics["lead_time"] = {
        "danger_mean_sec": float(np.mean(lead06)) if lead06 else None,
        "danger_median_sec": float(np.median(lead06)) if lead06 else None,
        "warn_mean_sec": float(np.mean(lead03)) if lead03 else None,
        "danger_before_fall_frac": float(np.mean(lead06_frac)) if lead06_frac else None,
        "n_fall_videos": len(lead06_frac),
    }
    metrics["smoothness"] = {
        "mean_abs_delta_risk": float(np.mean(smoothness)) if smoothness else None,
    }
    print(f"\n[3] Lead time (DANGER≥{THR_DANGER}, fps≈{FPS_ASSUMED:.0f}):")
    if lead06:
        print(f"      DANGER 平均提前 {np.mean(lead06):.2f}s  中位数 {np.median(lead06):.2f}s")
        print(f"      WARNING 平均提前 {np.mean(lead03):.2f}s")
        print(f"      跌倒前触发 DANGER 比例: {np.mean(lead06_frac):.2%}  (n={len(lead06_frac)})")
    elif items:
        print(f"      (跌倒视频 {len(lead06_frac)} 个，但无一窗口预测 ≥{THR_DANGER})")
    else:
        print("      (未找到逐视频窗口文件 skeleton_w35/<ds>/videos/)")
    print(f"\n[5] Smoothness: 相邻窗口 |Δrisk| 均值 = "
          f"{np.mean(smoothness):.4f}/帧  (目标 ≤0.05)"
          if smoothness else
          "\n[5] Smoothness: N/A")

    # ---- (4) ROC / PR + 阈值扫描 ----
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
        pos = (y_true >= THR_FALL).astype(np.int64)
        try:
            auc_roc = float(roc_auc_score(pos, pred))
        except ValueError:
            auc_roc = None
        try:
            auc_pr = float(average_precision_score(pos, pred))
        except ValueError:
            auc_pr = None
        thrs = np.linspace(0.05, 0.95, 19)
        scan = {"thr": [], "prec": [], "rec": [], "f1": []}
        for t in thrs:
            pp = (pred >= t).astype(np.int64)
            tp = int(np.sum((pp == 1) & (pos == 1)))
            fp = int(np.sum((pp == 1) & (pos == 0)))
            fn = int(np.sum((pp == 0) & (pos == 1)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            scan["thr"].append(float(t)); scan["prec"].append(prec)
            scan["rec"].append(rec); scan["f1"].append(f1)
        metrics["roc_pr"] = {"auc_roc": auc_roc, "auc_pr": auc_pr, "scan": scan}
        print(f"\n[4] ROC/PR (Fall = risk≥{THR_FALL}): "
              f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}")
    except ImportError:
        print("(sklearn not available — skip ROC/PR)")

    # ---- 输出 metrics.json ----
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved: {run_dir}/metrics.json")

    # ---- 画图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if "derived_classification" in metrics:
            cm = np.array(metrics["derived_classification"]["confusion"])
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            try:
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Normal", "Unstable", "Fall"],
                            yticklabels=["Normal", "Unstable", "Fall"], ax=ax)
            except ImportError:
                ax.imshow(cm, cmap="Blues")
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                                color="white" if cm[i, j] > cm.max() / 2 else "black")
                ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["N", "U", "F"])
                ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["N", "U", "F"])
            ax.set_title(f"Confusion (derived) — U-F1="
                         f"{metrics['derived_classification']['unstable_f1']:.3f}")
            fig.tight_layout()
            fig.savefig(str(run_dir / "confusion.png"), dpi=120)
            plt.close(fig)
            print(f"Saved: {run_dir}/confusion.png")

        if "roc_pr" in metrics and "scan" in metrics["roc_pr"]:
            scan = metrics["roc_pr"]["scan"]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(scan["thr"], scan["prec"], label="Precision")
            axes[0].plot(scan["thr"], scan["rec"], label="Recall")
            axes[0].plot(scan["thr"], scan["f1"], label="F1", color="black", lw=2)
            axes[0].set_xlabel("threshold"); axes[0].set_ylabel("score")
            axes[0].set_title("Threshold scan (Fall)"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
            axes[1].hist(pred[pos == 0], bins=30, alpha=0.6, label="Non-Fall", density=True)
            axes[1].hist(pred[pos == 1], bins=30, alpha=0.6, label="Fall", density=True)
            axes[1].set_xlabel("predicted risk"); axes[1].set_ylabel("density")
            axes[1].set_title("Pred risk dist"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
            for sc in sample_curves:
                xs = np.arange(len(sc["frame_true"]))
                axes[2].plot(xs, sc["frame_true"], "--", alpha=0.7, label=f"{sc['ds']} target")
                axes[2].plot(xs, sc["frame_pred"], alpha=0.9, label=f"{sc['ds']} pred")
            axes[2].axhline(THR_WARN, color="orange", ls=":", lw=1)
            axes[2].axhline(THR_FALL, color="red", ls=":", lw=1)
            axes[2].set_xlabel("frame"); axes[2].set_ylabel("risk")
            axes[2].set_title("Sample risk curves"); axes[2].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(str(run_dir / "risk_curves_sample.png"), dpi=120)
            plt.close(fig)
            print(f"Saved: {run_dir}/risk_curves_sample.png")

        if lead06:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.hist(lead06, bins=20, color="tomato", alpha=0.8)
            ax.axvline(0, color="black", ls="--", lw=1)
            ax.set_xlabel("DANGER lead time (sec, before fall)")
            ax.set_ylabel("# fall videos")
            ax.set_title(f"Lead time — mean={np.mean(lead06):.2f}s")
            fig.tight_layout()
            fig.savefig(str(run_dir / "lead_time_hist.png"), dpi=120)
            plt.close(fig)
            print(f"Saved: {run_dir}/lead_time_hist.png")
    except ImportError:
        print("(matplotlib not available — skip figures)")

    print("\nDone.")


if __name__ == "__main__":
    main()

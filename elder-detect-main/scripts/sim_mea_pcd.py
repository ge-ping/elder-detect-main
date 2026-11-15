"""Stage-0 property check: existing RMS-of-robust-z rule vs measurement-error-aware
personal change detection (MEA-PCD). All methods compared at MATCHED false-alarm rate.

Run from elder-detect-main:  python <this file>
"""
from __future__ import annotations

import sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, ".")
from longitudinal.baseline import _feature_stats, PersonalBaseline  # noqa: E402
from longitudinal.deviation import compute_deviation  # noqa: E402
from longitudinal.config import DEFAULT_CORE_FEATURES as F  # noqa: E402

RNG = np.random.default_rng(20261113)
K = len(F)
IDX = {n: i for i, n in enumerate(F)}
N_BASE, N_MON, ONSET = 14, 30, 8          # change starts on monitoring day 8
N_PERSON = 2000
TARGET_FAR = 0.10                          # P(>=1 false alarm in 30 null days)

# decline direction (+1 = increases when gait deteriorates, 0 = no prior)
DIR = np.zeros(K)
for n, d in {"gait_speed_proxy": -1, "cadence_spm": -1, "step_time_mean_s": +1,
             "step_time_cv": +1, "stance_width_norm": +1, "gait_asymmetry": +1,
             "trunk_tilt_mean_deg": +1, "trunk_sway_rms_deg": +1,
             "aspect_ratio_cv": +1, "emos_proxy": -1}.items():
    DIR[IDX[n]] = d

# measurement SE / natural day-to-day SD at 30 steps (variability features are noisy)
C30 = np.array([{"gait_speed_proxy": .5, "cadence_spm": .4, "step_time_mean_s": .4,
                 "step_time_cv": 1.5, "stance_width_norm": .8, "gait_asymmetry": 1.5,
                 "trunk_tilt_mean_deg": .6, "trunk_sway_rms_deg": 1.0,
                 "aspect_ratio_mean": .6, "aspect_ratio_cv": 1.2, "emos_proxy": 1.0}[n] for n in F])

R = np.eye(K)
def _c(a, b, r):
    R[IDX[a], IDX[b]] = R[IDX[b], IDX[a]] = r
_c("cadence_spm", "step_time_mean_s", -.9); _c("gait_speed_proxy", "cadence_spm", .6)
_c("gait_speed_proxy", "step_time_mean_s", -.55); _c("step_time_cv", "gait_asymmetry", .3)
_c("trunk_sway_rms_deg", "aspect_ratio_cv", .4); _c("stance_width_norm", "emos_proxy", .3)
LR = np.linalg.cholesky(R)

SCEN = {
    "S1_variability(sparse)": {"step_time_cv": 1, "gait_asymmetry": .5, "gait_speed_proxy": -.5},
    "S2_slowing(correlated)": {"gait_speed_proxy": -1, "cadence_spm": -.7, "step_time_mean_s": .7},
    "S3_balance": {"trunk_sway_rms_deg": 1, "stance_width_norm": .7, "emos_proxy": -.7, "aspect_ratio_cv": .5},
    "S4_off-prior(aspect_ratio_mean)": {"aspect_ratio_mean": 1},
}


def simulate(n_person, shift_vec, delta, ramp_days, hetero=True):
    T = N_BASE + N_MON
    nat = RNG.standard_normal((n_person, T, K)) @ LR.T                  # tau = 1
    steps = (np.exp(RNG.uniform(np.log(20), np.log(150), (n_person, T, 1)))
             if hetero else np.full((n_person, T, 1), 60.0))
    s_true = C30 * np.sqrt(30.0 / steps)                                   # (P,T,K)
    meas = (RNG.standard_normal((n_person, T, K)) @ LR.T) * s_true
    x = nat + meas
    # bootstrap SE estimate is itself noisy (df=10)
    s_hat = s_true * np.sqrt(RNG.chisquare(10, (n_person, T, K)) / 10)
    if delta:
        prof = np.zeros(T)
        days = np.arange(N_MON) - ONSET
        prof[N_BASE:] = np.clip((days + 1) / max(ramp_days, 1), 0, 1) * (days >= 0)
        x = x + delta * prof[None, :, None] * shift_vec[None, None, :]
    # random per-person level/scale so that nothing relies on unit scale
    return x, s_hat


# ---------- methods: each returns per-day statistic (P, N_MON) ----------
def existing_distance(x):
    """Exactly the project rule: per-feature median/MAD(->IQR) on 14 days, clipped RMS."""
    b, m = x[:, :N_BASE], x[:, N_BASE:]
    med = np.median(b, 1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(b - med), 1, keepdims=True)
    z = np.clip((m - med) / scale, -5, 5)
    return np.sqrt(np.mean(z ** 2, axis=2))


def dl_baseline(b, sb):
    """DerSimonian-Laird random effects per feature with known per-day SE."""
    v = sb ** 2
    w = 1 / v
    mu_fe = (w * b).sum(1, keepdims=True) / w.sum(1, keepdims=True)
    Q = (w * (b - mu_fe) ** 2).sum(1, keepdims=True)
    c = w.sum(1, keepdims=True) - (w ** 2).sum(1, keepdims=True) / w.sum(1, keepdims=True)
    tau2 = np.maximum(0, (Q - (b.shape[1] - 1)) / c)
    tau2 = np.maximum(tau2, 0.05 * np.median(v, 1, keepdims=True))      # floor
    ws = 1 / (tau2 + v)
    mu = (ws * b).sum(1, keepdims=True) / ws.sum(1, keepdims=True)
    var_mu = 1 / ws.sum(1, keepdims=True)
    return mu, tau2, var_mu


def mea_z(x, s_hat):
    b, sb = x[:, :N_BASE], s_hat[:, :N_BASE]
    mu, tau2, var_mu = dl_baseline(b, sb)
    z_mon = (x[:, N_BASE:] - mu) / np.sqrt(tau2 + s_hat[:, N_BASE:] ** 2 + var_mu)
    # leave-one-out-free baseline residuals for correlation estimate
    z_base = (b - mu) / np.sqrt(tau2 + sb ** 2)
    return z_mon, z_base


def shrink_corr(zb):
    """Ledoit-Wolf-style fixed shrinkage of per-person residual correlation."""
    P, n, k = zb.shape
    zc = zb - zb.mean(1, keepdims=True)
    S = np.einsum("pnk,pnl->pkl", zc, zc) / (n - 1)
    d = np.sqrt(np.einsum("pkk->pk", S))
    Rh = S / d[:, :, None] / d[:, None, :]
    lam = 0.5
    return lam * np.eye(k) + (1 - lam) * Rh


def chibar_stat(z):
    """One-sided (decline-direction) chi-bar^2 on directional features, two-sided on undirected."""
    u = np.where(DIR != 0, np.maximum(0, z * DIR), np.abs(z) / np.sqrt(2))
    return (u ** 2).sum(-1)


def null_cdf_q(T, Rp, n_mc=4000):
    """Per-person MC null distribution of chi-bar under estimated correlation -> normal score."""
    P = T.shape[0]
    q = np.empty_like(T)
    for p in range(P):
        L = np.linalg.cholesky(Rp[p] + 1e-9 * np.eye(K))
        sim = chibar_stat(RNG.standard_normal((n_mc, K)) @ L.T)
        sim.sort()
        r = np.searchsorted(sim, T[p], side="right")
        q[p] = norm.ppf(np.clip((r + 0.5) / (n_mc + 1), 1e-6, 1 - 1e-6))
    return q


def run_rule_consec(stat, thr, k=3):
    """alarm day index (first day with k consecutive stat>=thr), or -1."""
    above = stat >= thr
    run = np.zeros(stat.shape[0], int)
    first = np.full(stat.shape[0], -1)
    for t in range(stat.shape[1]):
        run = np.where(above[:, t], run + 1, 0)
        hit = (run >= k) & (first < 0)
        first[hit] = t
    return first


def run_cusum(q, h, kref=0.5):
    C = np.zeros(q.shape[0]); first = np.full(q.shape[0], -1)
    for t in range(q.shape[1]):
        C = np.maximum(0, C + q[:, t] - kref)
        hit = (C >= h) & (first < 0)
        first[hit] = t
    return first


def calibrate(null_stat, rule, grid):
    """smallest threshold whose 30-day null FAR <= target."""
    for thr in grid:
        if np.mean(rule(null_stat, thr) >= 0) <= TARGET_FAR:
            return thr
    return grid[-1]


def methods(x, s_hat):
    out = {}
    out["A0/A1 existing RMS robust-z"] = existing_distance(x)
    zm, zb = mea_z(x, s_hat)
    out["A2 +measurement-error-aware z (RMS)"] = np.sqrt(np.mean(np.clip(zm, -5, 5) ** 2, -1))
    T = chibar_stat(zm)
    out["A3 +directional chi-bar (3-day rule)"] = T
    out["A4 full MEA-PCD (chi-bar + CUSUM)"] = ("cusum", T, shrink_corr(zb))
    return out


def evaluate(hetero=True, delta=1.0, ramp=14):
    xn, sn = simulate(N_PERSON, np.zeros(K), 0, 1, hetero)
    null = methods(xn, sn)
    thr = {}
    for name, st in null.items():
        if isinstance(st, tuple):
            q = null_cdf_q(st[1], st[2]); null[name] = ("cusum", q)
            thr[name] = calibrate(q, lambda s, h: run_cusum(s, h), np.arange(0.5, 30, 0.1))
        else:
            thr[name] = calibrate(st, run_rule_consec, np.quantile(st, np.linspace(.5, .9999, 600)))
    default_far = np.mean(run_rule_consec(null["A0/A1 existing RMS robust-z"], 3.0) >= 0)
    rows = []
    for sname, spec in SCEN.items():
        sv = np.zeros(K)
        for n, v in spec.items():
            sv[IDX[n]] = v
        xa, sa = simulate(N_PERSON, sv, delta, ramp, hetero)
        alt = methods(xa, sa)
        res = {}
        d0 = run_rule_consec(alt["A0/A1 existing RMS robust-z"], 3.0)
        res["A0 existing default (RMS>=3 x3d)"] = d0
        for name, st in alt.items():
            if isinstance(st, tuple):
                first = run_cusum(null_cdf_q(st[1], st[2]), thr[name])
            else:
                first = run_rule_consec(st, thr[name])
            res[name.replace("A0/A1 existing", "A1 existing@matchedFAR")] = first
        for name, first in res.items():
            det = (first >= ONSET) & (first <= ONSET + 21)
            delay = first[det] - ONSET
            rows.append((sname, name, det.mean(), np.median(delay) if det.any() else np.nan))
    return default_far, thr, rows


if __name__ == "__main__":
    # fidelity check: vectorised existing_distance == project compute_deviation
    x, _ = simulate(3, np.zeros(K), 0, 1)
    vec = existing_distance(x)
    for p in range(3):
        feats = {n: _feature_stats(list(x[p, :N_BASE, i])) for i, n in enumerate(F)}
        bl = PersonalBaseline("p", "t", N_BASE, tuple(), feats, "fp")
        ref = compute_deviation({n: x[p, N_BASE, i] for i, n in enumerate(F)}, bl).distance
        assert abs(ref - vec[p, 0]) < 1e-9, (ref, vec[p, 0])
    print("fidelity check vs project compute_deviation: OK")
    for hetero in (True, False):
        for delta, ramp in ((1.0, 14), (1.5, 1)):
            far0, thr, rows = evaluate(hetero, delta, ramp)
            print(f"\n=== session quality {'heterogeneous(20-150 steps)' if hetero else 'homogeneous(60 steps)'}"
                  f" | delta={delta} tau | {'ramp 14d' if ramp > 1 else 'step'} | "
                  f"existing default null FAR/30d = {far0:.3f} | matched FAR target = {TARGET_FAR}")
            cur = None
            for s, m, dr, dl in rows:
                if s != cur:
                    print(f"  [{s}]"); cur = s
                print(f"    {m:42s} detect@21d={dr:6.1%}  median_delay={dl:5.1f}d")

#!/usr/bin/env python3
"""
Hierarchical 2x2 diversity x synthesis analysis.

This script fits the matched MoA 2x2 ablation with a random intercept per task.
It is a conservative alternative to the fixed-effects regression in
analyze_2x2_matched.py:

    Q ~ diversity * synthesis + (1 | task)

The implementation uses a closed-form REML objective for a random-intercept
linear mixed model, so it does not require statsmodels.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


DATA_DIR = "results/full-v2"
OUT_DIR = "results/analysis"

TASKS = {
    # taskId: (direction, s_bad, s_ref)
    "bench-difference-bases": ("minimize", 14.0, 1.0),
    "bench-erdos-overlap": ("minimize", 0.2939, 0.2321),
    "bench-molecule-qed": ("maximize", 0.60, 1.0),
}

PROTOCOLS = {
    # protocolId: (diversity, synthesis)
    "moa": (1, 1),  # diverse + synthesis
    "moa-same-model": (0, 1),  # same + synthesis
    "moa-nosynth": (1, 0),  # diverse + selection
    "best-of-3": (0, 0),  # same + selection
}

SEED_RANGES = {
    "moa": range(1, 11),
    "moa-same-model": range(1, 11),
    "moa-nosynth": range(1, 11),
    "best-of-3": range(1, 11),
}


def compute_q(score: float, direction: str, s_bad: float, s_ref: float) -> float:
    """Normalize a raw hidden score to Q in [0, 1]."""
    if direction == "minimize":
        q = (s_bad - score) / (s_bad - s_ref)
    else:
        q = (score - s_bad) / (s_ref - s_bad)
    return max(0.0, min(1.0, q))


@dataclass
class MixedModelFit:
    beta: np.ndarray
    vcov: np.ndarray
    sigma2: float
    tau2: float
    loglik: float
    success: bool
    message: str


def load_data() -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    missing: List[str] = []

    for task_idx, (task_id, (direction, s_bad, s_ref)) in enumerate(TASKS.items()):
        for proto_id, (div_flag, synth_flag) in PROTOCOLS.items():
            for seed in SEED_RANGES[proto_id]:
                fname = f"{task_id}_{proto_id}_s{seed}.json"
                fpath = os.path.join(DATA_DIR, fname)
                if not os.path.exists(fpath):
                    missing.append(fname)
                    continue
                with open(fpath) as f:
                    data = json.load(f)
                score = data["hiddenScore"]
                q = compute_q(score, direction, s_bad, s_ref)
                rows.append(
                    {
                        "task": task_id,
                        "task_idx": task_idx,
                        "protocol": proto_id,
                        "seed": seed,
                        "Q": q,
                        "diversity": float(div_flag),
                        "synthesis": float(synth_flag),
                        "interaction": float(div_flag * synth_flag),
                    }
                )

    df = pd.DataFrame(rows)
    print(
        f"Loaded {len(df)} observations across {len(TASKS)} tasks "
        f"and {len(PROTOCOLS)} protocols"
    )
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        print(f"WARNING: {len(missing)} missing files: {preview}{suffix}")
    print()
    return df


def group_indices(tasks: np.ndarray) -> List[np.ndarray]:
    groups: List[np.ndarray] = []
    for task_id in np.unique(tasks):
        groups.append(np.where(tasks == task_id)[0])
    return groups


def fit_random_intercept_reml(X: np.ndarray, y: np.ndarray, groups: List[np.ndarray]) -> MixedModelFit:
    """
    Fit y = X beta + b_g + eps, b_g ~ N(0, tau2), eps ~ N(0, sigma2).
    The covariance within each group is sigma2 * I + tau2 * 11'.
    """

    n, p = X.shape

    def reml_nll(log_params: np.ndarray) -> float:
        log_sigma2, log_tau2 = log_params
        sigma2 = float(np.exp(log_sigma2))
        tau2 = float(np.exp(log_tau2))

        xtvix = np.zeros((p, p), dtype=float)
        xtviy = np.zeros(p, dtype=float)
        yviy = 0.0
        logdet_v = 0.0

        for idx in groups:
            yg = y[idx]
            Xg = X[idx]
            ng = len(idx)
            a = sigma2
            b = tau2

            # V^{-1} for sigma2 I + tau2 11'
            inv = (1.0 / a) * np.eye(ng) - (b / (a * (a + ng * b))) * np.ones((ng, ng))

            xtvix += Xg.T @ inv @ Xg
            xtviy += Xg.T @ inv @ yg
            yviy += float(yg.T @ inv @ yg)
            logdet_v += (ng - 1) * np.log(a) + np.log(a + ng * b)

        sign, logdet_xtvix = np.linalg.slogdet(xtvix)
        if sign <= 0:
            return np.inf

        quad = yviy - float(xtviy.T @ np.linalg.solve(xtvix, xtviy))
        return 0.5 * (logdet_v + logdet_xtvix + quad + (n - p) * np.log(2 * np.pi))

    # A small positive initialization works well on this dataset.
    init = np.log(np.array([0.05, 0.01], dtype=float))
    res = minimize(
        reml_nll,
        init,
        method="Nelder-Mead",
        options={"maxiter": 20_000, "xatol": 1e-8, "fatol": 1e-8},
    )

    sigma2 = float(np.exp(res.x[0]))
    tau2 = float(np.exp(res.x[1]))

    xtvix = np.zeros((p, p), dtype=float)
    xtviy = np.zeros(p, dtype=float)
    logdet_v = 0.0
    yviy = 0.0

    for idx in groups:
        yg = y[idx]
        Xg = X[idx]
        ng = len(idx)
        a = sigma2
        b = tau2
        inv = (1.0 / a) * np.eye(ng) - (b / (a * (a + ng * b))) * np.ones((ng, ng))

        xtvix += Xg.T @ inv @ Xg
        xtviy += Xg.T @ inv @ yg
        yviy += float(yg.T @ inv @ yg)
        logdet_v += (ng - 1) * np.log(a) + np.log(a + ng * b)

    beta = np.linalg.solve(xtvix, xtviy)
    vcov = np.linalg.inv(xtvix)
    quad = yviy - float(xtviy.T @ beta)
    sign, logdet_xtvix = np.linalg.slogdet(xtvix)
    loglik = -0.5 * (logdet_v + logdet_xtvix + quad + (n - p) * np.log(2 * np.pi))

    return MixedModelFit(
        beta=beta,
        vcov=vcov,
        sigma2=sigma2,
        tau2=tau2,
        loglik=loglik,
        success=bool(res.success),
        message=str(res.message),
    )


def fit_ols_with_task_dummies(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    sigma2 = float(np.sum(resid ** 2) / (len(y) - X.shape[1]))
    vcov = sigma2 * np.linalg.inv(X.T @ X)
    return beta, vcov, sigma2


def ci_from_vcov(beta: np.ndarray, vcov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    se = np.sqrt(np.diag(vcov))
    return beta - 1.96 * se, beta + 1.96 * se


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()

    # Random-intercept model: intercept + diversity + synthesis + interaction.
    X_re = np.column_stack(
        [
            np.ones(len(df)),
            df["diversity"].to_numpy(),
            df["synthesis"].to_numpy(),
            df["interaction"].to_numpy(),
        ]
    )
    y = df["Q"].to_numpy()
    groups = group_indices(df["task_idx"].to_numpy())

    mixed = fit_random_intercept_reml(X_re, y, groups)
    lo, hi = ci_from_vcov(mixed.beta, mixed.vcov)

    # Fixed-effects task-dummy baseline for comparison.
    task_dummies = np.zeros((len(df), len(TASKS) - 1))
    task_idx = df["task_idx"].to_numpy().astype(int)
    for t in range(1, len(TASKS)):
        task_dummies[:, t - 1] = (task_idx == t).astype(float)

    X_fe = np.column_stack(
        [
            np.ones(len(df)),
            df["diversity"].to_numpy(),
            df["synthesis"].to_numpy(),
            df["interaction"].to_numpy(),
            task_dummies,
        ]
    )
    fe_beta, fe_vcov, fe_sigma2 = fit_ols_with_task_dummies(X_fe, y)
    fe_lo, fe_hi = ci_from_vcov(fe_beta, fe_vcov)

    cols_re = ["intercept", "diversity", "synthesis", "diversity:synthesis"]
    cols_fe = ["intercept", "diversity", "synthesis", "diversity:synthesis"] + [
        f"task_{task}" for task in list(TASKS.keys())[1:]
    ]

    print("=" * 80)
    print("RANDOM-INTERCEPT MIXED MODEL")
    print("Model: Q ~ diversity * synthesis + (1 | task)")
    print("=" * 80)
    print(f"Converged: {mixed.success} ({mixed.message})")
    print(f"REML logLik: {mixed.loglik:.4f}")
    print(f"Residual variance sigma^2: {mixed.sigma2:.6f}")
    print(f"Task intercept variance tau^2: {mixed.tau2:.6f}")
    print()
    for name, b, l, h in zip(cols_re, mixed.beta, lo, hi):
        sig = "*" if (l > 0 or h < 0) else ""
        print(f"  {name:20s} {b:+.4f}  [{l:+.4f}, {h:+.4f}] {sig}")
    print()

    print("=" * 80)
    print("TASK-FIXED EFFECTS BASELINE")
    print("Model: Q ~ diversity * synthesis + task dummies")
    print("=" * 80)
    for name, b, l, h in zip(cols_fe, fe_beta, fe_lo, fe_hi):
        sig = "*" if (l > 0 or h < 0) else ""
        print(f"  {name:20s} {b:+.4f}  [{l:+.4f}, {h:+.4f}] {sig}")
    print(f"  sigma^2 (OLS resid): {fe_sigma2:.6f}")
    print()

    summary = pd.DataFrame(
        [
            {
                "model": "random_intercept",
                "term": term,
                "estimate": est,
                "ci_lo": l,
                "ci_hi": h,
                "sigma2": mixed.sigma2,
                "tau2": mixed.tau2,
                "loglik": mixed.loglik,
            }
            for term, est, l, h in zip(cols_re, mixed.beta, lo, hi)
        ]
    )
    summary_path = os.path.join(OUT_DIR, "2x2_hierarchical.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

# autorace/model/evaluate.py（Gemini作成、Claude確認）
from __future__ import annotations

import numpy as np
import pandas as pd

from autorace.model.features import RACE_KEYS


def _compute_metrics_subset(df: pd.DataFrame, n_bins: int = 20, eps: float = 1e-15) -> dict:
    """欠場を除いた行で、1着確率の LogLoss・ブライア・ECE(20区間) を計算する"""
    if df.empty:
        return {"num_races": 0, "num_runners": 0, "binary_logloss": None,
                "race_multiclass_logloss": None, "brier_score": None, "ece_20": None}
    p = np.clip(df["pred_prob"].to_numpy(dtype=float), eps, 1.0 - eps)
    y = df["is_win"].to_numpy(dtype=float)
    binary_logloss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    winners = df[df["is_win"] == 1.0]
    race_logloss = float(-np.mean(np.log(np.clip(winners["pred_prob"].to_numpy(dtype=float), eps, 1.0)))) if not winners.empty else None
    brier_score = float(np.mean((p - y) ** 2))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p, bin_edges[1:-1], right=False)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.any():
            ece += (mask.sum() / len(p)) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {"num_races": int(df.groupby(RACE_KEYS, sort=False).ngroups), "num_runners": int(len(p)),
            "binary_logloss": binary_logloss, "race_multiclass_logloss": race_logloss,
            "brier_score": brier_score, "ece_20": float(ece)}


def evaluate_predictions(pred_df: pd.DataFrame, race_result: pd.DataFrame, n_bins: int = 20) -> dict[str, dict]:
    """全レース（事故込み）と事故なしレースの2通りで精度指標を出す"""
    merged = pred_df.merge(race_result[RACE_KEYS + ["car_no", "order", "accident_code"]],
                           on=RACE_KEYS + ["car_no"], how="inner")
    if "is_absent" in merged.columns:
        merged = merged[~merged["is_absent"]].copy()
    merged["is_win"] = (merged["order"] == 1).astype(float)
    merged["_has_acc"] = merged["accident_code"].notna() & (merged["accident_code"].astype(str).str.strip() != "")
    race_has_accident = merged.groupby(RACE_KEYS, sort=False)["_has_acc"].transform("any")
    race_has_winner = merged.groupby(RACE_KEYS, sort=False)["is_win"].transform("sum") > 0
    valid_all = merged[race_has_winner].copy()
    valid_no_acc = valid_all[~race_has_accident[valid_all.index]].copy()
    return {"all_races": _compute_metrics_subset(valid_all, n_bins=n_bins),
            "no_accident_races": _compute_metrics_subset(valid_no_acc, n_bins=n_bins)}

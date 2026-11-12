# autorace/model/evaluate_v1.py （Gemini作成、Claude確認）
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from autorace.model.features import RACE_KEYS
from autorace.scripts.backtest_v0 import actual_rt3

ECE_BIN_EDGES: list[float] = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 1.000001]
ECE_BIN_LABELS: list[str] = ["0-1%", "1-2%", "2-3%", "3-4%", "4-5%", "5-6%", "6-7%", "7-8%", "8-9%", "9-10%", "10%以上"]


def compute_win_logloss(win_pred_df: pd.DataFrame, race_result: pd.DataFrame) -> tuple[float, int]:
    """レース単位の1着 LogLoss（1着が1人に決まるレースのみ）。戻り値: (logloss, レース数)"""
    if win_pred_df.empty or race_result.empty:
        return float("nan"), 0
    res = race_result.copy()
    res["order_num"] = pd.to_numeric(res.get("order"), errors="coerce")
    res["_is_win"] = (res["order_num"] == 1).astype(np.int32)
    win_cnt = res.groupby(RACE_KEYS, sort=False)["_is_win"].transform("sum")
    winners = res[(res["_is_win"] == 1) & (win_cnt == 1)][RACE_KEYS + ["car_no"]].drop_duplicates(subset=RACE_KEYS)
    if winners.empty:
        return float("nan"), 0
    merged = winners.merge(win_pred_df[RACE_KEYS + ["car_no", "pred_prob"]], on=RACE_KEYS + ["car_no"], how="inner")
    if merged.empty:
        return float("nan"), 0
    probs = np.clip(pd.to_numeric(merged["pred_prob"], errors="coerce").to_numpy(), 1e-15, 1.0)
    return float(-np.mean(np.log(probs))), int(len(probs))


def compute_rt3_logloss_and_hit(rt3_pred_df: pd.DataFrame, race_result: pd.DataFrame) -> tuple[float, float, int]:
    """3連単の的中組の LogLoss と、確率1位の組の的中率。戻り値: (logloss, 的中率, レース数)"""
    if rt3_pred_df.empty or race_result.empty:
        return float("nan"), float("nan"), 0
    act = actual_rt3(race_result)
    if act.empty:
        return float("nan"), float("nan"), 0
    act = act.rename(columns={"combo": "actual_combo"}).drop_duplicates(subset=RACE_KEYS)
    merged_hit = act.merge(rt3_pred_df[RACE_KEYS + ["combo", "combo_prob"]],
                           left_on=RACE_KEYS + ["actual_combo"], right_on=RACE_KEYS + ["combo"], how="inner")
    if merged_hit.empty:
        return float("nan"), float("nan"), 0
    hit_probs = np.clip(pd.to_numeric(merged_hit["combo_prob"], errors="coerce").to_numpy(), 1e-15, 1.0)
    rt3_ll = float(-np.mean(np.log(hit_probs)))

    sorted_pred = rt3_pred_df.sort_values(RACE_KEYS + ["combo_prob", "combo"], ascending=[True, True, True, False, True])
    top1 = (sorted_pred.drop_duplicates(subset=RACE_KEYS, keep="first")[RACE_KEYS + ["combo"]]
            .rename(columns={"combo": "top1_combo"}))
    top1_eval = act.merge(top1, on=RACE_KEYS, how="inner")
    top1_hit_rate = float(np.mean(top1_eval["actual_combo"] == top1_eval["top1_combo"]))
    return rt3_ll, top1_hit_rate, int(len(merged_hit))


def compute_rt3_ece(rt3_pred_df: pd.DataFrame, race_result: pd.DataFrame) -> tuple[float, list[dict[str, Any]]]:
    """3連単確率の ECE（0–1%,…,9–10%,10%以上 の11区間）と区間ごとの内訳。"""
    if rt3_pred_df.empty or race_result.empty:
        return float("nan"), []
    act = actual_rt3(race_result)
    if act.empty:
        return float("nan"), []
    act = act.rename(columns={"combo": "actual_combo"}).drop_duplicates(subset=RACE_KEYS)
    merged = rt3_pred_df[RACE_KEYS + ["combo", "combo_prob"]].merge(act, on=RACE_KEYS, how="inner")
    if merged.empty:
        return float("nan"), []
    probs = np.clip(pd.to_numeric(merged["combo_prob"], errors="coerce").to_numpy(), 0.0, 1.0)
    hits = (merged["combo"].to_numpy() == merged["actual_combo"].to_numpy()).astype(np.float64)
    bin_indices = np.clip(np.digitize(probs, ECE_BIN_EDGES, right=False) - 1, 0, len(ECE_BIN_LABELS) - 1)

    total_n = len(probs)
    ece = 0.0
    bins_report: list[dict[str, Any]] = []
    for b_idx, label in enumerate(ECE_BIN_LABELS):
        mask = bin_indices == b_idx
        cnt = int(np.sum(mask))
        mean_p = float(np.mean(probs[mask])) if cnt else 0.0
        act_rate = float(np.mean(hits[mask])) if cnt else 0.0
        gap = abs(mean_p - act_rate) if cnt else 0.0
        ece += (cnt / total_n) * gap
        bins_report.append({"bin": label, "count": cnt, "mean_pred": round(mean_p, 6),
                            "actual_rate": round(act_rate, 6), "abs_gap": round(gap, 6)})
    return float(ece), bins_report


def evaluate_predictions(win_pred_df: pd.DataFrame, rt3_pred_df: pd.DataFrame, race_result: pd.DataFrame) -> dict[str, Any]:
    win_ll, n_win_races = compute_win_logloss(win_pred_df, race_result)
    rt3_ll, rt3_top1_hit, n_rt3_races = compute_rt3_logloss_and_hit(rt3_pred_df, race_result)
    rt3_ece, ece_bins = compute_rt3_ece(rt3_pred_df, race_result)
    return {"n_win_races": n_win_races, "n_rt3_races": n_rt3_races, "win_logloss": win_ll,
            "rt3_logloss": rt3_ll, "rt3_top1_hit_rate": rt3_top1_hit, "rt3_ece": rt3_ece, "ece_bins": ece_bins}

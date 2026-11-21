# autorace/model/market.py （Gemini作成、Claude確認）
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from autorace.model.features import RACE_KEYS

EPS = 1e-15


def market_win_probs(odds_df: pd.DataFrame, gamma: float | None = None) -> pd.DataFrame:
    """単勝オッズ（bet_type='tns'）から市場の1着確率。gamma=None は 1/odds 正規化、数値はパワー法 p ∝ (1/odds)^gamma。
    戻り値: RACE_KEYS + [car_no, odds, market_prob]"""
    out_cols = list(RACE_KEYS) + ["car_no", "odds", "market_prob"]
    if odds_df is None or odds_df.empty:
        return pd.DataFrame(columns=out_cols)
    work = odds_df.copy()
    if "bet_type" in work.columns:
        work = work.loc[work["bet_type"] == "tns"].copy()
    if work.empty:
        return pd.DataFrame(columns=out_cols)
    if "car_no" not in work.columns:
        work["car_no"] = pd.to_numeric(work["combo"].astype(str).str.strip(), errors="coerce")
    else:
        work["car_no"] = pd.to_numeric(work["car_no"], errors="coerce")
    work["odds"] = pd.to_numeric(work["odds"], errors="coerce")
    valid = work["car_no"].notna() & work["odds"].notna() & np.isfinite(work["odds"].to_numpy(dtype=np.float64, na_value=np.nan)) & (work["odds"] > 0.0)
    work = work.loc[valid].copy()
    if work.empty:
        return pd.DataFrame(columns=out_cols)
    work["car_no"] = work["car_no"].astype(int)
    work = work.drop_duplicates(subset=list(RACE_KEYS) + ["car_no"], keep="last")

    inv_odds = 1.0 / work["odds"].to_numpy(dtype=np.float64)
    if gamma is None:
        raw_weight = inv_odds
    else:
        work["_log_inv"] = float(gamma) * np.log(np.maximum(inv_odds, EPS))
        max_log = work.groupby(list(RACE_KEYS), sort=False)["_log_inv"].transform("max")
        raw_weight = np.exp(work["_log_inv"].to_numpy(dtype=np.float64) - max_log.to_numpy(dtype=np.float64))
    work["_w"] = raw_weight
    sum_w = work.groupby(list(RACE_KEYS), sort=False)["_w"].transform("sum")
    work = work.loc[sum_w > 0.0].copy()
    work["market_prob"] = work["_w"] / work.groupby(list(RACE_KEYS), sort=False)["_w"].transform("sum")
    return work.sort_values(list(RACE_KEYS) + ["car_no"]).reset_index(drop=True)[out_cols]


def fit_market_gamma(odds_df: pd.DataFrame, race_result: pd.DataFrame, bounds: tuple[float, float] = (0.1, 3.0)) -> float:
    """単勝オッズのパワー法指数 gamma を1着 LogLoss 最小で推定。"""
    base = market_win_probs(odds_df, gamma=None)
    if base.empty or race_result is None or race_result.empty:
        return 1.0
    res = race_result[list(RACE_KEYS) + ["car_no", "order"]].copy()
    res["car_no"] = pd.to_numeric(res["car_no"], errors="coerce")
    res["order"] = pd.to_numeric(res["order"], errors="coerce")
    res = res.dropna(subset=["car_no"]).copy()
    res["car_no"] = res["car_no"].astype(int)
    res["is_win"] = (res["order"] == 1).astype(np.float64)
    merged = base.merge(res[list(RACE_KEYS) + ["car_no", "is_win"]], on=list(RACE_KEYS) + ["car_no"], how="inner")
    if merged.empty:
        return 1.0
    merged = merged.sort_values(list(RACE_KEYS) + ["car_no"]).reset_index(drop=True)
    g = merged.groupby(list(RACE_KEYS), sort=False)
    merged = merged.loc[(g["is_win"].transform("sum") > 0.0) & (g["car_no"].transform("count") >= 2)].copy()
    if merged.empty:
        return 1.0
    merged["target"] = merged["is_win"] / merged.groupby(list(RACE_KEYS), sort=False)["is_win"].transform("sum")
    race_codes, _ = pd.factorize(merged[list(RACE_KEYS)].astype(str).agg("|".join, axis=1), sort=False)
    starts = np.r_[0, np.flatnonzero(np.diff(race_codes)) + 1]
    counts = np.diff(np.r_[starts, len(race_codes)])
    n_races = len(starts)
    log_inv_odds = -np.log(np.maximum(merged["odds"].to_numpy(dtype=np.float64), EPS))
    target = merged["target"].to_numpy(dtype=np.float64)

    def nll(gamma_val: float) -> float:
        u = float(gamma_val) * log_inv_odds
        u_max = np.repeat(np.maximum.reduceat(u, starts), counts)
        exp_u = np.exp(u - u_max)
        sum_exp = np.repeat(np.add.reduceat(exp_u, starts), counts)
        log_p = (u - u_max) - np.log(np.maximum(sum_exp, EPS))
        return float(-np.sum(target * log_p) / n_races)

    return float(minimize_scalar(nll, bounds=bounds, method="bounded").x)


def market_rt3_probs(odds_df: pd.DataFrame) -> pd.DataFrame:
    """3連単オッズから市場の3連単確率（1/オッズ正規化）。戻り値: RACE_KEYS + [bet_type, combo, odds, market_prob]"""
    out_cols = list(RACE_KEYS) + ["bet_type", "combo", "odds", "market_prob"]
    if odds_df is None or odds_df.empty:
        return pd.DataFrame(columns=out_cols)
    work = odds_df.copy()
    if "bet_type" in work.columns:
        work = work.loc[work["bet_type"] == "rt3"].copy()
    else:
        work["bet_type"] = "rt3"
    if work.empty:
        return pd.DataFrame(columns=out_cols)
    work["combo"] = work["combo"].astype(str).str.strip()
    work["odds"] = pd.to_numeric(work["odds"], errors="coerce")
    valid = work["odds"].notna() & np.isfinite(work["odds"].to_numpy(dtype=np.float64, na_value=np.nan)) & (work["odds"] > 0.0) & (work["combo"] != "")
    work = work.loc[valid].copy()
    if work.empty:
        return pd.DataFrame(columns=out_cols)
    work = work.drop_duplicates(subset=list(RACE_KEYS) + ["combo"], keep="last")
    work["_w"] = 1.0 / work["odds"].to_numpy(dtype=np.float64)
    work["market_prob"] = work["_w"] / work.groupby(list(RACE_KEYS), sort=False)["_w"].transform("sum")
    work["bet_type"] = "rt3"
    return work.sort_values(list(RACE_KEYS) + ["combo"]).reset_index(drop=True)[out_cols]


def pick_snapshot(odds_pre_df: pd.DataFrame, how: str = "latest", minutes: float = 5.0) -> pd.DataFrame:
    """試走後〜締切前のオッズ履歴から、レース・券種・組ごとに1時点を選ぶ。
    latest: 最新 / earliest: 試走後最初 / min_before_close: 締切まで minutes 分以上ある中で最新
    （締切までの分が全部欠損の組は earliest、全部 minutes 未満の組は除外）"""
    if how not in ("latest", "earliest", "min_before_close"):
        raise ValueError(f"Unsupported snapshot selection mode: {how!r}")
    if odds_pre_df is None or odds_pre_df.empty:
        return odds_pre_df.copy() if odds_pre_df is not None else pd.DataFrame()
    orig_cols = list(odds_pre_df.columns)
    work = odds_pre_df.copy()
    group_cols = list(RACE_KEYS) + ["bet_type", "combo"]
    work["_ts"] = pd.to_datetime(work["fetched_at"], utc=True, errors="coerce") if "fetched_at" in work.columns else pd.NaT
    work["_mtc"] = pd.to_numeric(work["minutes_to_close"], errors="coerce") if "minutes_to_close" in work.columns else np.nan
    work["_row_ord"] = np.arange(len(work), dtype=np.int64)
    sorted_work = work.sort_values(by=["_ts", "_mtc", "_row_ord"], ascending=[True, False, True], na_position="last")
    if how == "earliest":
        picked = sorted_work.drop_duplicates(subset=group_cols, keep="first")
    elif how == "latest":
        picked = sorted_work.drop_duplicates(subset=group_cols, keep="last")
    else:
        has_any_mtc = sorted_work.groupby(group_cols, sort=False)["_mtc"].transform(lambda s: s.notna().any())
        picked_known = sorted_work.loc[has_any_mtc & (sorted_work["_mtc"] >= float(minutes))].drop_duplicates(subset=group_cols, keep="last")
        picked_unknown = sorted_work.loc[~has_any_mtc].drop_duplicates(subset=group_cols, keep="first")
        picked = pd.concat([picked_known, picked_unknown], ignore_index=True)
    return picked.sort_values(list(RACE_KEYS) + ["bet_type", "combo"]).reset_index(drop=True)[orig_cols]

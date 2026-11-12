# autorace/model/features_v1.py （Gemini作成、Claude確認）
from __future__ import annotations

import numpy as np
import pandas as pd

from autorace.model.features import RACE_KEYS, build_features

CATEGORICAL_COLS: list[str] = [
    "place_code",
    "grade_code",
    "race_type_code",
]

FEATURE_COLS: list[str] = [
    # 既存v0特徴量
    "max_handicap_diff",
    "same_handicap_inner_order",
    "same_handicap_count",
    "trial_rank_in_race",
    "trial_diff_from_mean",
    "rank_score",
    "rank_order_in_race",
    "prior_win_rate",
    "prior_top3_rate",
    "prior_races",
    # 優先度1〜8：物理換算ゴール時間・試走偏差・壁・スタート
    "diff_exp_time_raw",
    "diff_exp_time_adj",
    "prior_time_gap_10",
    "diff_trial_from_best",
    "is_retrial",
    "num_cars_ahead",
    "rel_inner_order",
    "wall_slow_cars_ahead",
    "wall_best_trial_diff_ahead",
    "prior_st_10",
    "effective_start_pos",
    "diff_effective_start_pos",
    "effective_start_order",
    "race_max_handicap_span",
    "rel_handicap",
    "road_temp_clean",
    "road_temp_x_handicap",
    # 湿走路・走路適性
    "race_min_trial_time",
    "is_non_good_track",
    "wet_order_diff",
    "prior_wet_races",
    # 節間の調子（同節・前日以前）
    "meeting_prior_races",
    "meeting_prior_avg_order",
    "meeting_trial_diff",
    # レース属性
    "place_code",
    "grade_code",
    "paragraph_day",
    "race_type_code",
    "active_car_count",
]


def _classify_race_name(series: pd.Series) -> pd.Series:
    """race_name を 0:その他, 1:予選, 2:準決, 3:優勝戦 に区分する。"""
    s = series.fillna("").astype(str)
    out = np.zeros(len(s), dtype=np.int32)
    out = np.where(s.str.contains("予選", na=False), 1, out)
    out = np.where(s.str.contains("準決", na=False), 2, out)
    out = np.where(s.str.contains("優勝", na=False), 3, out)
    return pd.Series(out, index=series.index, dtype=np.int32)


def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    df["race_date"] = df["race_date"].astype(str)
    df["place_code"] = pd.to_numeric(df["place_code"], errors="coerce").fillna(0).astype(np.int32)
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(np.int32)
    return df


def _compute_player_prior_rolling(
    program_df: pd.DataFrame,
    race_result_hist: pd.DataFrame | None,
    race_info: pd.DataFrame | None,
    rolling_window: int = 10,
) -> pd.DataFrame:
    """選手ごとの過去集計（試走とレースの乖離・平均ST・湿走路適性）。
    merge_asof(allow_exact_matches=False) で、開催日より厳密に前の日付の結果だけを使う。"""
    base_cols = ["prior_time_gap_10", "prior_st_10", "wet_order_diff", "prior_wet_races"]
    if race_result_hist is None or race_result_hist.empty:
        res = pd.DataFrame(index=program_df.index)
        res["prior_time_gap_10"] = 0.0
        res["prior_st_10"] = 0.15
        res["wet_order_diff"] = 0.0
        res["prior_wet_races"] = 0.0
        return res

    hist = _norm_keys(race_result_hist.copy())
    hist["player_code"] = pd.to_numeric(hist["player_code"], errors="coerce").fillna(-1).astype(np.int64)

    # 試走ごろの走路状態（situation_code）を結合（結果側の race_situation_code は使わない）
    if race_info is not None and not race_info.empty and "situation_code" in race_info.columns:
        info_sub = _norm_keys(race_info[RACE_KEYS + ["situation_code"]].drop_duplicates(subset=RACE_KEYS).copy())
        hist = hist.merge(info_sub, on=RACE_KEYS, how="left")
    else:
        hist["situation_code"] = np.nan

    trial_t = pd.to_numeric(hist.get("trial_time"), errors="coerce")
    race_t = pd.to_numeric(hist.get("race_time"), errors="coerce")
    st_val = pd.to_numeric(hist.get("st"), errors="coerce")
    order_val = pd.to_numeric(hist.get("order"), errors="coerce")

    hist["valid_time_gap"] = np.where((trial_t > 0) & (race_t > 0), race_t - trial_t, np.nan)
    hist["valid_st"] = np.where((st_val > 0) & (st_val < 1.0), st_val, np.nan)
    hist["valid_order"] = np.where((order_val >= 1) & (order_val <= 8), order_val, np.nan)

    # race_info が無い過去分は、レース内最小試走 >= 3.45 を湿走路の目安にする
    hist["_valid_trial"] = np.where(trial_t > 0, trial_t, np.nan)
    race_min_tr = hist.groupby(RACE_KEYS, sort=False)["_valid_trial"].transform("min")
    sit_code = pd.to_numeric(hist["situation_code"], errors="coerce")
    is_wet_hist = np.where(sit_code.notna(), sit_code != 0, race_min_tr >= 3.45)
    hist["is_wet_hist"] = is_wet_hist.astype(bool)

    hist["wet_order"] = np.where(hist["is_wet_hist"], hist["valid_order"], np.nan)
    hist["good_order"] = np.where(~hist["is_wet_hist"], hist["valid_order"], np.nan)
    hist["wet_count_flag"] = np.where(hist["is_wet_hist"] & hist["valid_order"].notna(), 1.0, 0.0)

    hist = hist.sort_values(["player_code", "race_date", "place_code", "race_no"]).reset_index(drop=True)
    grp = hist.groupby("player_code", sort=False)

    hist["prior_time_gap_10"] = grp["valid_time_gap"].transform(lambda s: s.rolling(rolling_window, min_periods=1).mean())
    hist["prior_st_10"] = grp["valid_st"].transform(lambda s: s.rolling(rolling_window, min_periods=1).mean())
    wet_mean = grp["wet_order"].transform(lambda s: s.expanding(min_periods=1).mean())
    good_mean = grp["good_order"].transform(lambda s: s.expanding(min_periods=1).mean())
    hist["wet_order_diff"] = (wet_mean - good_mean).fillna(0.0)
    hist["prior_wet_races"] = grp["wet_count_flag"].transform(lambda s: s.cumsum())

    # 選手・日ごとの最終状態を代表値にする
    daily_snap = hist.groupby(["player_code", "race_date"], as_index=False)[base_cols].last().copy()
    daily_snap["race_date_dt"] = pd.to_datetime(daily_snap["race_date"])
    daily_snap = daily_snap.sort_values(["race_date_dt", "player_code"]).reset_index(drop=True)

    left = program_df[["player_code", "race_date"]].copy()
    left["_orig_idx"] = np.arange(len(left))
    left["player_code"] = pd.to_numeric(left["player_code"], errors="coerce").fillna(-1).astype(np.int64)
    left["race_date_dt"] = pd.to_datetime(left["race_date"].astype(str))
    left_sorted = left.sort_values(["race_date_dt", "player_code"]).reset_index(drop=True)

    merged = pd.merge_asof(
        left_sorted,
        daily_snap[["player_code", "race_date_dt"] + base_cols],
        on="race_date_dt", by="player_code", direction="backward", allow_exact_matches=False,
    )
    merged = merged.sort_values("_orig_idx").reset_index(drop=True)
    merged.index = program_df.index

    merged["prior_time_gap_10"] = merged["prior_time_gap_10"].fillna(0.0)
    merged["prior_st_10"] = merged["prior_st_10"].fillna(0.15)
    merged["wet_order_diff"] = merged["wet_order_diff"].fillna(0.0)
    merged["prior_wet_races"] = merged["prior_wet_races"].fillna(0.0)
    return merged[base_cols]


def _compute_meeting_features(
    df: pd.DataFrame, race_result_hist: pd.DataFrame | None, race_info: pd.DataFrame | None,
) -> pd.DataFrame:
    """同じ節（同場・同 period_start_date）の前日までの出走数・平均着順・試走平均との差。"""
    cols = ["meeting_prior_races", "meeting_prior_avg_order", "meeting_trial_diff"]

    def _empty() -> pd.DataFrame:
        res = pd.DataFrame(index=df.index)
        res["meeting_prior_races"] = 0.0
        res["meeting_prior_avg_order"] = 4.5
        res["meeting_trial_diff"] = 0.0
        return res

    if (race_result_hist is None or race_result_hist.empty or "period_start_date" not in df.columns
            or df["period_start_date"].isna().all()):
        return _empty()

    hist = _norm_keys(race_result_hist.copy())
    hist["player_code"] = pd.to_numeric(hist["player_code"], errors="coerce").fillna(-1).astype(np.int64)

    if race_info is not None and not race_info.empty and "period_start_date" in race_info.columns:
        info_p = _norm_keys(race_info[RACE_KEYS + ["period_start_date"]].drop_duplicates(subset=RACE_KEYS).copy())
        hist = hist.merge(info_p, on=RACE_KEYS, how="left")
    else:
        hist["period_start_date"] = np.nan

    hist = hist[hist["period_start_date"].notna()].copy()
    if hist.empty:
        return _empty()

    hist["period_start_date"] = hist["period_start_date"].astype(str)
    order_val = pd.to_numeric(hist.get("order"), errors="coerce")
    trial_val = pd.to_numeric(hist.get("trial_time"), errors="coerce")
    hist["valid_order"] = np.where((order_val >= 1) & (order_val <= 8), order_val, np.nan)
    hist["valid_trial"] = np.where(trial_val > 0, trial_val, np.nan)
    hist["start_flag"] = 1.0

    hist = hist.sort_values(["place_code", "period_start_date", "player_code", "race_date", "race_no"]).reset_index(drop=True)
    m_grp = hist.groupby(["place_code", "period_start_date", "player_code"], sort=False)
    hist["meeting_prior_races"] = m_grp["start_flag"].transform("cumsum")
    hist["meeting_prior_avg_order"] = m_grp["valid_order"].transform(lambda s: s.expanding(min_periods=1).mean())
    hist["meeting_prior_avg_trial"] = m_grp["valid_trial"].transform(lambda s: s.expanding(min_periods=1).mean())

    daily_m = (
        hist.groupby(["place_code", "period_start_date", "player_code", "race_date"], as_index=False)[
            ["meeting_prior_races", "meeting_prior_avg_order", "meeting_prior_avg_trial"]].last().copy()
    )
    daily_m["race_date_dt"] = pd.to_datetime(daily_m["race_date"])
    daily_m = daily_m.sort_values(["race_date_dt", "place_code", "player_code"]).reset_index(drop=True)

    left = df[["place_code", "period_start_date", "player_code", "race_date", "_valid_trial"]].copy()
    left["_orig_idx"] = np.arange(len(left))
    left["place_code"] = pd.to_numeric(left["place_code"], errors="coerce").fillna(0).astype(np.int32)
    left["period_start_date"] = left["period_start_date"].fillna("").astype(str)
    left["player_code"] = pd.to_numeric(left["player_code"], errors="coerce").fillna(-1).astype(np.int64)
    left["race_date_dt"] = pd.to_datetime(left["race_date"].astype(str))
    left_sorted = left.sort_values(["race_date_dt", "place_code", "player_code"]).reset_index(drop=True)

    merged = pd.merge_asof(
        left_sorted,
        daily_m[["place_code", "period_start_date", "player_code", "race_date_dt",
                 "meeting_prior_races", "meeting_prior_avg_order", "meeting_prior_avg_trial"]],
        on="race_date_dt", by=["place_code", "period_start_date", "player_code"],
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.sort_values("_orig_idx").reset_index(drop=True)
    merged.index = df.index

    merged["meeting_prior_races"] = merged["meeting_prior_races"].fillna(0.0)
    merged["meeting_prior_avg_order"] = merged["meeting_prior_avg_order"].fillna(4.5)
    merged["meeting_trial_diff"] = (merged["_valid_trial"] - merged["meeting_prior_avg_trial"]).fillna(0.0)
    return merged[cols]


def _compute_intra_race_wall_and_ahead(df: pd.DataFrame) -> pd.DataFrame:
    """レース内（出走車のみ）で、前を走る車の台数・前にいる自分より試走が遅い車の台数・
    前の車の最速試走−自分の試走 を計算する。"""
    n = len(df)
    num_cars_ahead = np.zeros(n, dtype=np.float64)
    wall_slow_cars_ahead = np.zeros(n, dtype=np.float64)
    wall_best_trial_diff_ahead = np.zeros(n, dtype=np.float64)

    handicaps = df["_handicap_num"].to_numpy(dtype=np.float64)
    car_nos = df["_car_no_num"].to_numpy(dtype=np.float64)
    trials = df["_valid_trial"].to_numpy(dtype=np.float64)
    absent = df["is_absent"].to_numpy(dtype=bool)

    for idx in df.groupby(RACE_KEYS, sort=False).indices.values():
        if len(idx) == 0:
            continue
        h, c, t, active = handicaps[idx], car_nos[idx], trials[idx], ~absent[idx]
        # j が i より前：出走していて、ハンデが小さい か 同ハンデで内側（車番が小さい）
        ahead_mask = active[None, :] & ((h[None, :] < h[:, None]) | ((h[None, :] == h[:, None]) & (c[None, :] < c[:, None])))
        ahead_cnt = ahead_mask.sum(axis=1).astype(np.float64)
        valid_pair = ahead_mask & (~np.isnan(t[:, None])) & (~np.isnan(t[None, :]))
        slow_cnt = (valid_pair & (t[None, :] > t[:, None])).sum(axis=1).astype(np.float64)
        t_ahead = np.where(ahead_mask & (~np.isnan(t[None, :])), t[None, :], np.inf)
        min_t_ahead = np.min(t_ahead, axis=1)
        has_valid_ahead = np.isfinite(min_t_ahead) & (~np.isnan(t))
        best_diff = np.where(has_valid_ahead, min_t_ahead - t, 0.0)

        num_cars_ahead[idx] = np.where(active, ahead_cnt, 0.0)
        wall_slow_cars_ahead[idx] = np.where(active, slow_cnt, 0.0)
        wall_best_trial_diff_ahead[idx] = np.where(active, best_diff, 0.0)

    return pd.DataFrame({"num_cars_ahead": num_cars_ahead, "wall_slow_cars_ahead": wall_slow_cars_ahead,
                         "wall_best_trial_diff_ahead": wall_best_trial_diff_ahead}, index=df.index)


def build_features_v1(
    program: pd.DataFrame,
    race_info: pd.DataFrame | None,
    race_result_hist: pd.DataFrame | None,
    window: int = 20,
) -> pd.DataFrame:
    """v0 の build_features の上に v1 用の特徴量を追加する。1行＝1出走（欠車は is_absent=True で残す）。
    race_info の race_* 系（レース時の値）は使わない。"""
    df = _norm_keys(build_features(program, race_result_hist, window=window).copy())

    # 発走前に分かる race_info の列だけを結合
    allowed_info_cols = ["distance", "situation_code", "temp", "humid", "road_temp", "weather", "grade_code",
                         "paragraph_day", "period_start_date", "race_name", "final_race_no"]
    if race_info is not None and not race_info.empty:
        avail_cols = [c for c in allowed_info_cols if c in race_info.columns]
        info_clean = _norm_keys(race_info[RACE_KEYS + avail_cols].drop_duplicates(subset=RACE_KEYS).copy())
        drop_overlap = [c for c in avail_cols if c in df.columns]
        if drop_overlap:
            df = df.drop(columns=drop_overlap)
        df = df.merge(info_clean, on=RACE_KEYS, how="left")

    for col, default_val in [("distance", 3100.0), ("situation_code", 0), ("road_temp", 25.0), ("grade_code", 0),
                             ("paragraph_day", 1.0), ("race_name", ""), ("period_start_date", np.nan)]:
        if col not in df.columns:
            df[col] = default_val

    df["is_absent"] = df["is_absent"].fillna(False).astype(bool)
    active_mask = ~df["is_absent"]

    df["_handicap_num"] = pd.to_numeric(df["handicap"], errors="coerce").fillna(0.0)
    df["_car_no_num"] = pd.to_numeric(df["car_no"], errors="coerce").fillna(0.0)
    raw_trial = pd.to_numeric(df["trial_run_time"], errors="coerce")
    df["_valid_trial"] = np.where(active_mask & (raw_trial > 0), raw_trial, np.nan)

    distance_num = pd.to_numeric(df["distance"], errors="coerce").fillna(3100.0)
    distance_num = np.where(distance_num > 0, distance_num, 3100.0)

    # 1. 選手ごとの過去集計
    prior_df = _compute_player_prior_rolling(df, race_result_hist, race_info, rolling_window=10)
    for col in prior_df.columns:
        df[col] = prior_df[col]

    # 2. 予想ゴールタイム（試走そのまま／試走とレースの乖離で補正）とレース最小値との差
    df["exp_time"] = df["_valid_trial"] * (distance_num + df["_handicap_num"]) / 100.0
    df["exp_time_adj"] = (df["_valid_trial"] + df["prior_time_gap_10"]) * (distance_num + df["_handicap_num"]) / 100.0
    race_min_exp = df.groupby(RACE_KEYS, sort=False)["exp_time"].transform("min")
    race_min_exp_adj = df.groupby(RACE_KEYS, sort=False)["exp_time_adj"].transform("min")
    df["diff_exp_time_raw"] = (df["exp_time"] - race_min_exp).fillna(0.5)
    df["diff_exp_time_adj"] = (df["exp_time_adj"] - race_min_exp_adj).fillna(0.5)

    # 3. レース内最小試走・試走差・再試走フラグ
    race_min_trial = df.groupby(RACE_KEYS, sort=False)["_valid_trial"].transform("min")
    df["race_min_trial_time"] = race_min_trial.fillna(3.35)
    df["diff_trial_from_best"] = (df["_valid_trial"] - race_min_trial).fillna(0.10)
    if "trial_retry_code" in df.columns:
        retry_s = df["trial_retry_code"].fillna("").astype(str).str.strip()
        df["is_retrial"] = (~retry_s.isin(["", "0", "0.0", "None", "nan"])).astype(np.float64)
    else:
        df["is_retrial"] = 0.0

    # 4. 前を走る車の台数・同ハンデ内の内外・壁
    wall_df = _compute_intra_race_wall_and_ahead(df)
    for col in wall_df.columns:
        df[col] = wall_df[col]
    sh_cnt = pd.to_numeric(df["same_handicap_count"], errors="coerce").fillna(1.0).clip(lower=1.0)
    sh_ord = pd.to_numeric(df["same_handicap_inner_order"], errors="coerce").fillna(0.0)
    df["rel_inner_order"] = sh_ord / sh_cnt

    # 5. 実効スタート位置（handicap + 平均ST×15）
    df["effective_start_pos"] = np.where(active_mask, df["_handicap_num"] + df["prior_st_10"] * 15.0, np.nan)
    min_eff_pos = df.groupby(RACE_KEYS, sort=False)["effective_start_pos"].transform("min")
    df["diff_effective_start_pos"] = (df["effective_start_pos"] - min_eff_pos).fillna(0.0)
    df["effective_start_order"] = (df.groupby(RACE_KEYS, sort=False)["effective_start_pos"]
                                   .rank(method="min", ascending=True).fillna(8.0))
    df["effective_start_pos"] = df["effective_start_pos"].fillna(df["_handicap_num"] + 2.25)

    # 6. ハンデの広がりと相対ハンデ
    df["_active_handicap"] = np.where(active_mask, df["_handicap_num"], np.nan)
    h_min = df.groupby(RACE_KEYS, sort=False)["_active_handicap"].transform("min").fillna(0.0)
    h_max = df.groupby(RACE_KEYS, sort=False)["_active_handicap"].transform("max").fillna(0.0)
    h_span = (h_max - h_min).clip(lower=0.0)
    df["race_max_handicap_span"] = h_span
    df["rel_handicap"] = np.where(h_span > 0, (df["_handicap_num"] - h_min) / h_span, 0.0)

    # 7. 走路温度×ハンデ
    road_t = pd.to_numeric(df["road_temp"], errors="coerce").fillna(25.0)
    df["road_temp_clean"] = road_t
    df["road_temp_x_handicap"] = road_t * df["_handicap_num"]

    # 8. 走路状態（試走ごろ）
    sit_num = pd.to_numeric(df["situation_code"], errors="coerce").fillna(0)
    df["is_non_good_track"] = (sit_num != 0).astype(np.float64)

    # 9. 節間の調子
    meeting_df = _compute_meeting_features(df, race_result_hist, race_info)
    for col in meeting_df.columns:
        df[col] = meeting_df[col]

    # 10. レース属性・出走数
    df["grade_code"] = pd.to_numeric(df["grade_code"], errors="coerce").fillna(0).astype(np.int32)
    df["paragraph_day"] = pd.to_numeric(df["paragraph_day"], errors="coerce").fillna(1.0).astype(np.float64)
    df["race_type_code"] = _classify_race_name(df["race_name"])
    df["_active_flag"] = active_mask.astype(np.int32)
    df["active_car_count"] = df.groupby(RACE_KEYS, sort=False)["_active_flag"].transform("sum").astype(np.float64)

    df = df.drop(columns=[c for c in ["_handicap_num", "_car_no_num", "_valid_trial", "_active_handicap", "_active_flag"]
                          if c in df.columns])
    for cat_col in CATEGORICAL_COLS:
        df[cat_col] = pd.to_numeric(df[cat_col], errors="coerce").fillna(0).astype(np.int32)
    return df

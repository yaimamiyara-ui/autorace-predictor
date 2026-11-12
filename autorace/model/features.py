# autorace/model/features.py（Gemini作成、Claude確認）
from __future__ import annotations

import numpy as np
import pandas as pd

RACE_KEYS = ["place_code", "race_date", "race_no"]


def parse_rank_series(rank_ser: pd.Series) -> pd.Series:
    """'S-12', 'A-190', 'B-117' を通し順位の数値に変換する（S:0+番号, A:1000+, B:2000+, 不明:3000）"""
    extracted = rank_ser.fillna("").astype(str).str.extract(r"^([SAB])-(\d+)$")
    tier = extracted[0]
    num = pd.to_numeric(extracted[1], errors="coerce")
    offset = tier.map({"S": 0.0, "A": 1000.0, "B": 2000.0})
    return (offset + num).fillna(3000.0).astype(float)


def compute_PriorPlayerStats(race_result: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """選手ごとの直近N走の1着率・3着内率（その日の終了時点）。使う側で race_date より厳密に前の日付を結合する。"""
    cols = ["player_code", "stat_date", "prior_win_rate", "prior_top3_rate", "prior_races"]
    if race_result.empty:
        return pd.DataFrame(columns=cols)

    df = race_result.copy()
    has_order = df["order"].notna()
    has_acc = df["accident_code"].notna() & (df["accident_code"].astype(str) != "")
    df = df[has_order | has_acc].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["race_no_num"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0)
    df = df.sort_values(["player_code", "race_date", "place_code", "race_no_num"]).reset_index(drop=True)
    df["is_win"] = (df["order"] == 1).astype(float)
    df["is_top3"] = df["order"].isin([1, 2, 3]).astype(float)

    grp = df.groupby("player_code", sort=False)
    df["cum_win_rate"] = grp["is_win"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    df["cum_top3_rate"] = grp["is_top3"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    df["cum_races"] = grp["is_win"].transform(lambda s: s.rolling(window, min_periods=1).count())

    daily_last = (
        df.groupby(["player_code", "race_date"], as_index=False).last()
        [["player_code", "race_date", "cum_win_rate", "cum_top3_rate", "cum_races"]]
        .rename(columns={"race_date": "stat_date", "cum_win_rate": "prior_win_rate",
                         "cum_top3_rate": "prior_top3_rate", "cum_races": "prior_races"})
    )
    return daily_last.sort_values(["stat_date", "player_code"]).reset_index(drop=True)


def build_features(program: pd.DataFrame, race_result: pd.DataFrame | None = None, window: int = 20) -> pd.DataFrame:
    """出走表と過去結果から特徴量を作る。集計は必ずそのレースの前日までのデータだけを使う。"""
    if program.empty:
        return program.copy()

    df = program.copy()
    df["car_no"] = pd.to_numeric(df["car_no"], errors="coerce").astype(int)
    df["handicap"] = pd.to_numeric(df["handicap"], errors="coerce").fillna(0.0)
    df["trial_run_time"] = pd.to_numeric(df["trial_run_time"], errors="coerce")
    df["is_absent"] = df["absent"].notna() & (df["absent"].astype(str).str.strip() != "")
    active_mask = ~df["is_absent"]

    active_df = df[active_mask].copy()
    if not active_df.empty:
        race_grp = active_df.groupby(RACE_KEYS, sort=False)
        max_h = race_grp["handicap"].transform("max")
        min_h = race_grp["handicap"].transform("min")
        active_df["max_handicap_diff"] = max_h - active_df["handicap"]
        active_df["is_all_same_handicap"] = (max_h == min_h).astype(int)

        active_df = active_df.sort_values(RACE_KEYS + ["handicap", "car_no"])
        h_grp = active_df.groupby(RACE_KEYS + ["handicap"], sort=False)
        active_df["same_handicap_inner_order"] = h_grp.cumcount() + 1
        active_df["same_handicap_count"] = h_grp["car_no"].transform("count")

        active_df["valid_trial_time"] = active_df["trial_run_time"].where(active_df["trial_run_time"] > 0, np.nan)
        r_grp = active_df.groupby(RACE_KEYS, sort=False)
        active_df["trial_rank_in_race"] = r_grp["valid_trial_time"].rank(ascending=True, method="min", na_option="bottom")
        trial_mean = r_grp["valid_trial_time"].transform("mean")
        active_df["trial_diff_from_mean"] = (active_df["valid_trial_time"] - trial_mean).fillna(0.0)

        active_df["rank_score"] = parse_rank_series(active_df["rank"])
        active_df["rank_order_in_race"] = r_grp["rank_score"].rank(ascending=True, method="min", na_option="bottom")

        merge_cols = ["place_code", "race_date", "race_no", "car_no", "max_handicap_diff", "is_all_same_handicap",
                      "same_handicap_inner_order", "same_handicap_count", "trial_rank_in_race",
                      "trial_diff_from_mean", "rank_score", "rank_order_in_race"]
        df = df.merge(active_df[merge_cols], on=["place_code", "race_date", "race_no", "car_no"], how="left")
    else:
        df["max_handicap_diff"] = 0.0
        df["is_all_same_handicap"] = 1
        df["same_handicap_inner_order"] = 1
        df["same_handicap_count"] = 1
        df["trial_rank_in_race"] = 1.0
        df["trial_diff_from_mean"] = 0.0
        df["rank_score"] = parse_rank_series(df["rank"])
        df["rank_order_in_race"] = 1.0

    df["max_handicap_diff"] = df["max_handicap_diff"].fillna(0.0)
    df["is_all_same_handicap"] = df["is_all_same_handicap"].fillna(0).astype(int)
    df["same_handicap_inner_order"] = df["same_handicap_inner_order"].fillna(1).astype(int)
    df["same_handicap_count"] = df["same_handicap_count"].fillna(1).astype(int)
    df["rank_score"] = df["rank_score"].fillna(parse_rank_series(df["rank"]))
    df["rank_order_in_race"] = df["rank_order_in_race"].fillna(4.5)
    df["trial_rank_in_race"] = df["trial_rank_in_race"].fillna(4.5)
    df["trial_diff_from_mean"] = df["trial_diff_from_mean"].fillna(0.0)

    if race_result is not None and not race_result.empty:
        prior_stats = compute_PriorPlayerStats(race_result, window=window)
        if not prior_stats.empty:
            df["_orig_idx"] = np.arange(len(df))
            df["race_date_dt"] = pd.to_datetime(df["race_date"])
            prior_stats["stat_date_dt"] = pd.to_datetime(prior_stats["stat_date"])
            merged = pd.merge_asof(
                df.sort_values("race_date_dt"),
                prior_stats.sort_values("stat_date_dt")[["player_code", "stat_date_dt", "prior_win_rate", "prior_top3_rate", "prior_races"]],
                left_on="race_date_dt", right_on="stat_date_dt", by="player_code",
                direction="backward", allow_exact_matches=False,  # 厳密に前日以前のみ
            )
            df = merged.sort_values("_orig_idx").drop(columns=["_orig_idx", "race_date_dt", "stat_date_dt"])
        else:
            df["prior_win_rate"] = np.nan
            df["prior_top3_rate"] = np.nan
            df["prior_races"] = 0.0
    else:
        df["prior_win_rate"] = np.nan
        df["prior_top3_rate"] = np.nan
        df["prior_races"] = 0.0

    df["prior_races"] = df["prior_races"].fillna(0.0)
    return df.reset_index(drop=True)

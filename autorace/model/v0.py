# autorace/model/v0.py（Gemini作成、Claude確認：不要な行の整理のみ）
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from autorace.model.features import RACE_KEYS, build_features


@dataclass
class V0Params:
    alpha_shrinkage: float = 30.0   # ベイズ縮小の擬似サンプル数
    default_base_rate: float = 0.125
    beta_trial_rank: float = 0.22   # 試走順位による補正
    beta_trial_diff: float = 2.5    # 試走タイム差(秒)による補正
    beta_rank_order: float = 0.08   # ランク順位による補正
    beta_prior_win: float = 0.50    # 直近1着率による補正
    history_window: int = 20


class BaselineModelV0:
    def __init__(self, params: V0Params | None = None):
        self.params = params or V0Params()

    def predict_historical(self, program: pd.DataFrame, race_result: pd.DataFrame) -> pd.DataFrame:
        """各レースの前日までのデータだけで1着確率 pred_prob を出す"""
        if program.empty:
            res = program.copy()
            res["pred_prob"] = pd.Series(dtype=float)
            return res

        feat_df = build_features(program, race_result=race_result, window=self.params.history_window)

        res_clean = race_result[
            race_result["order"].notna()
            | (race_result["accident_code"].notna() & (race_result["accident_code"].astype(str) != ""))
        ].copy()

        merged_hist = feat_df[RACE_KEYS + ["car_no", "is_absent", "max_handicap_diff", "same_handicap_inner_order"]].merge(
            res_clean[RACE_KEYS + ["car_no", "order"]], on=RACE_KEYS + ["car_no"], how="inner")
        merged_hist = merged_hist[~merged_hist["is_absent"]].copy()
        merged_hist["is_win"] = (merged_hist["order"] == 1).astype(float)

        cell_cols = ["max_handicap_diff", "same_handicap_inner_order"]
        daily_cell = (merged_hist.groupby(["race_date"] + cell_cols, as_index=False)
                      .agg(daily_wins=("is_win", "sum"), daily_trials=("is_win", "count")))
        daily_global = (merged_hist.groupby("race_date", as_index=False)
                        .agg(g_wins=("is_win", "sum"), g_trials=("is_win", "count")))

        all_dates = sorted(feat_df["race_date"].unique())
        cell_cum_wins: dict[tuple[float, int], float] = {}
        cell_cum_trials: dict[tuple[float, int], float] = {}
        global_wins = 0.0
        global_trials = 0.0
        daily_cell_map = {d: grp for d, grp in daily_cell.groupby("race_date", sort=False)}
        daily_global_map = daily_global.set_index("race_date").to_dict("index")

        # 出走表の最初の日より前の結果があれば、それで事前に集計しておく
        if not res_clean.empty:
            prior_res = res_clean[res_clean["race_date"] < min(all_dates)].copy()
            if not prior_res.empty and "handicap" in prior_res.columns:
                prior_res["car_no"] = pd.to_numeric(prior_res["car_no"], errors="coerce").astype(int)
                prior_res["handicap"] = pd.to_numeric(prior_res["handicap"], errors="coerce").fillna(0.0)
                p_grp = prior_res.groupby(RACE_KEYS, sort=False)
                prior_res["max_handicap_diff"] = p_grp["handicap"].transform("max") - prior_res["handicap"]
                prior_res = prior_res.sort_values(RACE_KEYS + ["handicap", "car_no"])
                prior_res["same_handicap_inner_order"] = prior_res.groupby(RACE_KEYS + ["handicap"], sort=False).cumcount() + 1
                prior_res["is_win"] = (prior_res["order"] == 1).astype(float)
                for (hd, io), grp in prior_res.groupby(cell_cols):
                    key = (float(hd), int(io))
                    cell_cum_wins[key] = cell_cum_wins.get(key, 0.0) + float(grp["is_win"].sum())
                    cell_cum_trials[key] = cell_cum_trials.get(key, 0.0) + float(len(grp))
                global_wins += float(prior_res["is_win"].sum())
                global_trials += float(len(prior_res))

        out_chunks = []
        for d in all_dates:
            day_df = feat_df[feat_df["race_date"] == d].copy()
            g_mean = global_wins / global_trials if global_trials > 0 else self.params.default_base_rate

            keys = list(zip(day_df["max_handicap_diff"].astype(float), day_df["same_handicap_inner_order"].astype(int)))
            c_wins = np.array([cell_cum_wins.get(k, 0.0) for k in keys], dtype=float)
            c_trials = np.array([cell_cum_trials.get(k, 0.0) for k in keys], dtype=float)
            alpha = self.params.alpha_shrinkage
            base_prob = (c_wins + alpha * g_mean) / (c_trials + alpha)
            day_df["base_cell_prob"] = base_prob

            day_df["active_count"] = day_df.groupby(RACE_KEYS, sort=False)["is_absent"].transform(lambda s: (~s).sum())
            center_rank = (day_df["active_count"].clip(lower=1) + 1.0) / 2.0

            log_multiplier = (
                self.params.beta_trial_rank * (center_rank - day_df["trial_rank_in_race"])
                - self.params.beta_trial_diff * day_df["trial_diff_from_mean"]
                + self.params.beta_rank_order * (center_rank - day_df["rank_order_in_race"])
                + self.params.beta_prior_win * (day_df["prior_win_rate"].fillna(g_mean) - g_mean)
            )
            multiplier = np.exp(np.clip(log_multiplier, -5.0, 5.0))
            day_df["_raw_score"] = np.where(day_df["is_absent"], 0.0, base_prob * multiplier)
            race_sum = day_df.groupby(RACE_KEYS, sort=False)["_raw_score"].transform("sum")
            day_df["pred_prob"] = np.where(race_sum > 0, day_df["_raw_score"] / race_sum, 0.0)
            out_chunks.append(day_df.drop(columns=["_raw_score"]))

            # 当日の結果は翌日以降の予測にだけ使う
            if d in daily_cell_map:
                for row in daily_cell_map[d].itertuples(index=False):
                    k = (float(row.max_handicap_diff), int(row.same_handicap_inner_order))
                    cell_cum_wins[k] = cell_cum_wins.get(k, 0.0) + float(row.daily_wins)
                    cell_cum_trials[k] = cell_cum_trials.get(k, 0.0) + float(row.daily_trials)
            if d in daily_global_map:
                global_wins += float(daily_global_map[d]["g_wins"])
                global_trials += float(daily_global_map[d]["g_trials"])

        return pd.concat(out_chunks, ignore_index=True)

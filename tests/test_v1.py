# tests/test_v1.py （Gemini作成、Claude確認）
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from autorace.model.evaluate_v1 import evaluate_predictions
from autorace.model.features import RACE_KEYS
from autorace.model.features_v1 import FEATURE_COLS, build_features_v1
from autorace.model.v1 import ThreeStageModel
from autorace.scripts.train_v1 import attach_targets, generate_walk_forward_splits


def _generate_synthetic_autorace_data(n_days: int = 90, races_per_day: int = 3, seed: int = 42):
    """試走が速い（小さい）車ほど上位に来やすい人工データ。"""
    rng = np.random.default_rng(seed)
    start_dt = pd.Timestamp("2026-01-01")
    prog_rows, res_rows, info_rows = [], [], []
    handicaps = [0, 10, 10, 20, 20, 30, 30, 40]

    for d in range(n_days):
        r_date = (start_dt + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        period_start = (start_dt + pd.Timedelta(days=(d // 4) * 4)).strftime("%Y-%m-%d")
        p_day = (d % 4) + 1
        for r_no in range(1, races_per_day + 1):
            place_code = 2
            sit_code = 0 if rng.random() > 0.15 else 2
            road_temp = float(rng.integers(15, 45))
            info_rows.append({
                "place_code": place_code, "race_date": r_date, "race_no": r_no, "distance": 3100,
                "situation_code": sit_code, "temp": 20.0, "humid": 50.0, "road_temp": road_temp, "weather": "晴",
                "race_situation_code": sit_code, "race_road_temp": road_temp + 2.0, "grade_code": 0,
                "paragraph_day": p_day, "period_start_date": period_start,
                "race_name": "一般予選" if p_day <= 2 else ("準決勝戦" if p_day == 3 else "優勝戦"),
                "final_race_no": races_per_day,
            })
            trials, st_list, finish = [], [], []
            for car_no in range(1, 9):
                player_code = 100 + (r_no - 1) * 8 + car_no
                h = handicaps[car_no - 1]
                base_trial = 3.36 - (car_no * 0.012) + rng.normal(0.0, 0.025)
                if sit_code != 0:
                    base_trial += 0.30
                trial_t = round(float(base_trial), 2)
                st_val = round(float(rng.uniform(0.08, 0.22)), 2)
                race_t = round(float(trial_t + 0.06 + rng.normal(0.0, 0.008)), 3)
                trials.append(trial_t)
                st_list.append(st_val)
                finish.append(race_t * (3100.0 + h) / 100.0 + st_val)
                prog_rows.append({
                    "place_code": place_code, "race_date": r_date, "race_no": r_no, "player_code": player_code,
                    "car_no": car_no, "rank": f"S-{car_no}" if car_no >= 6 else f"A-{car_no * 10}", "handicap": h,
                    "trial_run_time": trial_t, "trial_retry_code": None, "absent": None, "race_dev": 0.06,
                    "rate2": 0.3, "rate3": 0.5, "age": 35, "bike_class": 1, "graduation_code": 30,
                    "branch_place_code": place_code,
                })
            orders = np.argsort(np.argsort(finish)) + 1
            for i, car_no in enumerate(range(1, 9)):
                res_rows.append({
                    "place_code": place_code, "race_date": r_date, "race_no": r_no,
                    "player_code": 100 + (r_no - 1) * 8 + car_no, "car_no": car_no, "order": int(orders[i]),
                    "accident_code": None, "handicap": handicaps[i], "trial_time": trials[i],
                    "race_time": round(float(finish[i] * 100.0 / (3100.0 + handicaps[i])), 3), "st": st_list[i],
                    "foul_code": None,
                })
    return pd.DataFrame(prog_rows), pd.DataFrame(info_rows), pd.DataFrame(res_rows)


class TestModelV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.program, cls.race_info, cls.race_result = _generate_synthetic_autorace_data()

    def test_no_future_or_sameday_leakage(self) -> None:
        """当日以降の結果を書き換えても、当日までのレースの特徴量が変わらない"""
        cutoff = "2026-02-15"
        orig = build_features_v1(self.program, self.race_info, self.race_result)
        mutated = self.race_result.copy()
        m = mutated["race_date"] >= cutoff
        mutated.loc[m, "order"] = 9 - mutated.loc[m, "order"]
        mutated.loc[m, "race_time"] = 4.99
        mutated.loc[m, "st"] = 0.99
        mut = build_features_v1(self.program, self.race_info, mutated)
        pd.testing.assert_frame_equal(
            orig[orig["race_date"] <= cutoff].reset_index(drop=True)[FEATURE_COLS],
            mut[mut["race_date"] <= cutoff].reset_index(drop=True)[FEATURE_COLS],
            check_exact=False, atol=1e-12, rtol=1e-12)

    def test_race_time_fields_not_used(self) -> None:
        """レース時の値（race_situation_code など）を変えても特徴量が変わらない"""
        info2 = self.race_info.copy()
        info2["race_situation_code"] = 9
        info2["race_road_temp"] = 99.0
        a = build_features_v1(self.program, self.race_info, self.race_result)[FEATURE_COLS]
        b = build_features_v1(self.program, info2, self.race_result)[FEATURE_COLS]
        pd.testing.assert_frame_equal(a, b)

    def test_walk_forward_splits(self) -> None:
        sp = generate_walk_forward_splits("2025-01-01", "2026-11-11")
        self.assertEqual(sp[0]["train_end"], "2025-12-31")
        self.assertEqual(sp[0]["test_start"], "2026-03-01")
        for s in sp:
            self.assertLess(s["train_end"], s["valid_start"])
            self.assertLess(s["valid_end"], s["test_start"])
        self.assertEqual(sp[-1]["test_end"], "2026-11-11")

    def test_train_predict_and_metrics(self) -> None:
        full_df = attach_targets(build_features_v1(self.program, self.race_info, self.race_result), self.race_result)
        tr_df = full_df[full_df["race_date"] <= "2026-02-19"]
        va_df = full_df[(full_df["race_date"] >= "2026-02-20") & (full_df["race_date"] <= "2026-03-05")]
        te_df = full_df[full_df["race_date"] >= "2026-03-06"]
        te_res = self.race_result[self.race_result["race_date"] >= "2026-03-06"]

        model = ThreeStageModel(num_boost_round=120, early_stopping_rounds=20).fit(tr_df, va_df)
        win_pred = model.predict_win(te_df)
        rt3_pred = model.predict_rt3(te_df)

        for s in win_pred.groupby(RACE_KEYS)["pred_prob"].sum().to_numpy():
            self.assertAlmostEqual(float(s), 1.0, places=6)
        g = rt3_pred.groupby(RACE_KEYS)["combo_prob"]
        for s, c in zip(g.sum().to_numpy(), g.count().to_numpy()):
            self.assertAlmostEqual(float(s), 1.0, places=6)
            self.assertEqual(int(c), 336)

        metrics = evaluate_predictions(win_pred, rt3_pred, te_res)
        self.assertLess(metrics["win_logloss"], math.log(8.0))
        self.assertLess(metrics["rt3_logloss"], math.log(336.0))
        self.assertEqual(len(metrics["ece_bins"]), 11)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(Path(tmp_dir))
            loaded = ThreeStageModel.load(Path(tmp_dir))
            np.testing.assert_allclose(win_pred["pred_prob"].to_numpy(),
                                       loaded.predict_win(te_df)["pred_prob"].to_numpy(), rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(rt3_pred["combo_prob"].to_numpy(),
                                       loaded.predict_rt3(te_df)["combo_prob"].to_numpy(), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

# tests/test_v0_pipeline.py（Gemini作成、Claude確認）
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from autorace.model.combos import generate_rt3_combos
from autorace.model.evaluate import evaluate_predictions
from autorace.model.v0 import BaselineModelV0


def _make_synthetic_data():
    """2日分の合成データ。2日目は8番車が欠場。"""
    prog_rows, res_rows = [], []
    for r_no in [1, 2]:
        for c in range(1, 9):
            h = 0 if c <= 2 else (10 if c <= 5 else 20)
            prog_rows.append({"place_code": "02", "race_date": "2026-11-01", "race_no": r_no, "player_code": f"P{c:02d}",
                              "car_no": c, "branch_place_code": "02", "graduation_code": 30, "age": 30 + c,
                              "rank": f"S-{c}" if c <= 2 else (f"A-{c*10}" if c <= 6 else f"B-{c*10}"),
                              "handicap": h, "trial_run_time": 3.30 + c * 0.01, "absent": None,
                              "race_dev": 50.0, "rate2": 0.3, "rate3": 0.5})
            res_rows.append({"place_code": "02", "race_date": "2026-11-01", "race_no": r_no, "player_code": f"P{c:02d}",
                             "car_no": c, "order": c, "accident_code": None, "handicap": h,
                             "trial_time": 3.30 + c * 0.01, "race_time": 3.40 + c * 0.01, "st": 0.15})
    for c in range(1, 9):
        h = 0 if c <= 3 else 10
        is_abs = "欠場" if c == 8 else None
        prog_rows.append({"place_code": "02", "race_date": "2026-11-02", "race_no": 1, "player_code": f"P{c:02d}",
                          "car_no": c, "branch_place_code": "02", "graduation_code": 30, "age": 30 + c,
                          "rank": f"A-{c*5}", "handicap": h, "trial_run_time": None if is_abs else 3.32 + c * 0.01,
                          "absent": is_abs, "race_dev": 50.0, "rate2": 0.3, "rate3": 0.5})
        res_rows.append({"place_code": "02", "race_date": "2026-11-02", "race_no": 1, "player_code": f"P{c:02d}",
                         "car_no": c, "order": None if is_abs else (8 - c), "accident_code": None, "handicap": h,
                         "trial_time": None if is_abs else 3.32 + c * 0.01,
                         "race_time": None if is_abs else 3.42 + c * 0.01, "st": 0.15})
    return pd.DataFrame(prog_rows), pd.DataFrame(res_rows)


class TestAutoraceV0(unittest.TestCase):
    def test_no_future_or_sameday_leakage(self):
        """未来日や同日別レースの着順を変えても、1日目の予測が変わらないこと"""
        prog, res1 = _make_synthetic_data()
        res2 = res1.copy()
        day2 = res2["race_date"] == "2026-11-02"
        res2.loc[day2 & (res2["car_no"] == 1), "order"] = 1
        res2.loc[day2 & (res2["car_no"] == 7), "order"] = 7
        d1r2 = (res2["race_date"] == "2026-11-01") & (res2["race_no"] == 2)
        res2.loc[d1r2 & (res2["car_no"] == 1), "order"] = 8
        res2.loc[d1r2 & (res2["car_no"] == 8), "order"] = 1
        model = BaselineModelV0()
        p1, p2 = model.predict_historical(prog, res1), model.predict_historical(prog, res2)
        np.testing.assert_allclose(p1[p1["race_date"] == "2026-11-01"]["pred_prob"].to_numpy(),
                                   p2[p2["race_date"] == "2026-11-01"]["pred_prob"].to_numpy(), atol=1e-12)
        res3 = res1.copy()
        res3.loc[day2 & (res3["car_no"] == 1), "order"] = 1
        res3.loc[day2 & (res3["car_no"] == 7), "order"] = 7
        p3 = model.predict_historical(prog, res3)
        np.testing.assert_allclose(p1[p1["race_date"] == "2026-11-02"]["pred_prob"].to_numpy(),
                                   p3[p3["race_date"] == "2026-11-02"]["pred_prob"].to_numpy(), atol=1e-12)

    def test_probabilities_sum_to_one_and_absent_zero(self):
        prog, res = _make_synthetic_data()
        pred = BaselineModelV0().predict_historical(prog, res)
        for val in pred.groupby(["place_code", "race_date", "race_no"])["pred_prob"].sum():
            self.assertAlmostEqual(float(val), 1.0, places=9)
        self.assertEqual(float(pred.loc[(pred["race_date"] == "2026-11-02") & (pred["car_no"] == 8), "pred_prob"].iloc[0]), 0.0)
        stats = generate_rt3_combos(pred).groupby(["place_code", "race_date", "race_no"]).agg(
            total_prob=("combo_prob", "sum"), count=("combo", "count"))
        self.assertEqual(int(stats.loc[("02", "2026-11-01", 1), "count"]), 336)
        self.assertEqual(int(stats.loc[("02", "2026-11-02", 1), "count"]), 210)
        for val in stats["total_prob"]:
            self.assertAlmostEqual(float(val), 1.0, places=9)
        metrics = evaluate_predictions(pred, res)
        self.assertEqual(metrics["all_races"]["num_races"], 3)
        self.assertIsNotNone(metrics["all_races"]["ece_20"])


if __name__ == "__main__":
    unittest.main()

"""settle_daily のテスト（合成データ）"""
import itertools
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from autorace.scripts import settle_daily as sd

RK = ["place_code", "race_date", "race_no"]


class TestSettle(unittest.TestCase):
    def test_select_recommendations(self):
        combos = pd.DataFrame({"place_code": 6, "race_date": "2026-11-12", "race_no": 1, "bet_type": "rt3",
                               "combo": ["1-2-3", "1-3-2", "2-1-3", "2-3-1", "3-1-2", "3-2-1", "1-2-4"],
                               "combo_prob": [0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.005]})
        odds = pd.DataFrame({"place_code": 6, "race_date": "2026-11-12", "race_no": 1, "bet_type": "rt3", "snapshot": "pre",
                             "combo": ["1-2-3", "1-3-2", "2-1-3", "2-3-1", "3-1-2", "3-2-1", "1-2-4"],
                             "odds": [3.0, 7.0, 9.0, 13.0, 14.0, 15.0, 400.0]})
        recs = sd.select_recommendations(combos, odds)
        # 期待値：0.9, 1.4, 1.35, 1.3, 1.4, 1.5 → 1.2以上は5つ。確率1%未満（1-2-4）は除外
        self.assertEqual(len(recs), 5)
        self.assertTrue((recs["ev"] >= 1.2).all())
        self.assertNotIn("1-2-3", set(recs["combo"]))
        self.assertNotIn("1-2-4", set(recs["combo"]))

    def test_settle(self):
        recs = pd.DataFrame({"place_code": [6, 6], "race_date": "2026-11-12", "race_no": 1, "bet_type": "rt3",
                             "combo": ["1-3-2", "2-1-3"], "combo_prob": 0.2, "pre_odds": [7.0, 9.0], "ev": [1.4, 1.8]})
        res = pd.DataFrame({"place_code": 6, "race_date": "2026-11-12", "race_no": 1, "car_no": [1, 3, 2], "order": [1, 2, 3]})
        refund = pd.DataFrame({"place_code": [6], "race_date": ["2026-11-12"], "race_no": [1], "bet_type": ["rt3"], "combo": ["1-3-2"], "payout": [650]})
        fin = pd.DataFrame({"place_code": 6, "race_date": "2026-11-12", "race_no": 1, "bet_type": "rt3", "snapshot": "final",
                            "combo": ["1-3-2", "2-1-3"], "odds": [6.5, 11.0]})
        s = sd.settle_recommendations(recs, res, refund, fin).set_index("combo")
        self.assertTrue(s.loc["1-3-2", "is_hit"])
        self.assertAlmostEqual(s.loc["1-3-2", "return_amount"], 6.5)
        self.assertFalse(s.loc["2-1-3", "is_hit"])
        self.assertEqual(s.loc["2-1-3", "return_amount"], 0.0)
        self.assertAlmostEqual(s.loc["2-1-3", "odds_change_rate"], (11 - 9) / 9)

    def test_no_data_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(sd.run_settle_daily(Path(tmp), "2026-11-12"), 0)

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            def save(name, df):
                p = d / name / "2026"; p.mkdir(parents=True, exist_ok=True); df.to_parquet(p / "2026-11.parquet")
            prog_rows, res_rows = [], []
            for day in ("2026-11-10", "2026-11-11", "2026-11-12"):
                for c in range(1, 7):
                    prog_rows.append({"place_code": 6, "race_date": day, "race_no": 1, "player_code": 3000 + c, "car_no": c,
                                      "rank": f"A-{c}", "handicap": 10 * (c // 3), "trial_run_time": 3.30 + c / 100, "absent": None})
                    res_rows.append({"place_code": 6, "race_date": day, "race_no": 1, "player_code": 3000 + c, "car_no": c,
                                     "order": c, "accident_code": None, "handicap": 10 * (c // 3)})
            prog = pd.DataFrame(prog_rows)
            save("program", prog); save("race_result", pd.DataFrame(res_rows))
            pre = prog[prog.race_date == "2026-11-12"].assign(fetched_at="2026-11-12T12:00:00+09:00", minutes_to_close=5.0)
            save("program_pre", pre)
            odds = [{"place_code": 6, "race_date": "2026-11-12", "race_no": 1, "bet_type": "rt3", "snapshot": "pre",
                     "combo": f"{a}-{b}-{c}", "odds": 200.0} for a, b, c in itertools.permutations(range(1, 7), 3)]
            save("odds_pre", pd.DataFrame(odds))
            self.assertEqual(sd.run_settle_daily(d, "2026-11-12"), 0)
            summ = pd.read_parquet(d / "tracker/daily_summary.parquet")
            self.assertEqual(int(summ["num_races"].iloc[0]), 1)
            self.assertTrue((d / "tracker/dashboard.html").exists())
            preds = pd.read_parquet(d / "tracker/predictions/2026/2026-11.parquet")
            self.assertEqual(int(preds["is_win"].sum()), 1)  # 着順の列を正しく読めている
            sd.run_settle_daily(d, "2026-11-12")  # 再実行で上書き（重複しない）
            self.assertEqual(len(pd.read_parquet(d / "tracker/daily_summary.parquet")), 1)


if __name__ == "__main__":
    unittest.main()

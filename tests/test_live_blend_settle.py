"""本番用の予想（fit_live_model）と、それを使う答え合わせ（settle_daily の v1 合成）のテスト（人工データ）"""
from __future__ import annotations

import itertools
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from autorace.scripts import settle_daily as sd
from autorace.scripts.fit_live_model import fit_live_model, split_periods
from tests.test_v1 import _generate_synthetic_autorace_data

TARGET = "2026-03-31"


def _save_monthly(d: Path, name: str, df: pd.DataFrame) -> None:
    for ym, g in df.groupby(df["race_date"].astype(str).str[:7]):
        p = d / name / ym[:4]
        p.mkdir(parents=True, exist_ok=True)
        g.reset_index(drop=True).to_parquet(p / f"{ym}.parquet")


def _win_odds(program: pd.DataFrame) -> pd.DataFrame:
    """試走が速い車ほど人気（単勝オッズが低い）にする"""
    rows = []
    for key, g in program.groupby(["place_code", "race_date", "race_no"]):
        s = np.exp(-(g["trial_run_time"].to_numpy() - g["trial_run_time"].min()) * 30.0)
        p = s / s.sum()
        for car, pr in zip(g["car_no"], p):
            rows.append({"place_code": key[0], "race_date": key[1], "race_no": key[2], "bet_type": "tns",
                         "combo": str(int(car)), "odds": round(max(1.1, 0.75 / pr), 1)})
    return pd.DataFrame(rows)


class TestLiveBlendSettle(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.program, cls.info, cls.result = _generate_synthetic_autorace_data(n_days=90)
        cls.tmp = tempfile.mkdtemp()
        d = Path(cls.tmp) / "data"
        _save_monthly(d, "program", cls.program)
        _save_monthly(d, "race_result", cls.result)
        _save_monthly(d, "race_info", cls.info)
        final = _win_odds(cls.program).assign(snapshot="final")
        _save_monthly(d, "odds", final)
        # 本番用の予想を作って data/models/live に置く（Drive から取ってくるのと同じ場所）
        cls.meta = fit_live_model(d, Path(cls.tmp) / "out", "2026-03-30", train_months=2, blend_months=1)
        shutil.copytree(Path(cls.tmp) / "out" / "live", d / "models" / "live")
        # 対象日の直前データ：出走表・単勝・3連単（全組 60 倍）
        day = cls.program[cls.program["race_date"] == TARGET]
        _save_monthly(d, "program_pre", day.assign(fetched_at="2026-03-31T12:00:00+09:00", minutes_to_close=8.0))
        tns = _win_odds(day)
        rt3 = [{"place_code": int(k[0]), "race_date": TARGET, "race_no": int(k[1]), "bet_type": "rt3",
                "combo": f"{a}-{b}-{c}", "odds": 60.0}
               for k in day[["place_code", "race_no"]].drop_duplicates().itertuples(index=False)
               for a, b, c in itertools.permutations(range(1, 9), 3)]
        pre = pd.concat([tns, pd.DataFrame(rt3)], ignore_index=True).assign(
            snapshot="pre", fetched_at="2026-03-31T12:00:00+09:00", minutes_to_close=8.0)
        _save_monthly(d, "odds_pre", pre)
        cls.data_dir = d

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_split_periods(self) -> None:
        p = split_periods("2026-11-24", 12, 3)
        self.assertEqual(p, {"train_start": "2025-08-25", "blend_start": "2026-08-25", "end": "2026-11-24"})

    def test_live_model_files(self) -> None:
        live = self.data_dir / "models" / "live"
        self.assertTrue((live / "blend.json").exists())
        self.assertTrue((live / "model" / "meta.json").exists())
        self.assertGreater(self.meta["train_rows"], 0)
        self.assertGreater(self.meta["blend_rows"], 0)

    def test_settle_uses_v1_blend(self) -> None:
        self.assertEqual(sd.run_settle_daily(self.data_dir, TARGET), 0)
        tr = self.data_dir / "tracker"
        summ = pd.read_parquet(tr / "daily_summary.parquet")
        self.assertEqual(summ["model"].iloc[-1], "v1_blend")
        self.assertEqual(int(summ["num_races"].iloc[-1]), 3)
        recs = pd.read_parquet(tr / "recommendations" / "2026" / "2026-03.parquet")
        if len(recs):
            self.assertLessEqual(int(recs.groupby(["place_code", "race_no"]).size().max()), sd.V1_MAX_RECS_PER_RACE)
            self.assertTrue((recs["ev"] >= sd.V1_MIN_EV).all())
            self.assertTrue((recs["pre_odds"] <= sd.V1_MAX_ODDS).all())
        preds = pd.read_parquet(tr / "predictions" / "2026" / "2026-03.parquet")
        sums = preds.groupby(["place_code", "race_no"])["pred_prob"].sum().to_numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)
        page = (tr / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn("新（合成）", page)
        self.assertNotIn("LogLoss", page)

    def test_settle_falls_back_to_v0_without_live_model(self) -> None:
        d = Path(self.tmp) / "nolive"
        shutil.copytree(self.data_dir, d, ignore=shutil.ignore_patterns("models", "tracker"))
        self.assertEqual(sd.run_settle_daily(d, TARGET), 0)
        summ = pd.read_parquet(d / "tracker" / "daily_summary.parquet")
        self.assertEqual(summ["model"].iloc[-1], "v0")


if __name__ == "__main__":
    unittest.main()

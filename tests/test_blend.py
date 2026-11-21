# tests/test_blend.py （Gemini作成、Claude確認）
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from autorace.model.blend import BenterBlend, _get_combo_strings, _get_perm_indices, blended_rt3
from autorace.model.features import RACE_KEYS
from autorace.model.market import fit_market_gamma, market_rt3_probs, market_win_probs, pick_snapshot
from autorace.scripts.eval_blend import backtest_ev_grid, compute_win_logloss

K = list(RACE_KEYS)


class _DummyBooster:
    def __init__(self, col_idx: int) -> None:
        self.col_idx = col_idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X[:, self.col_idx], dtype=np.float64)


class _DummyThreeStageModel:
    """ThreeStageModel と同じ形のテスト用モデル"""

    def __init__(self) -> None:
        self.boosters = {1: _DummyBooster(0), 2: _DummyBooster(1), 3: _DummyBooster(2)}
        self.temperatures = {1: 1.2, 2: 1.1, 3: 1.0}

    def _extract_X(self, df: pd.DataFrame) -> np.ndarray:
        return df[["f1", "f2", "f3"]].to_numpy(dtype=np.float64)

    def predict_win(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[K + ["car_no"]].copy()
        out["_z1"] = self.boosters[1].predict(self._extract_X(df)) / self.temperatures[1]
        out["_ez"] = np.exp(out["_z1"] - out.groupby(K, sort=False)["_z1"].transform("max"))
        out["pred_prob"] = out["_ez"] / out.groupby(K, sort=False)["_ez"].transform("sum")
        return out[K + ["car_no", "pred_prob"]].reset_index(drop=True)

    def predict_rt3(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._extract_X(df)
        work = df[K + ["car_no"]].copy()
        work["_z1"], work["_z2"], work["_z3"] = (self.boosters[s].predict(X) / self.temperatures[s] for s in (1, 2, 3))
        rows = []
        for key, g in work.groupby(K, sort=False):
            cars = tuple(int(c) for c in g["car_no"])
            ii, jj, kk = _get_perm_indices(len(cars))
            w1, w2, w3 = (np.exp(g[c].to_numpy() - g[c].max()) for c in ("_z1", "_z2", "_z3"))
            p1 = w1 / w1.sum()
            p = p1[ii] * (w2[jj] / (w2.sum() - w2[ii])) * (w3[kk] / (w3.sum() - w3[ii] - w3[jj]))
            p /= p.sum()
            for c, pr in zip(_get_combo_strings(cars), p):
                rows.append({"place_code": key[0], "race_date": key[1], "race_no": key[2], "bet_type": "rt3",
                             "combo": c, "combo_prob": float(pr)})
        return pd.DataFrame(rows).sort_values(K + ["combo"]).reset_index(drop=True)


def _make_synthetic_dataset(n_races: int = 120, n_cars: int = 6, seed: int = 42):
    """真の強さに対して、モデルと市場がそれぞれ独立なノイズを持つ人工データ"""
    rng = np.random.default_rng(seed)
    feat_rows, odds_rows, res_rows = [], [], []
    for r in range(n_races):
        place, r_date, r_no = 2, f"2026-01-{(r // 12) + 1:02d}", (r % 12) + 1
        true_logit = rng.normal(0.0, 1.0, size=n_cars)
        model_logit = 0.75 * true_logit + rng.normal(0.0, 0.70, size=n_cars)
        market_logit = 0.85 * true_logit + rng.normal(0.0, 0.55, size=n_cars)
        true_p = np.exp(true_logit - true_logit.max())
        true_p /= true_p.sum()
        order_perm = rng.choice(n_cars, size=n_cars, replace=False, p=true_p)
        ranks = np.empty(n_cars, dtype=int)
        for o, c in enumerate(order_perm, start=1):
            ranks[c] = o
        mkt_w = np.exp(market_logit - market_logit.max())
        mkt_p = mkt_w / mkt_w.sum()
        tns_odds = np.maximum(1.0, 0.75 / mkt_p)
        for c in range(n_cars):
            feat_rows.append({"place_code": place, "race_date": r_date, "race_no": r_no, "car_no": c + 1,
                              "f1": float(model_logit[c] * 1.2), "f2": float(model_logit[c] * 1.1 + rng.normal(0, 0.1)),
                              "f3": float(model_logit[c] + rng.normal(0, 0.1))})
            odds_rows.append({"place_code": place, "race_date": r_date, "race_no": r_no, "bet_type": "tns",
                              "combo": str(c + 1), "odds": float(tns_odds[c]), "snapshot": "final"})
            res_rows.append({"place_code": place, "race_date": r_date, "race_no": r_no, "car_no": c + 1, "order": int(ranks[c])})
        ii, jj, kk = _get_perm_indices(n_cars)
        W = mkt_w.sum()
        rp = mkt_p[ii] * (mkt_w[jj] / (W - mkt_w[ii])) * (mkt_w[kk] / (W - mkt_w[ii] - mkt_w[jj]))
        rp /= rp.sum()
        for cs, cp in zip(_get_combo_strings(tuple(range(1, n_cars + 1))), rp):
            odds_rows.append({"place_code": place, "race_date": r_date, "race_no": r_no, "bet_type": "rt3",
                              "combo": cs, "odds": float(max(1.0, 0.70 / cp)), "snapshot": "final"})
    return pd.DataFrame(feat_rows), pd.DataFrame(odds_rows), pd.DataFrame(res_rows)


class TestMarketAndBlend(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.feat_df, cls.odds_df, cls.res_df = _make_synthetic_dataset()
        cls.v1 = _DummyThreeStageModel()
        cls.win_pred = cls.v1.predict_win(cls.feat_df)
        cls.rt3_pred = cls.v1.predict_rt3(cls.feat_df)
        cls.mkt_win = market_win_probs(cls.odds_df)
        cls.mkt_rt3 = market_rt3_probs(cls.odds_df)

    def test_power_method_gamma_one_matches_normalized(self) -> None:
        p_norm, p_pow1 = market_win_probs(self.odds_df), market_win_probs(self.odds_df, gamma=1.0)
        np.testing.assert_allclose(p_norm["market_prob"].to_numpy(), p_pow1["market_prob"].to_numpy(), atol=1e-12)
        sums = p_pow1.groupby(K)["market_prob"].sum().to_numpy()
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-12)
        g = fit_market_gamma(self.odds_df, self.res_df)
        self.assertTrue(0.1 < g < 3.0)

    def test_blend_probs_sum_to_one_per_race(self) -> None:
        blend = BenterBlend(alpha=0.4, beta=0.8, alpha3=0.35, beta3=0.75)
        w = blend.predict_win(self.win_pred, self.mkt_win)
        np.testing.assert_allclose(w.groupby(K)["pred_prob"].sum().to_numpy(), 1.0, atol=1e-12)
        a = blend.blended_rt3(method="a", rt3_pred_df=self.rt3_pred, market_rt3_df=self.mkt_rt3)
        b = blended_rt3(blend, method="b", v1_model=self.v1, df=self.feat_df, market_win_df=self.mkt_win)
        cols = K + ["bet_type", "combo", "combo_prob"]
        self.assertEqual(list(a.columns), cols)
        self.assertEqual(list(b.columns), cols)
        np.testing.assert_allclose(a.groupby(K)["combo_prob"].sum().to_numpy(), 1.0, atol=1e-12)
        np.testing.assert_allclose(b.groupby(K)["combo_prob"].sum().to_numpy(), 1.0, atol=1e-12)
        self.assertEqual(int(a.groupby(K).size().iloc[0]), 120)  # 6車立てで 6*5*4

    def test_alpha1_beta0_is_model_and_alpha0_beta1_is_market(self) -> None:
        bm = BenterBlend(alpha=1.0, beta=0.0, alpha3=1.0, beta3=0.0)
        m = bm.predict_win(self.win_pred, self.mkt_win).merge(self.win_pred.rename(columns={"pred_prob": "o"}), on=K + ["car_no"])
        np.testing.assert_allclose(m["pred_prob"], m["o"], atol=1e-12)
        rb = bm.blended_rt3(method="b", v1_model=self.v1, df=self.feat_df, market_win_df=self.mkt_win)
        mr = rb.merge(self.rt3_pred.rename(columns={"combo_prob": "o"}), on=K + ["combo"])
        np.testing.assert_allclose(mr["combo_prob"], mr["o"], atol=1e-12)
        ra = bm.predict_rt3_direct(self.rt3_pred, self.mkt_rt3).merge(self.rt3_pred.rename(columns={"combo_prob": "o"}), on=K + ["combo"])
        np.testing.assert_allclose(ra["combo_prob"], ra["o"], atol=1e-12)
        bk = BenterBlend(alpha=0.0, beta=1.0)
        k = bk.predict_win(self.win_pred, self.mkt_win).merge(self.mkt_win, on=K + ["car_no"])
        np.testing.assert_allclose(k["pred_prob"], k["market_prob"], atol=1e-12)

    def test_fit_lowers_logloss_and_save_load(self) -> None:
        ll_model, _ = compute_win_logloss(self.win_pred, self.res_df)
        ll_market, _ = compute_win_logloss(self.mkt_win.rename(columns={"market_prob": "pred_prob"}), self.res_df)
        blend = BenterBlend().fit(self.win_pred, self.mkt_win, self.res_df)
        blend.fit_rt3(self.rt3_pred, self.mkt_rt3, self.res_df)
        after = blend.predict_win(self.win_pred, self.mkt_win)
        ll_after, _ = compute_win_logloss(after, self.res_df)
        self.assertLess(ll_after, ll_model)
        self.assertLess(ll_after, ll_market)
        info = blend.fit_rt3_info_
        self.assertLessEqual(info["logloss_blend"], min(info["logloss_model_only"], info["logloss_market_only"]) + 1e-12)
        with tempfile.TemporaryDirectory() as t:
            blend.save(Path(t) / "blend.json")
            loaded = BenterBlend.load(Path(t) / "blend.json")
            self.assertAlmostEqual(blend.alpha, loaded.alpha, places=12)
            np.testing.assert_allclose(after["pred_prob"], loaded.predict_win(self.win_pred, self.mkt_win)["pred_prob"], atol=1e-12)

    def test_pick_snapshot_and_ev_backtest(self) -> None:
        base = {"place_code": 2, "race_date": "2026-01-01", "race_no": 1, "bet_type": "rt3", "combo": "1-2-3", "snapshot": "pre"}
        pre = pd.DataFrame([{**base, "odds": 25.0, "fetched_at": "2026-01-01T02:00:00Z", "minutes_to_close": 20.0},
                            {**base, "odds": 18.0, "fetched_at": "2026-01-01T02:13:00Z", "minutes_to_close": 7.0},
                            {**base, "odds": 14.0, "fetched_at": "2026-01-01T02:18:00Z", "minutes_to_close": 2.0}])
        self.assertEqual(float(pick_snapshot(pre, how="earliest")["odds"].iloc[0]), 25.0)
        self.assertEqual(float(pick_snapshot(pre, how="latest")["odds"].iloc[0]), 14.0)
        self.assertEqual(float(pick_snapshot(pre, how="min_before_close", minutes=5.0)["odds"].iloc[0]), 18.0)
        refund = pd.DataFrame([{"place_code": 2, "race_date": "2026-01-01", "race_no": 1, "bet_type": "rt3", "combo": "1-2-3", "payout": 1500}])
        grid = backtest_ev_grid(self.rt3_pred.iloc[:60], self.odds_df, refund, thresholds=(1.0,), max_odds_list=(200.0,), max_k_list=(1,))
        self.assertEqual(len(grid), 1)
        self.assertNotIn("combo", grid[0])
        self.assertIn("roi", grid[0])
        self.assertIn("monthly_roi", grid[0])


if __name__ == "__main__":
    unittest.main()

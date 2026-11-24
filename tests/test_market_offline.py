# tests/test_market_offline.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autorace.model.features import RACE_KEYS
from autorace.model.market import fit_market_gamma, market_rt3_probs, market_win_probs, pick_snapshot

K = list(RACE_KEYS)


class TestMarketWinProbs:
    def test_empty_and_none_inputs(self) -> None:
        expected_cols = K + ["car_no", "odds", "market_prob"]
        res_none = market_win_probs(None)
        assert list(res_none.columns) == expected_cols
        assert res_none.empty

        res_empty = market_win_probs(pd.DataFrame())
        assert list(res_empty.columns) == expected_cols
        assert res_empty.empty

    def test_filters_non_tns_bet_types(self) -> None:
        df = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "rt3", "combo": "1-2-3", "odds": 10.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "rtw", "combo": "1-2", "odds": 5.0},
        ])
        res = market_win_probs(df)
        assert res.empty

    def test_normal_input_without_bet_type_or_car_no(self) -> None:
        df = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": " 1 ", "odds": 2.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "2", "odds": 4.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "3", "odds": 4.0},
        ])
        res = market_win_probs(df)
        assert len(res) == 3
        assert list(res["car_no"]) == [1, 2, 3]
        probs = res["market_prob"].to_numpy()
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(probs, [0.5, 0.25, 0.25], atol=1e-12)

    def test_explicit_car_no_and_invalid_scratched_rows(self) -> None:
        df = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "car_no": 1, "odds": 2.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "car_no": 2, "odds": 0.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "car_no": 3, "odds": -5.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "car_no": 4, "odds": np.nan},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "car_no": 5, "odds": np.inf},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "car_no": "欠", "odds": 3.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "car_no": 6, "odds": 6.0},
        ])
        res = market_win_probs(df)
        assert list(res["car_no"]) == [1, 6]
        np.testing.assert_allclose(res["market_prob"].to_numpy(), [0.75, 0.25], atol=1e-12)

    def test_all_invalid_odds_returns_empty(self) -> None:
        df = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "1", "odds": 0.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "2", "odds": "欠"},
        ])
        assert market_win_probs(df).empty

    def test_deduplication_keeps_last_and_gamma_boundaries(self) -> None:
        df = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "1", "odds": 10.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "1", "odds": 2.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "2", "odds": 6.0},
        ])
        res_g0 = market_win_probs(df, gamma=0.0)
        np.testing.assert_allclose(res_g0["market_prob"].to_numpy(), [0.5, 0.5], atol=1e-12)

        res_g2 = market_win_probs(df, gamma=2.0)
        # (1/2)^2 = 1/4, (1/6)^2 = 1/36 -> 9:1 -> 0.9, 0.1
        np.testing.assert_allclose(res_g2["market_prob"].to_numpy(), [0.9, 0.1], atol=1e-12)


class TestFitMarketGamma:
    def test_empty_or_non_overlapping_returns_default_one(self) -> None:
        assert fit_market_gamma(pd.DataFrame(), pd.DataFrame()) == 1.0
        odds = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "1", "odds": 2.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "2", "odds": 3.0},
        ])
        assert fit_market_gamma(odds, None) == 1.0
        assert fit_market_gamma(odds, pd.DataFrame()) == 1.0

        res_diff_race = pd.DataFrame([
            {"place_code": 3, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1},
            {"place_code": 3, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 2},
        ])
        assert fit_market_gamma(odds, res_diff_race) == 1.0

    def test_no_winner_or_single_car_race_returns_one(self) -> None:
        odds = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "1", "odds": 2.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "2", "odds": 3.0},
        ])
        res_no_win = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 3},
        ])
        assert fit_market_gamma(odds, res_no_win) == 1.0

        res_single_car = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1},
        ])
        assert fit_market_gamma(odds, res_single_car) == 1.0

    def test_valid_fit_stays_within_bounds(self) -> None:
        odds = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "1", "odds": 1.5},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "2", "odds": 4.5},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "bet_type": "tns", "combo": "1", "odds": 1.5},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "bet_type": "tns", "combo": "2", "odds": 4.5},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 3, "bet_type": "tns", "combo": "1", "odds": 1.5},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 3, "bet_type": "tns", "combo": "2", "odds": 4.5},
        ])
        res = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 1, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 2, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 3, "car_no": 1, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 3, "car_no": 2, "order": 2},
        ])
        # Independent NLL calculation: for each race, odds are (1.5, 4.5) and car 1 always wins.
        # p1(gamma) = (1/1.5)^gamma / ((1/1.5)^gamma + (1/4.5)^gamma) = 1 / (1 + 3^{-gamma})
        def manual_nll(g_val: float) -> float:
            w1 = (1.0 / 1.5) ** g_val
            w2 = (1.0 / 4.5) ** g_val
            p1 = w1 / (w1 + w2)
            return float(-np.log(p1))

        g_high = fit_market_gamma(odds, res, bounds=(0.2, 2.5))
        assert 1.0 < g_high <= 2.5
        assert manual_nll(g_high) < manual_nll(1.0)

        # When the underdog (car 2, odds 4.5) always wins, optimum gamma is < 1.0
        res_underdog = res.copy()
        res_underdog["order"] = [2, 1, 2, 1, 2, 1]

        def manual_nll_underdog(g_val: float) -> float:
            w1 = (1.0 / 1.5) ** g_val
            w2 = (1.0 / 4.5) ** g_val
            p2 = w2 / (w1 + w2)
            return float(-np.log(p2))

        g_low = fit_market_gamma(odds, res_underdog, bounds=(0.2, 2.5))
        assert 0.2 <= g_low < 1.0
        assert manual_nll_underdog(g_low) < manual_nll_underdog(1.0)


class TestMarketRt3Probs:
    def test_empty_and_none(self) -> None:
        expected_cols = K + ["bet_type", "combo", "odds", "market_prob"]
        assert list(market_rt3_probs(None).columns) == expected_cols
        assert market_rt3_probs(None).empty
        assert market_rt3_probs(pd.DataFrame()).empty

    def test_normal_and_invalid_rows(self) -> None:
        df = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": " 1-2-3 ", "odds": 5.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "1-3-2", "odds": 15.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "", "odds": 10.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "2-1-3", "odds": 0.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "2-3-1", "odds": -10.0},
        ])
        res = market_rt3_probs(df)
        assert len(res) == 2
        assert list(res["combo"]) == ["1-2-3", "1-3-2"]
        assert set(res["bet_type"]) == {"rt3"}
        probs = res["market_prob"].to_numpy()
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(probs, [0.75, 0.25], atol=1e-12)


class TestPickSnapshot:
    def test_invalid_how_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported snapshot selection mode"):
            pick_snapshot(pd.DataFrame(), how="invalid_mode")

    def test_empty_and_none(self) -> None:
        assert pick_snapshot(None).empty
        empty_df = pd.DataFrame(columns=K + ["bet_type", "combo", "odds"])
        res = pick_snapshot(empty_df, how="latest")
        assert res.empty
        assert list(res.columns) == list(empty_df.columns)

    def test_min_before_close_missing_mtc_and_all_below_threshold(self) -> None:
        base = {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "rt3"}
        df = pd.DataFrame([
            # combo 1-2-3: all minutes_to_close are NaN -> falls back to earliest
            {**base, "combo": "1-2-3", "odds": 20.0, "fetched_at": "2026-11-01T02:00:00Z", "minutes_to_close": np.nan},
            {**base, "combo": "1-2-3", "odds": 15.0, "fetched_at": "2026-11-01T02:05:00Z", "minutes_to_close": np.nan},
            # combo 1-3-2: all minutes_to_close < 5.0 -> excluded
            {**base, "combo": "1-3-2", "odds": 30.0, "fetched_at": "2026-11-01T02:08:00Z", "minutes_to_close": 4.0},
            {**base, "combo": "1-3-2", "odds": 28.0, "fetched_at": "2026-11-01T02:09:00Z", "minutes_to_close": 1.0},
            # combo 2-1-3: mix of >= 5.0 and < 5.0 -> picks latest with >= 5.0
            {**base, "combo": "2-1-3", "odds": 50.0, "fetched_at": "2026-11-01T02:00:00Z", "minutes_to_close": 10.0},
            {**base, "combo": "2-1-3", "odds": 40.0, "fetched_at": "2026-11-01T02:05:00Z", "minutes_to_close": 5.0},
            {**base, "combo": "2-1-3", "odds": 35.0, "fetched_at": "2026-11-01T02:08:00Z", "minutes_to_close": 2.0},
        ])
        picked = pick_snapshot(df, how="min_before_close", minutes=5.0)
        assert list(picked["combo"]) == ["1-2-3", "2-1-3"]
        by_combo = picked.set_index("combo")["odds"].to_dict()
        assert by_combo["1-2-3"] == 20.0
        assert by_combo["2-1-3"] == 40.0

    def test_without_timestamp_or_mtc_columns(self) -> None:
        df = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "1", "odds": 3.0},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "bet_type": "tns", "combo": "1", "odds": 2.5},
        ])
        assert float(pick_snapshot(df, how="earliest")["odds"].iloc[0]) == 3.0
        assert float(pick_snapshot(df, how="latest")["odds"].iloc[0]) == 2.5
        assert float(pick_snapshot(df, how="min_before_close", minutes=5.0)["odds"].iloc[0]) == 3.0

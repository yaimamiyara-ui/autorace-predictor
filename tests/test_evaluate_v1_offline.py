# tests/test_evaluate_v1_offline.py
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from autorace.model.combos import compute_harville_rt3_for_race, generate_rt3_combos
from autorace.model.evaluate import evaluate_predictions as evaluate_predictions_v0
from autorace.model.evaluate_v1 import (
    ECE_BIN_LABELS,
    compute_rt3_ece,
    compute_rt3_logloss_and_hit,
    compute_win_logloss,
    evaluate_predictions,
)
from autorace.model.v1 import compute_group_softmax, fit_temperature, prepare_stage_dataset
from autorace.scripts.backtest_v0 import actual_rt3


class TestEvaluateV1:
    def test_compute_win_logloss_empty_and_dead_heat(self) -> None:
        ll, n = compute_win_logloss(pd.DataFrame(), pd.DataFrame())
        assert math.isnan(ll) and n == 0

        pred = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "pred_prob": 0.5},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "pred_prob": 0.5},
        ])
        # Dead heat (two order==1 in the same race) should be excluded
        dead_heat_res = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 1},
        ])
        ll_dh, n_dh = compute_win_logloss(pred, dead_heat_res)
        assert math.isnan(ll_dh) and n_dh == 0

    def test_compute_win_logloss_normal_and_zero_prob_clip(self) -> None:
        pred = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "pred_prob": 0.8},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "pred_prob": 0.2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 1, "pred_prob": 0.0},
        ])
        res = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 1, "order": 1},
        ])
        ll, n = compute_win_logloss(pred, res)
        assert n == 2
        assert math.isfinite(ll)
        assert ll == pytest.approx((-math.log(0.8) - math.log(1e-15)) / 2.0)

    def test_compute_rt3_logloss_and_hit_and_ece(self) -> None:
        ll, hit, n = compute_rt3_logloss_and_hit(pd.DataFrame(), pd.DataFrame())
        assert math.isnan(ll) and math.isnan(hit) and n == 0
        ece, bins = compute_rt3_ece(pd.DataFrame(), pd.DataFrame())
        assert math.isnan(ece) and bins == []

        rt3_pred = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "1-2-3", "combo_prob": 0.6},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "1-3-2", "combo_prob": 0.4},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "combo": "2-1-3", "combo_prob": 0.5},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "combo": "1-2-3", "combo_prob": 0.5},
        ])
        res = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 3, "order": 3},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 2, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 1, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 3, "order": 3},
        ])
        # In R2, "1-2-3" and "2-1-3" are tied at 0.5; tie-breaking sorts by combo ascending ("1-2-3" first), so R2 misses top1
        win_pred = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "pred_prob": 0.7},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 2, "pred_prob": 0.5},
        ])
        metrics = evaluate_predictions(win_pred, rt3_pred, res)
        assert metrics["n_win_races"] == 2
        assert metrics["n_rt3_races"] == 2
        assert metrics["rt3_top1_hit_rate"] == pytest.approx(0.5)
        assert metrics["rt3_logloss"] == pytest.approx((-math.log(0.6) - math.log(0.5)) / 2.0)
        assert len(metrics["ece_bins"]) == len(ECE_BIN_LABELS)
        assert sum(b["count"] for b in metrics["ece_bins"]) == 4
        assert 0.0 <= metrics["rt3_ece"] <= 1.0


class TestCombosAndV1PureHelpers:
    def test_harville_fewer_than_3_active_cars(self) -> None:
        assert compute_harville_rt3_for_race(np.array([1, 2]), np.array([0.6, 0.4])) == []
        assert compute_harville_rt3_for_race(np.array([1, 2, 3]), np.array([0.6, 0.4, 0.0])) == []
        assert generate_rt3_combos(pd.DataFrame()).empty

        combos = compute_harville_rt3_for_race(np.array([1, 2, 3, 4]), np.array([0.5, 0.3, 0.2, 0.0]))
        assert len(combos) == 6
        probs = np.array([p for _, p in combos])
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)

    def test_group_softmax_and_fit_temperature(self) -> None:
        assert len(compute_group_softmax(np.array([]), np.array([]))) == 0
        logits = np.array([1000.0, 1000.0, 0.0, 1.0, 2.0])
        groups = np.array([2, 3])
        probs = compute_group_softmax(logits, groups)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        np.testing.assert_allclose(probs[:2], [0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(probs[2:].sum(), 1.0, atol=1e-12)

        assert fit_temperature(np.array([]), np.array([]), np.array([])) == 1.0
        t = fit_temperature(logits, np.array([1.0, 0.0, 0.0, 0.0, 1.0]), groups)
        assert 0.1 <= t <= 10.0

    def test_prepare_stage_dataset_filtering(self) -> None:
        sub_e, y_e, g_e = prepare_stage_dataset(pd.DataFrame(), stage=1)
        assert sub_e.empty and len(y_e) == 0 and len(g_e) == 0

        df = pd.DataFrame([
            # Race 1: 4 active cars, clean finish 1..4
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": c, "order": c,
             "is_absent": False, "accident_code": None, "foul_code": None}
            for c in range(1, 5)
        ] + [
            # Race 2: 4 active cars, car 4 had an accident (order=NaN)
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 1, "order": 1, "is_absent": False, "accident_code": None},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 2, "order": 2, "is_absent": False, "accident_code": None},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 3, "order": 3, "is_absent": False, "accident_code": None},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 4, "order": np.nan, "is_absent": False, "accident_code": "1"},
        ])
        # Stage 1 includes both races (each has a single 1st-place winner and >=3 active cars)
        _, y1, g1 = prepare_stage_dataset(df, stage=1)
        assert list(g1) == [4, 4]
        assert y1.sum() == 2.0

        # Stage 2 & 3 exclude Race 2 due to accident, and drop 1st (and 2nd for stage 3) finishers
        _, y2, g2 = prepare_stage_dataset(df, stage=2)
        assert list(g2) == [3]
        assert y2.sum() == 1.0

        _, y3, g3 = prepare_stage_dataset(df, stage=3)
        assert list(g3) == [2]
        assert y3.sum() == 1.0

        with pytest.raises(ValueError, match="Invalid stage"):
            prepare_stage_dataset(df, stage=4)

    def test_actual_rt3_and_v0_evaluate_accident_split(self) -> None:
        # Incomplete top-3 returns empty actual_rt3
        inc_res = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1, "accident_code": None},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 2, "accident_code": "1"},
        ])
        assert actual_rt3(inc_res).empty

        pred = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "pred_prob": 0.6, "is_absent": False},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "pred_prob": 0.4, "is_absent": False},
        ])
        ev = evaluate_predictions_v0(pred, inc_res)
        assert ev["all_races"]["num_races"] == 1
        assert ev["no_accident_races"]["num_races"] == 0
        assert ev["no_accident_races"]["binary_logloss"] is None

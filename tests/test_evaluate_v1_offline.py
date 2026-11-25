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
        # 設計書どおり、1着同着は同着の選手すべてを1着として扱い、確率の合計で評価する
        dead_heat_res = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 1},
        ])
        ll_dh, n_dh = compute_win_logloss(pred, dead_heat_res)
        assert n_dh == 1
        assert ll_dh == pytest.approx(-math.log(0.5 + 0.5))

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

    def test_compute_rt3_ece_multi_bin_boundary_and_manual_values(self) -> None:
        # 5 combo predictions across 2 races, spread across 3 bins including boundary 0.01:
        # Bin 0 ("0-1%"): 0.004 (miss) -> count=1, mean_pred=0.004, actual_rate=0.0, abs_gap=0.004
        # Bin 1 ("1-2%"): 0.010 (boundary, miss), 0.016 (hit) -> count=2, mean_pred=0.013, actual_rate=0.5, abs_gap=0.487
        # Bin 10 ("10%以上"): 0.200 (hit), 0.400 (miss) -> count=2, mean_pred=0.300, actual_rate=0.5, abs_gap=0.200
        rt3_pred = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "1-2-3", "combo_prob": 0.016},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "1-3-2", "combo_prob": 0.010},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "2-1-3", "combo_prob": 0.004},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "combo": "2-1-3", "combo_prob": 0.200},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "combo": "1-2-3", "combo_prob": 0.400},
        ])
        res = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 3, "order": 3},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 2, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 1, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 3, "order": 3},
        ])
        ece, bins = compute_rt3_ece(rt3_pred, res)
        assert len(bins) == len(ECE_BIN_LABELS)
        by_label = {b["bin"]: b for b in bins}

        assert by_label["0-1%"]["count"] == 1
        assert by_label["0-1%"]["mean_pred"] == pytest.approx(0.004)
        assert by_label["0-1%"]["actual_rate"] == pytest.approx(0.0)
        assert by_label["0-1%"]["abs_gap"] == pytest.approx(0.004)

        assert by_label["1-2%"]["count"] == 2
        assert by_label["1-2%"]["mean_pred"] == pytest.approx(0.013)
        assert by_label["1-2%"]["actual_rate"] == pytest.approx(0.5)
        assert by_label["1-2%"]["abs_gap"] == pytest.approx(0.487)

        assert by_label["10%以上"]["count"] == 2
        assert by_label["10%以上"]["mean_pred"] == pytest.approx(0.300)
        assert by_label["10%以上"]["actual_rate"] == pytest.approx(0.5)
        assert by_label["10%以上"]["abs_gap"] == pytest.approx(0.200)

        for empty_label in ECE_BIN_LABELS[2:10]:
            assert by_label[empty_label]["count"] == 0
            assert by_label[empty_label]["mean_pred"] == 0.0
            assert by_label[empty_label]["actual_rate"] == 0.0
            assert by_label[empty_label]["abs_gap"] == 0.0

        expected_ece = (1 / 5) * 0.004 + (2 / 5) * 0.487 + (2 / 5) * 0.200
        assert ece == pytest.approx(expected_ece)

    def test_compute_rt3_logloss_and_hit_missing_actual_combo_race_set_mismatch(self) -> None:
        # Race 1 has actual combo "1-2-3" in predictions (and it's top-1).
        # Race 2 has predictions (so top-1 exists), but actual combo "2-1-3" is missing from predictions.
        rt3_pred = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "1-2-3", "combo_prob": 0.7},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "1-3-2", "combo_prob": 0.3},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "combo": "1-2-3", "combo_prob": 0.9},
        ])
        res = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 3, "order": 3},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 2, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 1, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 3, "order": 3},
        ])
        ll, hit_rate, n_races = compute_rt3_logloss_and_hit(rt3_pred, res)
        assert n_races == 1
        assert ll == pytest.approx(-math.log(0.7))
        # On the 1 evaluated race in merged_hit, top-1 hit rate should be 1.0, not 0.5
        assert hit_rate == pytest.approx(1.0)

        # When every race lacks the actual combo, return (nan, nan, 0)
        rt3_pred_no_hit = rt3_pred[rt3_pred["race_no"] == 2].copy()
        ll_empty, hit_empty, n_empty = compute_rt3_logloss_and_hit(rt3_pred_no_hit, res)
        assert math.isnan(ll_empty) and math.isnan(hit_empty) and n_empty == 0


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

        # Independent NLL helper for two 2-car groups with logits [2, 0] each
        def manual_temp_nll(logits_arr: np.ndarray, labels_arr: np.ndarray, groups_arr: np.ndarray, temp: float) -> float:
            nlls: list[float] = []
            idx = 0
            for g_size in groups_arr:
                g_logits = logits_arr[idx : idx + g_size] / temp
                g_labels = labels_arr[idx : idx + g_size]
                shifted = g_logits - np.max(g_logits)
                exp_l = np.exp(shifted)
                probs_g = exp_l / np.sum(exp_l)
                for p, y in zip(probs_g, g_labels):
                    if y == 1.0:
                        nlls.append(-math.log(max(float(p), 1e-15)))
                idx += g_size
            return float(np.mean(nlls))

        # Case 1: Higher-logit car always wins -> sharp distribution (T < 1.0) minimizes NLL
        logits_cal = np.array([2.0, 0.0, 2.0, 0.0])
        groups_cal = np.array([2, 2])
        labels_sharp = np.array([1.0, 0.0, 1.0, 0.0])
        t_low = fit_temperature(logits_cal, labels_sharp, groups_cal)
        assert 0.1 <= t_low < 1.0
        assert manual_temp_nll(logits_cal, labels_sharp, groups_cal, t_low) < manual_temp_nll(
            logits_cal, labels_sharp, groups_cal, 1.0
        )

        # Case 2: Lower-logit car always wins -> flat distribution (T > 1.0) minimizes NLL
        labels_overconf = np.array([0.0, 1.0, 0.0, 1.0])
        t_high = fit_temperature(logits_cal, labels_overconf, groups_cal)
        assert 1.0 < t_high <= 10.0
        assert manual_temp_nll(logits_cal, labels_overconf, groups_cal, t_high) < manual_temp_nll(
            logits_cal, labels_overconf, groups_cal, 1.0
        )

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

    def test_actual_rt3_dead_heat_is_row_order_independent(self) -> None:
        # 設計書どおり、同着の選手はどちらもその着順として扱う：当たり目は複数になる
        # 1着同着（1号車・4号車）。次の車を「2着」と数える記録
        res_order_a = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 4, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 3, "order": 3},
        ])
        res_order_b = res_order_a.iloc[[1, 0, 2, 3]].reset_index(drop=True)
        out_a = actual_rt3(res_order_a)
        out_b = actual_rt3(res_order_b)
        assert list(out_a["combo"]) == ["1-4-2", "4-1-2"]
        assert list(out_a["combo"]) == list(out_b["combo"])

        # 同じ1着同着を、次の車を「3着」と数える記録で表しても同じ結果になる
        res_skip = res_order_a.copy()
        res_skip.loc[res_skip["car_no"] == 2, "order"] = 3
        res_skip.loc[res_skip["car_no"] == 3, "order"] = 4
        assert list(actual_rt3(res_skip)["combo"]) == ["1-4-2", "4-1-2"]

        # レース1は3着同着（3号車・4号車）で当たり目2つ、レース2は通常で1つ
        mixed = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 3, "order": 3},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 4, "order": 3},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 5, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 2, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 2, "car_no": 1, "order": 3},
        ])
        out_mixed = actual_rt3(mixed)
        got = sorted(zip(out_mixed["race_no"].astype(int), out_mixed["combo"]))
        assert got == [(1, "1-2-3"), (1, "1-2-4"), (2, "5-2-1")]

    def test_dead_heat_counts_every_winning_combo_in_rt3_metrics(self) -> None:
        # 3着同着のレース：当たり目は 1-2-3 と 1-2-4。確率の合計で LogLoss、1位の組がどちらかなら的中
        res = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "order": 1},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "order": 2},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 3, "order": 3},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 4, "order": 3},
        ])
        pred = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "1-2-4", "combo_prob": 0.30},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "1-2-3", "combo_prob": 0.20},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "combo": "2-1-3", "combo_prob": 0.50},
        ])
        ll, hit, n = compute_rt3_logloss_and_hit(pred, res)
        assert n == 1
        assert ll == pytest.approx(-math.log(0.30 + 0.20))
        assert hit == 0.0  # 1位の組 2-1-3 は当たり目ではない
        pred.loc[pred["combo"] == "2-1-3", "combo_prob"] = 0.10
        _, hit2, _ = compute_rt3_logloss_and_hit(pred, res)
        assert hit2 == 1.0  # 1位の組 1-2-4 は当たり目の1つ

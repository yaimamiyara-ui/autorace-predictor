# tests/test_features_offline.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autorace.model.features import build_features, compute_PriorPlayerStats, parse_rank_series
from autorace.model.features_v1 import (
    FEATURE_COLS,
    _classify_race_name,
    _compute_intra_race_wall_and_ahead,
    _compute_meeting_features,
    _compute_player_prior_rolling,
    _norm_keys,
    build_features_v1,
)


class TestFeaturesV0:
    def test_parse_rank_series_normal_and_invalid(self) -> None:
        s = pd.Series(["S-1", "S-48", "A-1", "A-230", "B-1", "B-117", "", None, "C-10", "S-", "A-12a"])
        res = parse_rank_series(s)
        expected = [1.0, 48.0, 1001.0, 1230.0, 2001.0, 2117.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0]
        np.testing.assert_allclose(res.to_numpy(), expected, atol=1e-12)

    def test_compute_prior_player_stats_empty_and_no_valid_rows(self) -> None:
        expected_cols = ["player_code", "stat_date", "prior_win_rate", "prior_top3_rate", "prior_races"]
        empty_res = compute_PriorPlayerStats(pd.DataFrame(columns=["player_code", "race_date", "place_code", "race_no", "order", "accident_code"]))
        assert empty_res.empty
        assert list(empty_res.columns) == expected_cols

        # Scratched / unstarted rows where both order and accident_code are missing
        no_start = pd.DataFrame([
            {"player_code": 9001, "race_date": "2026-11-01", "place_code": 2, "race_no": 1, "order": np.nan, "accident_code": ""},
            {"player_code": 9002, "race_date": "2026-11-01", "place_code": 2, "race_no": 1, "order": None, "accident_code": None},
        ])
        assert compute_PriorPlayerStats(no_start).empty

    def test_compute_prior_player_stats_rolling_window_and_accident(self) -> None:
        rows = [
            {"player_code": 9001, "race_date": "2026-11-01", "place_code": 2, "race_no": 1, "order": 1, "accident_code": None},
            {"player_code": 9001, "race_date": "2026-11-01", "place_code": 2, "race_no": 5, "order": 3, "accident_code": None},
            {"player_code": 9001, "race_date": "2026-11-02", "place_code": 2, "race_no": 2, "order": np.nan, "accident_code": "1"},
            {"player_code": 9001, "race_date": "2026-11-03", "place_code": 2, "race_no": 1, "order": 2, "accident_code": None},
        ]
        stats = compute_PriorPlayerStats(pd.DataFrame(rows), window=2)
        assert len(stats) == 3
        d1 = stats[stats["stat_date"] == "2026-11-01"].iloc[0]
        assert d1["prior_races"] == 2.0
        assert d1["prior_win_rate"] == pytest.approx(0.5)
        assert d1["prior_top3_rate"] == pytest.approx(1.0)

        # Window=2 on day 2 uses (day1 R5: order=3) and (day2 R2: accident -> not win, not top3)
        d2 = stats[stats["stat_date"] == "2026-11-02"].iloc[0]
        assert d2["prior_races"] == 2.0
        assert d2["prior_win_rate"] == pytest.approx(0.0)
        assert d2["prior_top3_rate"] == pytest.approx(0.5)

    def test_build_features_empty_and_all_absent(self) -> None:
        empty_prog = pd.DataFrame(columns=["place_code", "race_date", "race_no", "car_no", "player_code", "handicap", "trial_run_time", "absent", "rank"])
        assert build_features(empty_prog).empty

        all_absent = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "player_code": 9001,
             "handicap": 0, "trial_run_time": None, "absent": "欠場", "rank": "S-1"},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "player_code": 9002,
             "handicap": 10, "trial_run_time": None, "absent": "1", "rank": "A-10"},
        ])
        res = build_features(all_absent)
        assert len(res) == 2
        assert res["is_absent"].all()
        assert (res["max_handicap_diff"] == 0.0).all()
        assert (res["prior_races"] == 0.0).all()

    def test_build_features_handicap_trial_and_strict_prior_join(self) -> None:
        prog = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-02", "race_no": 1, "car_no": 1, "player_code": 9001,
             "handicap": 0, "trial_run_time": 3.35, "absent": None, "rank": "A-20"},
            {"place_code": 2, "race_date": "2026-11-02", "race_no": 1, "car_no": 2, "player_code": 9002,
             "handicap": 0, "trial_run_time": 3.30, "absent": "", "rank": "S-5"},
            {"place_code": 2, "race_date": "2026-11-02", "race_no": 1, "car_no": 3, "player_code": 9003,
             "handicap": 10, "trial_run_time": 0.0, "absent": "欠場", "rank": "S-1"},
        ])
        hist = pd.DataFrame([
            # Strict prior day -> should be joined
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "player_code": 9001, "order": 1, "accident_code": None},
            # Same day -> must NOT be joined
            {"place_code": 2, "race_date": "2026-11-02", "race_no": 1, "car_no": 2, "player_code": 9002, "order": 1, "accident_code": None},
        ])
        feat = build_features(prog, hist).set_index("car_no")
        assert feat.loc[1, "same_handicap_inner_order"] == 1
        assert feat.loc[2, "same_handicap_inner_order"] == 2
        assert feat.loc[1, "same_handicap_count"] == 2
        assert feat.loc[1, "is_all_same_handicap"] == 1  # Car 3 (handicap 10) is absent, so active cars all have handicap 0
        assert feat.loc[2, "trial_rank_in_race"] == 1.0
        assert feat.loc[1, "trial_rank_in_race"] == 2.0
        assert feat.loc[1, "prior_win_rate"] == pytest.approx(1.0)
        assert pd.isna(feat.loc[2, "prior_win_rate"])
        assert feat.loc[2, "prior_races"] == 0.0
        # Absent car defaults
        assert feat.loc[3, "trial_rank_in_race"] == 4.5
        assert feat.loc[3, "rank_order_in_race"] == 4.5


class TestFeaturesV1:
    def test_classify_race_name(self) -> None:
        s = pd.Series(["一般戦", "特別予選", "準決勝戦A", "優勝戦", None, "", "予選兼準決", "準決兼優勝"])
        res = _classify_race_name(s)
        assert list(res) == [0, 1, 2, 3, 0, 0, 2, 3]

    def test_norm_keys(self) -> None:
        df = pd.DataFrame({"race_date": [20261101, "2026-11-02"], "place_code": ["02", None], "race_no": ["5", "bad"]})
        out = _norm_keys(df.copy())
        assert list(out["race_date"]) == ["20261101", "2026-11-02"]
        assert list(out["place_code"]) == [2, 0]
        assert list(out["race_no"]) == [5, 0]

    def test_intra_race_wall_and_ahead(self) -> None:
        df = pd.DataFrame({
            "place_code": [2, 2, 2, 2],
            "race_date": ["2026-11-01"] * 4,
            "race_no": [1] * 4,
            "_handicap_num": [0.0, 0.0, 10.0, 10.0],
            "_car_no_num": [1.0, 2.0, 3.0, 4.0],
            "_valid_trial": [3.35, 3.30, 3.32, np.nan],
            "is_absent": [False, False, False, True],
        })
        wall = _compute_intra_race_wall_and_ahead(df)
        # Car 1: nobody ahead
        assert wall.loc[0, "num_cars_ahead"] == 0.0
        assert wall.loc[0, "wall_slow_cars_ahead"] == 0.0
        assert wall.loc[0, "wall_best_trial_diff_ahead"] == 0.0
        # Car 2 (h=0, c=2, t=3.30): Car 1 (h=0, c=1, t=3.35) is ahead and slower
        assert wall.loc[1, "num_cars_ahead"] == 1.0
        assert wall.loc[1, "wall_slow_cars_ahead"] == 1.0
        assert wall.loc[1, "wall_best_trial_diff_ahead"] == pytest.approx(3.35 - 3.30)
        # Car 3 (h=10, c=3, t=3.32): Cars 1 and 2 ahead; Car 1 (3.35) is slower, best ahead is Car 2 (3.30)
        assert wall.loc[2, "num_cars_ahead"] == 2.0
        assert wall.loc[2, "wall_slow_cars_ahead"] == 1.0
        assert wall.loc[2, "wall_best_trial_diff_ahead"] == pytest.approx(3.30 - 3.32)
        # Car 4 is absent -> all 0
        assert wall.loc[3, "num_cars_ahead"] == 0.0
        assert wall.loc[3, "wall_slow_cars_ahead"] == 0.0
        assert wall.loc[3, "wall_best_trial_diff_ahead"] == 0.0

    def test_player_prior_rolling_wet_fallback_and_defaults(self) -> None:
        prog = pd.DataFrame([
            {"player_code": 9001, "race_date": "2026-11-05"},
            {"player_code": 9099, "race_date": "2026-11-05"},
        ])
        # Empty history returns defaults
        empty_prior = _compute_player_prior_rolling(prog, None, None)
        assert list(empty_prior["prior_st_10"]) == [0.15, 0.15]
        assert list(empty_prior["prior_time_gap_10"]) == [0.0, 0.0]

        # History without race_info uses trial_time >= 3.45 as wet track fallback
        hist = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "player_code": 9001,
             "trial_time": 3.30, "race_time": 3.38, "st": 0.12, "order": 4},
            {"place_code": 2, "race_date": "2026-11-02", "race_no": 1, "player_code": 9001,
             "trial_time": 3.50, "race_time": 3.60, "st": 0.16, "order": 1},
        ])
        prior = _compute_player_prior_rolling(prog, hist, None)
        assert prior.loc[0, "prior_time_gap_10"] == pytest.approx((0.08 + 0.10) / 2.0)
        assert prior.loc[0, "prior_st_10"] == pytest.approx((0.12 + 0.16) / 2.0)
        assert prior.loc[0, "prior_wet_races"] == 1.0
        assert prior.loc[0, "wet_order_diff"] == pytest.approx(1.0 - 4.0)
        # Unseen player 9099 gets defaults
        assert prior.loc[1, "prior_st_10"] == pytest.approx(0.15)
        assert prior.loc[1, "prior_wet_races"] == 0.0

    def test_compute_meeting_features(self) -> None:
        df = pd.DataFrame([
            {"place_code": 2, "period_start_date": "2026-11-01", "player_code": 9001, "race_date": "2026-11-03", "_valid_trial": 3.31},
            {"place_code": 2, "period_start_date": "2026-11-01", "player_code": 9002, "race_date": "2026-11-03", "_valid_trial": 3.35},
        ])
        info = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "period_start_date": "2026-11-01"},
            {"place_code": 2, "race_date": "2026-11-02", "race_no": 1, "period_start_date": "2026-11-01"},
        ])
        hist = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "player_code": 9001, "order": 2, "trial_time": 3.33},
            {"place_code": 2, "race_date": "2026-11-02", "race_no": 1, "player_code": 9001, "order": 4, "trial_time": 3.35},
        ])
        m = _compute_meeting_features(df, hist, info)
        assert m.loc[0, "meeting_prior_races"] == 2.0
        assert m.loc[0, "meeting_prior_avg_order"] == pytest.approx(3.0)
        assert m.loc[0, "meeting_trial_diff"] == pytest.approx(3.31 - 3.34)
        assert m.loc[1, "meeting_prior_races"] == 0.0
        assert m.loc[1, "meeting_prior_avg_order"] == pytest.approx(4.5)
        assert m.loc[1, "meeting_trial_diff"] == 0.0

    def test_build_features_v1_without_race_info_or_history(self) -> None:
        prog = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "player_code": 9001,
             "handicap": 0, "trial_run_time": 3.32, "trial_retry_code": "1", "absent": None, "rank": "A-10"},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 2, "player_code": 9002,
             "handicap": 10, "trial_run_time": 3.28, "trial_retry_code": "0", "absent": None, "rank": "S-2"},
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 3, "player_code": 9003,
             "handicap": 20, "trial_run_time": None, "trial_retry_code": None, "absent": "欠場", "rank": "B-5"},
        ])
        out = build_features_v1(prog, race_info=None, race_result_hist=None)
        for col in FEATURE_COLS:
            assert col in out.columns
        assert list(out["is_retrial"]) == [1.0, 0.0, 0.0]
        assert list(out["active_car_count"]) == [2.0, 2.0, 2.0]
        assert list(out["race_max_handicap_span"]) == [10.0, 10.0, 10.0]
        assert out.loc[0, "rel_handicap"] == pytest.approx(0.0)
        assert out.loc[1, "rel_handicap"] == pytest.approx(1.0)

    def test_build_features_v1_uses_only_prerace_info_no_postrace_leakage(self) -> None:
        prog = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-02", "race_no": 1, "car_no": 1, "player_code": 9001,
             "handicap": 0, "trial_run_time": 3.30, "trial_retry_code": "0", "absent": None, "rank": "S-1"},
            {"place_code": 2, "race_date": "2026-11-02", "race_no": 1, "car_no": 2, "player_code": 9002,
             "handicap": 10, "trial_run_time": 3.35, "trial_retry_code": "0", "absent": None, "rank": "A-5"},
        ])
        hist = pd.DataFrame([
            {"place_code": 2, "race_date": "2026-11-01", "race_no": 1, "car_no": 1, "player_code": 9001,
             "trial_time": 3.32, "race_time": 3.40, "st": 0.12, "order": 1, "accident_code": None},
        ])
        # Pre-race values: road_temp=20.0, situation_code=0 (good track), temp=18.0, humid=50.0, weather="晴"
        # Post-race race_* values differ sharply: race_road_temp=45.0, race_situation_code=2 (wet), etc.
        info_a = pd.DataFrame([
            {
                "place_code": 2, "race_date": "2026-11-01", "race_no": 1,
                "distance": 3100, "situation_code": 0, "temp": 18.0, "humid": 50.0, "road_temp": 20.0, "weather": "晴",
                "race_situation_code": 2, "race_temp": 9.0, "race_humid": 95.0, "race_road_temp": 45.0, "race_weather": "雨",
                "grade_code": 1, "paragraph_day": 1, "period_start_date": "2026-11-01", "race_name": "予選", "final_race_no": 12,
            },
            {
                "place_code": 2, "race_date": "2026-11-02", "race_no": 1,
                "distance": 3100, "situation_code": 0, "temp": 18.0, "humid": 50.0, "road_temp": 20.0, "weather": "晴",
                "race_situation_code": 2, "race_temp": 9.0, "race_humid": 95.0, "race_road_temp": 45.0, "race_weather": "雨",
                "grade_code": 1, "paragraph_day": 2, "period_start_date": "2026-11-01", "race_name": "準決勝戦", "final_race_no": 12,
            },
        ])
        # Mutate post-race race_* columns completely and verify features are identical
        info_b = info_a.copy()
        info_b["race_road_temp"] = 99.0
        info_b["race_situation_code"] = 0
        info_b["race_temp"] = -5.0
        info_b["race_humid"] = 10.0
        info_b["race_weather"] = "雪"

        out_a = build_features_v1(prog, race_info=info_a, race_result_hist=hist)
        out_b = build_features_v1(prog, race_info=info_b, race_result_hist=hist)

        # Confirm pre-race values reached the features (not post-race race_* values)
        assert list(out_a["road_temp_clean"]) == [20.0, 20.0]
        assert list(out_a["road_temp_x_handicap"]) == [0.0, 200.0]
        assert list(out_a["is_non_good_track"]) == [0.0, 0.0]
        assert list(out_a["prior_wet_races"]) == [0.0, 0.0]
        for r_col in ("race_road_temp", "race_situation_code", "race_temp", "race_humid", "race_weather"):
            assert r_col not in out_a.columns
        pd.testing.assert_frame_equal(out_a[FEATURE_COLS], out_b[FEATURE_COLS])

    @pytest.mark.xfail(
        strict=True,
        raises=KeyError,
        reason="build_features returns empty program without v0 columns (e.g. is_absent), causing KeyError in build_features_v1",
    )
    def test_build_features_v1_empty_program_returns_empty_dataframe(self) -> None:
        empty_prog = pd.DataFrame(
            columns=["place_code", "race_date", "race_no", "car_no", "player_code", "handicap", "trial_run_time", "absent", "rank"]
        )
        out = build_features_v1(empty_prog, race_info=None, race_result_hist=None)
        assert out.empty

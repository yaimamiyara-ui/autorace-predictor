# tests/test_client_parsing.py
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from autorace.common.odds import (
    _extract_combo,
    _extract_range_leaf,
    _format_combo,
    _parse_float,
    _parse_int,
    parse_odds_body,
    parse_refund_body,
)
from autorace.common.schema import (
    HoldInfo,
    OddsEntry,
    OddsPlayerEntry,
    ProgramEntry,
    RaceInfoEntry,
    RefundEntry,
    ResultEntry,
    _empty_to_none,
    _to_bool_or_none,
    _to_code_str,
)
from autorace.fetch.client import AutoraceAPIError
from autorace.scripts.backfill import date_range, parse_kinds, parse_places
from autorace.scripts.live_record import (
    _has_trial_run_time,
    _is_absent,
    _parse_optional_int,
    compute_minutes_to_close,
    is_trial_announced,
    resolve_race_date,
)

JST = ZoneInfo("Asia/Tokyo")


class TestAutoraceAPIError:
    def test_error_attributes_and_defaults(self) -> None:
        err1 = AutoraceAPIError("Request failed")
        assert str(err1) == "Request failed"
        assert err1.errors == []
        assert err1.response_data is None

        raw_resp = {"result": "Failure", "errors": ["E001"]}
        err2 = AutoraceAPIError("API Failure", errors=["E001"], response_data=raw_resp)
        assert err2.errors == ["E001"]
        assert err2.response_data == raw_resp


class TestSchemaHelpersAndModels:
    def test_empty_to_none_and_bool_and_code_str(self) -> None:
        for val in (None, "", "  ", "-", "---"):
            assert _empty_to_none(val) is None
            assert _to_bool_or_none(val) is None
            assert _to_code_str(val) is None

        assert _to_bool_or_none(True) is True
        assert _to_bool_or_none(False) is False
        assert _to_bool_or_none(1) is True
        assert _to_bool_or_none(0) is False
        for t_str in ("1", "true", "TRUE", "yes", "on"):
            assert _to_bool_or_none(t_str) is True
        for f_str in ("0", "false", "FALSE", "no", "off"):
            assert _to_bool_or_none(f_str) is False

        assert _to_code_str(0) == "0"
        assert _to_code_str(12) == "12"
        assert _to_code_str(" 01 ") == "01"

    def test_program_and_result_malformed_inputs(self) -> None:
        assert ProgramEntry.from_api_response(2, "2026-11-01", 1, None) == []  # type: ignore[arg-type]
        assert ProgramEntry.from_api_response(2, "2026-11-01", 1, {"playerList": "not_a_list"}) == []
        assert ResultEntry.from_api_response(2, "2026-11-01", 1, None) == []  # type: ignore[arg-type]
        assert ResultEntry.from_api_response(2, "2026-11-01", 1, {"raceResult": None}) == []
        assert RaceInfoEntry.from_api_response(2, "2026-11-01", 1, None) is None
        assert RaceInfoEntry.from_api_response(2, "2026-11-01", 1, []) is None

        # Entries missing playerCode or non-dict items are skipped
        prog_body = {
            "playerList": [
                "invalid_item",
                {"carNo": 1, "playerCode": "", "playerName": "テスト選手A"},
                {"carNo": 2, "playerCode": "9001", "playerName": "テスト選手B", "placeCode": 3, "retireFlag": "false"},
            ]
        }
        progs = ProgramEntry.from_api_response(6, "2026-11-01", 1, prog_body)
        assert len(progs) == 1
        assert progs[0].player_code == 9001
        assert progs[0].place_code == 6
        assert progs[0].branch_place_code == 3
        assert progs[0].retire_flag is False

        # ResultEntry order<=0 or non-numeric order becomes None
        res_body = {
            "raceResult": [
                {"carNo": 1, "playerCode": "9001", "order": "-1"},
                {"carNo": 2, "playerCode": "9002", "order": "失格"},
                {"carNo": 3, "playerCode": "9003", "order": "2"},
            ]
        }
        results = ResultEntry.from_api_response(6, "2026-11-01", 1, res_body)
        assert [r.order for r in results] == [None, None, 2]


class TestOddsAndRefundParsingHelpers:
    def test_parse_float_and_int(self) -> None:
        for bad in (None, True, False, 0, -1.5, "", "-", "---", "欠", "特払", "不成立", "0", "-3.2", "abc"):
            assert _parse_float(bad) is None
        assert _parse_float("1,234.5") == 1234.5
        assert _parse_float(3.25) == 3.25

        for bad_int in (None, True, False, "", "-", "---", "欠", "abc"):
            assert _parse_int(bad_int) is None
        assert _parse_int("2,410円") == 2410
        assert _parse_int("11番人気") == 11
        assert _parse_int(5.0) == 5

    def test_format_combo_boundaries(self) -> None:
        assert _format_combo(["3", "1", "2"], unordered=False) == "3-1-2"
        assert _format_combo(["3", "1", "2"], unordered=True) == "1-2-3"
        assert _format_combo(["0", "1"], unordered=False) is None
        assert _format_combo(["1", "9"], unordered=False) is None
        assert _format_combo(["2", "2"], unordered=False) is None
        assert _format_combo(["1", "x"], unordered=False) is None

    def test_extract_range_leaf_shapes(self) -> None:
        assert _extract_range_leaf("1.5") == (1.5, None, True)
        assert _extract_range_leaf([]) == (None, None, True)
        assert _extract_range_leaf(["2.1"]) == (2.1, None, True)
        assert _extract_range_leaf(["3.5", "1.8"]) == (1.8, 3.5, True)
        assert _extract_range_leaf([1, 2, 3]) == (None, None, False)
        assert _extract_range_leaf({}) == (None, None, True)
        assert _extract_range_leaf({"min": "2.5", "max": "1.2"}) == (1.2, 2.5, True)
        assert _extract_range_leaf({"a": "4.0", "b": "2.0"}) == (2.0, 4.0, True)
        assert _extract_range_leaf({"nested": {"a": 1}}) == (None, None, False)

    def test_parse_odds_body_edge_cases(self) -> None:
        odds, players, unknown = parse_odds_body(2, "2026-11-01", 1, "not_a_dict")  # type: ignore[arg-type]
        assert odds == []
        assert players == []
        assert unknown == {"_body": "not_a_dict"}

        body = {
            "rfwOddsList": {"2": {"1": "5.5"}, "1": {"2": "5.5"}},  # unordered duplicate
            "tnsOddsList": {"9": "2.0", "1": "1.5"},  # car 9 invalid
            "playerList": [{"playerCode": "9001", "carNo": "invalid_car"}],
        }
        odds, players, unknown = parse_odds_body(2, "2026-11-01", 1, body)
        rfw = [o for o in odds if o.bet_type == "rfw"]
        assert len(rfw) == 1
        assert rfw[0].combo == "1-2"
        assert "tnsOddsList" in unknown
        assert "playerList_errors" in unknown

    def test_extract_combo_and_heuristic_refund_body(self) -> None:
        assert _extract_combo({"kumiban": "5-3-1"}, "rt3") == "5-3-1"
        assert _extract_combo({"kumiban": "5-3-1"}, "rf3") == "1-3-5"
        assert _extract_combo({"1st_car": 4, "2nd_car": 2}, "rtw") == "4-2"
        assert _extract_combo({"focus": "6=2"}, "rfw") == "2-6"

        heuristic_body = {
            "refundList": [
                {
                    "raceNo": "3",
                    "refundInfo": {
                        "rt3": [{"kumiban": "4-1-2", "dividend": "3,250円", "ninki": "12番人気"}],
                        "rfw": {"5=2": {"pay": 680, "rank": 3}},
                    },
                }
            ]
        }
        entries, unparsed = parse_refund_body(2, "2026-11-01", heuristic_body)
        assert unparsed == []
        by_bt = {e.bet_type: e for e in entries}
        assert by_bt["rt3"].combo == "4-1-2"
        assert by_bt["rt3"].payout == 3250
        assert by_bt["rt3"].popularity == 12
        assert by_bt["rfw"].combo == "2-5"
        assert by_bt["rfw"].payout == 680

    def test_parse_odds_and_refund_single_schema_definitions(self) -> None:
        import inspect
        import autorace.common.schema as schema_mod

        src = inspect.getsource(schema_mod)
        assert src.count("class OddsEntry(") == 1
        assert src.count("class OddsPlayerEntry(") == 1
        assert src.count("class RefundEntry(") == 1

        odds_body = {
            "tnsOddsList": {"1": "2.4", "2": "3.8"},
            "fnsOddsList": {"1": {"min": "1.1", "max": "1.5"}},
            "widOddsList": {"2": {"1": ["2.2", "1.4"]}},
            "rt3OddsList": {"1": {"2": {"3": "15.6"}}},
            "playerList": [
                {
                    "carNo": "1",
                    "playerCode": "9001",
                    "placeCode": 2,
                    "handicap": "10",
                    "trialTime": "3.30",
                    "trialRetryCode": 0,
                    "absent": "-",
                    "raceDev": "0.05",
                    "stAve": "0.13",
                    "goodTrackTraialAve": "3.31",
                    "goodTrackRaceAve": "3.39",
                    "goodTrackRaceBest": "3.35",
                    "goodTrackRaceBestPlace": 2,
                    "sunnyExpectCode": 1,
                    "rainExpectCode": "",
                    "stExpectCode": "2",
                    "aiExpectCode": "---",
                }
            ],
        }
        odds, players, unknown = parse_odds_body(6, "2026-11-01", 5, odds_body, snapshot="pre")
        assert unknown == {}
        assert all(isinstance(o, OddsEntry) for o in odds)
        by_key = {(o.bet_type, o.combo): o for o in odds}
        assert by_key[("tns", "1")].model_dump() == {
            "place_code": 6,
            "race_date": "2026-11-01",
            "race_no": 5,
            "bet_type": "tns",
            "combo": "1",
            "odds": 2.4,
            "odds_max": None,
            "snapshot": "pre",
        }
        assert by_key[("fns", "1")].odds == 1.1
        assert by_key[("fns", "1")].odds_max == 1.5
        assert by_key[("wid", "1-2")].odds == 1.4
        assert by_key[("wid", "1-2")].odds_max == 2.2
        assert by_key[("rt3", "1-2-3")].odds == 15.6

        assert len(players) == 1
        assert isinstance(players[0], OddsPlayerEntry)
        assert players[0].model_dump() == {
            "place_code": 6,
            "race_date": "2026-11-01",
            "race_no": 5,
            "player_code": 9001,
            "car_no": 1,
            "handicap": 10.0,
            "trial_time": 3.30,
            "trial_retry_code": "0",
            "absent": None,
            "race_dev": 0.05,
            "st_ave": 0.13,
            "good_track_trial_ave": 3.31,
            "good_track_race_ave": 3.39,
            "good_track_race_best": 3.35,
            "good_track_race_best_place": "2",
            "sunny_expect_code": "1",
            "rain_expect_code": None,
            "st_expect_code": "2",
            "ai_expect_code": None,
        }

        refund_body = [
            {
                "raceNo": 5,
                "refundInfo": {
                    "rt3": {
                        "typeCode": 0,
                        "typeName": "通常",
                        "list": [{"1thCarNo": 1, "2thCarNo": 2, "3thCarNo": 3, "refund": 1560, "pop": 4}],
                    },
                    "wid": {
                        "typeCode": 0,
                        "typeName": "通常",
                        "list": [
                            {"1thCarNo": 2, "2thCarNo": 1, "refund": 140, "pop": 1},
                            {"1thCarNo": 3, "2thCarNo": 1, "refund": 290, "pop": 5},
                        ],
                    },
                },
            }
        ]
        refunds, raw = parse_refund_body(6, "2026-11-01", refund_body)
        assert raw == []
        assert all(isinstance(r, RefundEntry) for r in refunds)
        assert [r.model_dump() for r in refunds] == [
            {
                "place_code": 6,
                "race_date": "2026-11-01",
                "race_no": 5,
                "bet_type": "rt3",
                "slot": 1,
                "combo": "1-2-3",
                "payout": 1560,
                "popularity": 4,
            },
            {
                "place_code": 6,
                "race_date": "2026-11-01",
                "race_no": 5,
                "bet_type": "wid",
                "slot": 1,
                "combo": "1-2",
                "payout": 140,
                "popularity": 1,
            },
            {
                "place_code": 6,
                "race_date": "2026-11-01",
                "race_no": 5,
                "bet_type": "wid",
                "slot": 2,
                "combo": "1-3",
                "payout": 290,
                "popularity": 5,
            },
        ]


class TestLiveRecordAndBackfillHelpers:
    def test_live_record_pure_helpers(self) -> None:
        assert _parse_optional_int(None) is None
        assert _parse_optional_int(True) is None
        assert _parse_optional_int(float("nan")) is None
        assert _parse_optional_int(" 12 ") == 12
        assert _parse_optional_int("bad") is None

        assert _is_absent(None) is False
        assert _is_absent("0") is False
        assert _is_absent("false") is False
        assert _is_absent("欠場") is True

        assert _has_trial_run_time("3.31") is True
        assert _has_trial_run_time("0") is False
        assert _has_trial_run_time(None) is False

        p1 = ProgramEntry.model_validate({"place_code": 2, "race_date": "2026-11-01", "race_no": 1,
                                          "playerCode": 9001, "carNo": 1, "trialRunTime": 3.30, "absent": None})
        p2_abs = ProgramEntry.model_validate({"place_code": 2, "race_date": "2026-11-01", "race_no": 1,
                                              "playerCode": 9002, "carNo": 2, "trialRunTime": None, "absent": "欠場"})
        assert is_trial_announced([p1, p2_abs]) is True
        assert is_trial_announced([p2_abs]) is False

        # Midnight boundary: 04:59 is previous day, 05:00 is same day
        assert resolve_race_date(datetime(2026, 11, 13, 4, 59, tzinfo=JST)) == "2026-11-12"
        assert resolve_race_date(datetime(2026, 11, 13, 5, 0, tzinfo=JST)) == "2026-11-13"

        now = datetime(2026, 11, 12, 23, 50, tzinfo=JST)
        assert compute_minutes_to_close("00:10", "2026-11-12", now) == 20.0
        assert compute_minutes_to_close(None, "2026-11-12", now) is None
        assert compute_minutes_to_close("bad:time", "2026-11-12", now) is None

    def test_backfill_pure_helpers(self) -> None:
        assert parse_places(["2, 3", "5", "6,"]) == [2, 3, 5, 6]
        assert parse_kinds("program, odds ,RESULT") == {"program", "odds", "result"}
        with pytest.raises(ValueError, match="Invalid --kinds"):
            parse_kinds("program,unknown_kind")

        dates = date_range("2026-11-01", "2026-11-03")
        assert [d.isoformat() for d in dates] == ["2026-11-01", "2026-11-02", "2026-11-03"]
        with pytest.raises(ValueError, match="--start must be on or before --end"):
            date_range("2026-11-05", "2026-11-01")

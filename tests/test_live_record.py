"""live_record のテスト（実際のAPI応答の形を使う偽クライアントで確認）"""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from autorace.scripts import live_record

JST = ZoneInfo("Asia/Tokyo")


def _prog(trial):
    return {"playerList": [{"carNo": c, "playerCode": str(3000 + c), "placeCode": 2, "handicap": 10 * c,
                            "trialRunTime": trial, "absent": None, "rank": "A-1"} for c in range(1, 4)]}


def _odds():
    return {"tnsOddsList": {"1": "2.0", "2": "3.0", "3": "4.0"},
            "rt3OddsList": {"1": {"2": {"3": "10.0"}, "3": {"2": "11.0"}},
                            "2": {"1": {"3": "12.0"}, "3": {"1": "13.0"}},
                            "3": {"1": {"2": "14.0"}, "2": {"1": "15.0"}}}}


class Fake:
    def __init__(self, venue, progs):
        self.venue, self.progs, self.calls = venue, progs, []

    def hold_today(self):
        return {"result": "Success", "body": {"today": [self.venue]}}

    def program(self, p, d, r):
        self.calls.append(("program", p, r))
        return {"result": "Success", "body": self.progs.get(r, {"playerList": []})}

    def odds(self, p, d, r):
        self.calls.append(("odds", p, r))
        return {"result": "Success", "body": _odds()}

    def other_race_info(self, p, d, r):
        self.calls.append(("info", p, r))
        return {"result": "Success", "body": {"placeCode": p, "raceNo": r, "distance": 3100, "situationCode": 0,
                                              "temp": "16.0", "roadtemp": "20.0", "paragraphday": 1}}


class TestLiveRecord(unittest.TestCase):
    def test_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            venue = {"placeCode": 6, "nowRaceNo": 2, "resultRaceNo": 1, "telvoteTime": "11:25", "cancelFlg": "0"}
            fake = Fake(venue, {2: _prog("3.31"), 3: _prog("")})  # 2Rは試走発表済み、3Rは未発表
            now = datetime(2026, 11, 12, 11, 20, tzinfo=JST)
            s1 = live_record.run_once(client=fake, data_dir=tmp, now=now)
            self.assertEqual(s1["recorded_races"], 1)
            self.assertNotIn(("odds", 6, 1), fake.calls)   # 結果が出たレースは取らない
            self.assertNotIn(("odds", 6, 3), fake.calls)   # 試走未発表は取らない
            df = pd.read_parquet(Path(tmp) / "odds_pre/2026/2026-11.parquet")
            self.assertEqual(set(df["snapshot"]), {"pre"})
            self.assertEqual(df["place_code"].iloc[0], 6)
            self.assertEqual(set(df["minutes_to_close"]), {5.0})
            self.assertEqual(len(df[df.bet_type == "rt3"]), 6)
            s2 = live_record.run_once(client=fake, data_dir=tmp, now=now)  # 2回目は取り直さない
            self.assertEqual(s2["recorded_races"], 0)
            self.assertEqual(fake.calls.count(("odds", 6, 2)), 1)

    def test_midnight_date_and_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            venue = {"placeCode": 5, "nowRaceNo": 8, "resultRaceNo": 7, "telvoteTime": "00:20", "cancelFlg": "0"}
            fake = Fake(venue, {8: _prog("3.30")})
            s = live_record.run_once(client=fake, data_dir=tmp, now=datetime(2026, 11, 13, 0, 15, tzinfo=JST))
            self.assertEqual(s["recorded_races"], 1)
            meta = pd.read_parquet(Path(tmp) / "live_meta/2026/2026-11.parquet")
            self.assertEqual(list(meta["race_date"]), ["2026-11-12"])  # 開催日で記録
            self.assertEqual(list(meta["minutes_to_close"]), [5.0])
            self.assertEqual(list(meta["ri_distance"]), [3100])      # レース情報も一緒に記録
            self.assertEqual(list(meta["ri_situation_code"]), [0])
            fake2 = Fake({"placeCode": 6, "cancelFlg": "1"}, {})
            live_record.run_once(client=fake2, data_dir=tmp, now=datetime(2026, 11, 12, 12, 0, tzinfo=JST))
            self.assertEqual(fake2.calls, [])


if __name__ == "__main__":
    unittest.main()

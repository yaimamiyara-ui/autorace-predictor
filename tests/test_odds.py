import unittest
from autorace.common.odds import parse_odds_body, parse_refund_body


class TestOdds(unittest.TestCase):
    def test_parse_odds_body(self):
        body = {
            "statusCode": "1",
            "tnsOddsList": {"1": "2.3", "2": "---", "3": "欠"},
            "fnsOddsList": {"1": {"min": "1.1", "max": "1.5"}, "2": {"lower": "---", "upper": "---"}},
            "rtwOddsList": {"1": {"2": "12.3", "3": "---"}, "2": {"1": "15.4"}},
            "rfwOddsList": {"2": {"1": "6.8"}},
            "rt3OddsList": {"1": {"2": {"3": "123.4"}}},
            "rf3OddsList": {"3": {"1": {"2": "25.0"}}},
            "widOddsList": {"2": {"1": {"odds1": "2.1", "odds2": "2.8"}}},
            "playerList": [{"carNo": "1", "playerCode": "2901", "handicap": "0", "trialTime": "3.29",
                            "stAve": "0.13", "goodTrackTraialAve": "3.30", "goodTrackRaceBestPlace": "2"}],
        }
        odds, players, unknown = parse_odds_body(2, "2026-11-12", 12, body)
        self.assertEqual(unknown, {})
        self.assertEqual(players[0].st_ave, 0.13)
        k = {(o.bet_type, o.combo): o for o in odds}
        self.assertEqual(k[("tns", "1")].odds, 2.3)
        self.assertIsNone(k[("tns", "2")].odds)
        self.assertEqual((k[("fns", "1")].odds, k[("fns", "1")].odds_max), (1.1, 1.5))
        self.assertEqual(k[("rtw", "1-2")].odds, 12.3)
        self.assertEqual(k[("rfw", "1-2")].odds, 6.8)
        self.assertEqual(k[("rt3", "1-2-3")].odds, 123.4)
        self.assertEqual(k[("rf3", "1-2-3")].odds, 25.0)
        self.assertEqual((k[("wid", "1-2")].odds, k[("wid", "1-2")].odds_max), (2.1, 2.8))

    def test_full_trifecta_count(self):
        # 実データと同じ形：3連単は 8×7×6 = 336 通り
        rt3 = {str(a): {str(b): {str(c): "10.0" for c in range(1, 9) if c not in (a, b)}
                        for b in range(1, 9) if b != a} for a in range(1, 9)}
        odds, _, unknown = parse_odds_body(6, "2015-11-12", 11, {"rt3OddsList": rt3})
        self.assertEqual(len(odds), 336)
        self.assertEqual(unknown, {})

    def test_parse_refund_body_heuristic(self):
        body = [{"raceNo": "1", "refundInfo": {
            "tns": {"carNo": "1", "refund": "230", "pop": "1"},
            "rtw": {"carNo1": "1", "carNo2": "3", "payout": "1,450", "popularity": "5"},
            "rt3": {"1": {"kumi": "1-3-5", "price": "8,920", "pop": "24"}},
            "wid": [{"combo": "1-3", "pay": "250", "pop": "2"}, {"combo": "5-1", "pay": "540", "pop": "8"}],
            "absent": []}}]
        entries, unparsed = parse_refund_body(2, "2026-11-12", body)
        self.assertEqual(unparsed, [])
        k = {(e.bet_type, e.slot): e for e in entries}
        self.assertEqual((k[("tns", 1)].combo, k[("tns", 1)].payout), ("1", 230))
        self.assertEqual((k[("rtw", 1)].combo, k[("rtw", 1)].payout), ("1-3", 1450))
        self.assertEqual((k[("rt3", 1)].combo, k[("rt3", 1)].payout), ("1-3-5", 8920))
        self.assertEqual(k[("wid", 2)].combo, "1-5")


if __name__ == "__main__":
    unittest.main()

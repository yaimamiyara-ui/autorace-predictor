import unittest
from autorace.common.odds import parse_odds_body, parse_refund_body


def _rt3():
    return {str(a): {str(b): {str(c): f"{a}{b}{c}.0" for c in range(1, 9) if c not in (a, b)}
                     for b in range(1, 9) if b != a} for a in range(1, 9)}


class TestOdds(unittest.TestCase):
    def test_real_shape_counts(self):
        body = {
            "tnsOddsList": {str(i): "2.5" for i in range(1, 9)},
            "fnsOddsList": {str(i): {"min": "1.2", "max": "1.8"} for i in range(1, 9)},
            "rtwOddsList": {str(a): {str(b): "10.0" for b in range(1, 9) if b != a} for a in range(1, 9)},
            "rt3OddsList": _rt3(),
            "playerList": [{"carNo": 1, "playerCode": "3206", "placeCode": 2, "handicap": 50,
                            "trialTime": "3.29", "stAve": "0.12", "goodTrackTraialAve": "3.31"}],
        }
        odds, players, unknown = parse_odds_body(6, "2026-11-11", 11, body)
        by = {}
        for o in odds:
            by[o.bet_type] = by.get(o.bet_type, 0) + 1
        self.assertEqual(by["tns"], 8)
        self.assertEqual(by["fns"], 8)
        self.assertEqual(by["rtw"], 56)
        self.assertEqual(by["rt3"], 336)
        fns = [o for o in odds if o.bet_type == "fns"][0]
        self.assertEqual((fns.odds, fns.odds_max), (1.2, 1.8))
        rt3 = {o.combo: o.odds for o in odds if o.bet_type == "rt3"}
        self.assertEqual(rt3["8-1-2"], 812.0)
        self.assertEqual(players[0].place_code, 6)  # 所属場(2)ではなく開催場
        self.assertEqual(unknown, {})

    def test_refund_unknown_kept_raw(self):
        body = [{"raceNo": 1, "refundInfo": {"rt3": {"weird": [1, 2, 3]}, "absent": []}}]
        entries, raw = parse_refund_body(6, "2026-11-11", body)
        self.assertEqual(entries, [])


if __name__ == "__main__":
    unittest.main()


class TestRefundRealShape(unittest.TestCase):
    def test_real_refund(self):
        body = [{"raceNo": 1, "refundInfo": {
            "rt3": {"typeCode": 0, "typeName": "通常", "list": [{"1thCarNo": 5, "2thCarNo": 3, "3thCarNo": 4, "refund": 2410, "pop": 11, "refundVotes": "3074"}]},
            "rf3": {"typeCode": 0, "typeName": "通常", "list": [{"1thCarNo": 3, "2thCarNo": 4, "3thCarNo": 5, "refund": 1190, "pop": 5, "refundVotes": "359"}]},
            "wid": {"typeCode": 0, "typeName": "通常", "list": [{"1thCarNo": 3, "2thCarNo": 5, "refund": 210, "pop": 4},
                                                              {"1thCarNo": 4, "2thCarNo": 5, "refund": 130, "pop": 2},
                                                              {"1thCarNo": 3, "2thCarNo": 4, "refund": 1060, "pop": 10}]},
            "tns": {"typeCode": 0, "typeName": "通常", "list": [{"carNo": 5, "refund": 100, "pop": 1, "refundVotes": "1524"}]},
            "fns": {"typeCode": 0, "typeName": "通常", "list": [{"carNo": 5, "order": 1, "refund": 100, "pop": 1},
                                                              {"carNo": 3, "order": 2, "refund": 280, "pop": 4}]},
            "absent": []}}]
        entries, raw = parse_refund_body(5, "2026-11-11", body)
        self.assertEqual(raw, [])
        got = {(e.bet_type, e.combo): (e.payout, e.popularity) for e in entries}
        self.assertEqual(got[("rt3", "5-3-4")], (2410, 11))
        self.assertEqual(got[("rf3", "3-4-5")], (1190, 5))
        self.assertEqual(got[("fns", "3")], (280, 4))
        self.assertEqual(got[("tns", "5")], (100, 1))
        self.assertEqual(len([e for e in entries if e.bet_type == "wid"]), 3)

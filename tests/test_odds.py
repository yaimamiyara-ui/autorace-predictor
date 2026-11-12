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

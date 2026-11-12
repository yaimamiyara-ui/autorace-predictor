import unittest
from autorace.common.schema import HoldInfo, ProgramEntry, ResultEntry


class TestAutoraceSchema(unittest.TestCase):
    def test_hold_info_parsing(self):
        sample_hold = {
            "placeCode": "2", "placeKey": "kawaguchi", "gradeName": "GI", "gradeCode": "1",
            "periodStartDate": "2026-11-10", "periodEndDate": "2026-11-14", "paragraphday": "3",
            "nighterCode": "0", "nighterName": "", "nowRaceNo": "5", "raceStartTime": "12:30",
            "temp": "18.5", "humid": "45", "roadtemp": "22.0", "weather": "晴",
            "raceName": "開設記念グランプリレース", "raceClass": "オープン", "distance": "3100",
            "telvoteTime": "12:25", "cancelFlg": "0",
        }
        model = HoldInfo.model_validate(sample_hold)
        self.assertEqual(model.place_code, 2)
        self.assertEqual(model.paragraph_day, 3)
        self.assertEqual(model.temp, 18.5)
        self.assertIsNone(model.nighter_name)
        self.assertFalse(model.cancel_flg)

    def test_program_and_result_join_keys(self):
        program_body = {"playerList": [{
            "carNo": "1", "playerCode": "2901", "playerName": "テスト選手", "retireFlag": "0",
            "placeCode": "5", "graduationCode": "29", "age": "38", "bikeClass": "1", "bikeName": "セア",
            "rank": "S-1", "handicap": "0", "trialRunTime": "3.28", "trialRetryCode": "", "absent": "0",
            "raceDev": "0.045", "rate2": "65.5", "rate3": "82.1",
        }]}
        result_body = {"raceResult": [
            {"order": "1", "accidentCode": "", "accidentName": "", "carNo": "1", "playerCode": "2901",
             "retireFlag": "0", "motorcycleName": "セア", "handicap": "0", "traialRetryCode": None,
             "traialTime": "3.28", "raceTime": "3.395", "st": "0.12", "foulCode": ""},
            {"order": "0", "accidentCode": "1", "accidentName": "落妨", "carNo": "2", "playerCode": "3099",
             "retireFlag": "0", "motorcycleName": "セア2", "handicap": "10", "traialRetryCode": "",
             "traialTime": "3.33", "raceTime": "", "st": "0.18", "foulCode": "1"},
        ]}
        programs = ProgramEntry.from_api_response(2, "2026-11-12", 12, program_body)
        results = ResultEntry.from_api_response(2, "2026-11-12", 12, result_body)
        self.assertEqual(len(programs), 1)
        self.assertEqual(len(results), 2)
        p, r1, r2 = programs[0], results[0], results[1]
        self.assertEqual((p.place_code, p.race_date, p.race_no, p.player_code),
                         (r1.place_code, r1.race_date, r1.race_no, r1.player_code))
        self.assertEqual(p.branch_place_code, 5)
        self.assertEqual(p.trial_run_time, 3.28)
        self.assertIsNone(p.trial_retry_code)
        self.assertEqual(r1.order, 1)
        self.assertEqual(r1.race_time, 3.395)
        self.assertIsNone(r2.order)
        self.assertIsNone(r2.race_time)
        self.assertEqual(r2.accident_code, "1")

    # レビューで追加：実際のAPI応答（2026-11-12 山陽11R・2015-11-12 山陽11R）の形そのまま
    def test_real_response_shapes(self):
        prog = {"twicePlayerExist": 0, "playerList": [
            {"carNo": 1, "playerCode": "9032", "playerName": "染谷　和香", "retireFlag": 0, "placeCode": 2,
             "placeKey": "kawaguchi", "placeName": "川　口", "graduationCode": "38", "age": 22, "bikeClass": 2,
             "bikeName": "チーヤン", "rank": "B-117", "handicap": 0, "anotherRaceNo": None, "anotherRaceNo2": None,
             "trialRunTime": "3.45", "trialRetryCode": None, "absent": None, "sunnyExpectCode": "",
             "rainExpectCode": "", "raceDev": "076", "rate2": "10.0", "rate3": "20.0"}]}
        res = {"twicePlayerExist": 0, "raceResult": [
            {"order": 1, "accidentCode": None, "accidentName": None, "carNo": 8, "playerCode": "2910",
             "retireFlag": 0, "playerName": "金子　大輔", "playerNameEn": "DAISUKE　KANEKO",
             "motorcycleName": "Ｒマキロイ１", "handicap": 20, "traialRetryCode": None, "traialTime": "3.31",
             "raceTime": "3.353", "st": "0.06", "foulCode": None, "anotherRaceNo": None, "anotherRaceNo2": None}]}
        p = ProgramEntry.from_api_response(6, "2026-11-12", 11, prog)[0]
        r = ResultEntry.from_api_response(6, "2015-11-12", 11, res)[0]
        self.assertEqual(p.place_code, 6)
        self.assertEqual(p.branch_place_code, 2)
        self.assertEqual(p.race_dev, 76.0)
        self.assertEqual(p.rank, "B-117")
        self.assertEqual(r.player_code, 2910)
        self.assertEqual(r.handicap, 20.0)
        self.assertEqual(r.st, 0.06)


if __name__ == "__main__":
    unittest.main()

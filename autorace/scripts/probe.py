"""実サイトからの取得確認。件数と項目名だけを表示する（中身は出さない）。"""
import json
from autorace.fetch.client import AutoraceClient, AutoraceAPIError
from autorace.common.schema import HoldInfo, ProgramEntry, ResultEntry


def shape(x, depth=0):
    if isinstance(x, dict):
        return {k: shape(v, depth + 1) for k, v in list(x.items())[:40]} if depth < 3 else "{...}"
    if isinstance(x, list):
        return [f"len={len(x)}", shape(x[0], depth + 1)] if x else ["len=0"]
    return type(x).__name__


c = AutoraceClient(min_interval=3)
today = c.hold_today()["body"]["today"]
print("HOLD_TODAY", len(today), [HoldInfo.model_validate(h).place_key for h in today])
for place, date, rno in [(6, "2026-11-11", 11), (6, "2015-11-12", 11), (2, "2010-01-10", 5), (1, "2015-06-07", 3)]:
    try:
        p = ProgramEntry.from_api_response(place, date, rno, c.program(place, date, rno)["body"])
    except AutoraceAPIError as e:
        p = f"Failure {e.errors}"
    try:
        r = ResultEntry.from_api_response(place, date, rno, c.race_result(place, date, rno)["body"])
    except AutoraceAPIError as e:
        r = f"Failure {e.errors}"
    print("RACE", place, date, rno, "program:", p if isinstance(p, str) else len(p),
          "result:", r if isinstance(r, str) else len(r),
          "trial_time_ok:", (not isinstance(r, str)) and all(x.trial_time is not None for x in r))
for place, date, rno in [(6, "2026-11-11", 11), (6, "2015-11-12", 11)]:
    try:
        ob = c.odds(place, date, rno).get("body") or {}
        print("ODDS_KEYS", date, list(ob.keys()))
        for k, v in ob.items():
            if k.endswith("OddsList") and isinstance(v, dict):
                n = sum(len(x) if isinstance(x, dict) else 1 for x in v.values())
                deep = sum(len(y) for x in v.values() if isinstance(x, dict) for y in x.values() if isinstance(y, dict))
                print("ODDS_COUNT", date, k, "level2=", n, "level3=", deep)
    except AutoraceAPIError as e:
        print("ODDS Failure", date, e.errors)
    try:
        print("REFUND_SHAPE", date, json.dumps(shape(c.race_refund(place, date).get("body")), ensure_ascii=False)[:1500])
    except AutoraceAPIError as e:
        print("REFUND Failure", date, e.errors)
# 学習モデル v1 用：レース単位の項目（距離・周回・走路状態・気温など）の有無を確認（項目名と型だけ）
for place, date, rno in [(6, "2026-11-11", 11), (3, "2024-08-10", 12), (5, "2025-06-15", 1), (2, "2024-02-03", 1)]:
    for kind, fn in (("PROGRAM", c.program), ("RESULT", c.race_result)):
        try:
            b = fn(place, date, rno).get("body") or {}
            print(kind + "_SHAPE", place, date, rno, json.dumps(shape(b), ensure_ascii=False)[:2500])
        except AutoraceAPIError as e:
            print(kind + " Failure", e.errors)

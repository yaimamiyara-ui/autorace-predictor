# autorace/scripts/quick_eval_v1.py（Claude作成）
# 集まっている実データだけで v1 をすぐ試す。日付で 前60%=学習 / 次15%=検証 / 最後25%=テスト に分け、
# テスト期間で「1着の当たり方」「3連単の当たり方」「確率の当てになり具合」「市場（オッズ）との比較」を出す。
# 出力は集計値だけ（個別の買い目・オッズは出さない）。
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from autorace.model.combos import generate_rt3_combos
from autorace.model.evaluate_v1 import evaluate_predictions
from autorace.model.features import RACE_KEYS
from autorace.model.features_v1 import build_features_v1
from autorace.model.v0 import BaselineModelV0
from autorace.model.v1 import ThreeStageModel
from autorace.scripts.backtest_v0 import actual_rt3, evaluate_ev_strategies, load_parquet_table
from autorace.scripts.train_v1 import attach_targets

WIN_BINS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0001]
WIN_LABELS = ["0-5%", "5-10%", "10-20%", "20-30%", "30-50%", "50%以上"]


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_date"] = df["race_date"].astype(str)
    df["place_code"] = pd.to_numeric(df["place_code"], errors="coerce").astype(int)
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").astype(int)
    return df


def winners(result: pd.DataFrame) -> pd.DataFrame:
    r = result[pd.to_numeric(result["order"], errors="coerce") == 1]
    cnt = r.groupby(RACE_KEYS)["car_no"].transform("count")
    return r[cnt == 1][RACE_KEYS + ["car_no"]].rename(columns={"car_no": "win_car"})


def win_summary(pred: pd.DataFrame, prob_col: str, result: pd.DataFrame) -> dict:
    """本命（確率1位）の勝率、勝った車に付けていた確率の平均、LogLoss、確率帯ごとの実際の勝率"""
    p = pred[pred[prob_col] > 0][RACE_KEYS + ["car_no", prob_col]].merge(winners(result), on=RACE_KEYS)
    if p.empty:
        return {}
    p["is_win"] = (p["car_no"] == p["win_car"]).astype(float)
    top = p.sort_values(RACE_KEYS + [prob_col], ascending=[True, True, True, False]).drop_duplicates(RACE_KEYS)
    wp = p[p["is_win"] == 1][prob_col].clip(1e-15, 1)
    p["bin"] = pd.cut(p[prob_col], WIN_BINS, labels=WIN_LABELS, right=False)
    cal = p.groupby("bin", observed=False).agg(n=("is_win", "size"), pred=(prob_col, "mean"), actual=("is_win", "mean"))
    return {
        "races": int(top.shape[0]),
        "favorite_win_rate": round(float(top["is_win"].mean()), 4),
        "favorite_avg_prob": round(float(top[prob_col].mean()), 4),
        "winner_avg_prob": round(float(wp.mean()), 4),
        "win_logloss": round(float(-np.log(wp).mean()), 4),
        "calibration": [{"band": str(b), "n": int(r.n), "pred": round(float(r.pred), 4) if r.n else None,
                         "actual": round(float(r.actual), 4) if r.n else None} for b, r in cal.iterrows()],
    }


def rt3_topn(rt3: pd.DataFrame, result: pd.DataFrame, ns=(1, 5, 10, 20)) -> dict:
    act = actual_rt3(result).rename(columns={"combo": "act"})
    r = rt3.sort_values(RACE_KEYS + ["combo_prob"], ascending=[True, True, True, False]).copy()
    r["rank"] = r.groupby(RACE_KEYS).cumcount() + 1
    m = r.merge(act, on=RACE_KEYS)
    hit_rank = m[m["combo"] == m["act"]].set_index(RACE_KEYS)["rank"]
    n_races = act.merge(r[RACE_KEYS].drop_duplicates(), on=RACE_KEYS).shape[0]
    out = {"races": int(n_races)}
    for n in ns:
        out[f"top{n}_hit_rate"] = round(float((hit_rank <= n).sum() / n_races), 4) if n_races else None
    out["median_rank_of_actual"] = float(hit_rank.median()) if len(hit_rank) else None
    return out


def market_win(odds: pd.DataFrame) -> pd.DataFrame:
    """単勝の確定オッズから市場の1着確率（1/オッズを合計1に正規化）"""
    t = odds[odds["bet_type"] == "tns"][RACE_KEYS + ["combo", "odds"]].copy()
    t["odds"] = pd.to_numeric(t["odds"], errors="coerce")
    t = t[t["odds"] > 0]
    t["car_no"] = pd.to_numeric(t["combo"].astype(str).str.split("-").str[0], errors="coerce")
    t["inv"] = 1.0 / t["odds"]
    t["mkt_prob"] = t["inv"] / t.groupby(RACE_KEYS)["inv"].transform("sum")
    return t[RACE_KEYS + ["car_no", "mkt_prob"]]


def market_rt3(odds: pd.DataFrame) -> pd.DataFrame:
    t = odds[odds["bet_type"] == "rt3"][RACE_KEYS + ["combo", "odds"]].copy()
    t["odds"] = pd.to_numeric(t["odds"], errors="coerce")
    t = t[t["odds"] > 0]
    t["inv"] = 1.0 / t["odds"]
    t["combo_prob"] = t["inv"] / t.groupby(RACE_KEYS)["inv"].transform("sum")
    t["bet_type"] = "rt3"
    return t[RACE_KEYS + ["bet_type", "combo", "combo_prob"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--end-date", default="2099-12-31")
    ap.add_argument("--out", default="out/quick_eval_v1.json")
    a = ap.parse_args()
    d = Path(a.data_dir)
    program = _norm(load_parquet_table(d, "program", a.end_date))
    result = _norm(load_parquet_table(d, "race_result", a.end_date))
    info = load_parquet_table(d, "race_info", a.end_date)
    odds = load_parquet_table(d, "odds", a.end_date)
    refund = load_parquet_table(d, "refund", a.end_date)
    odds = _norm(odds) if not odds.empty else odds
    refund = _norm(refund) if not refund.empty else refund

    dates = sorted(set(program["race_date"]) & set(result["race_date"]))
    n = len(dates)
    tr_end, va_end = dates[int(n * 0.60) - 1], dates[int(n * 0.75) - 1]
    print(f"[Data] days={n} {dates[0]}..{dates[-1]} program_rows={len(program)} race_info_rows={len(info)} "
          f"odds_rows={len(odds)} | train..{tr_end} valid..{va_end} test..{dates[-1]}")

    full = attach_targets(build_features_v1(program, info if not info.empty else None, result), result)
    tr = full[full["race_date"] <= tr_end]
    va = full[(full["race_date"] > tr_end) & (full["race_date"] <= va_end)]
    te = full[full["race_date"] > va_end]
    te_res = result[result["race_date"] > va_end]

    model = ThreeStageModel().fit(tr, va)
    v1_win = model.predict_win(te)
    v1_rt3 = model.predict_rt3(te)
    v1_harv = generate_rt3_combos(v1_win)
    v0_all = BaselineModelV0().predict_historical(program, result)
    v0_all["race_date"] = v0_all["race_date"].astype(str)
    v0_win = v0_all[v0_all["race_date"] > va_end]
    v0_rt3 = generate_rt3_combos(v0_win)

    rep = {"period": {"train_end": tr_end, "valid_end": va_end, "test_end": dates[-1]},
           "temperatures": model.temperatures,
           "win": {"v1": win_summary(v1_win, "pred_prob", te_res), "v0": win_summary(v0_win, "pred_prob", te_res)},
           "rt3_rank": {"v1_3stage": rt3_topn(v1_rt3, te_res), "v1_harville": rt3_topn(v1_harv, te_res),
                        "v0_harville": rt3_topn(v0_rt3, te_res)},
           "rt3_metrics": {"v1_3stage": evaluate_predictions(v1_win, v1_rt3, te_res),
                           "v1_harville": evaluate_predictions(v1_win, v1_harv, te_res),
                           "v0_harville": evaluate_predictions(v0_win, v0_rt3, te_res)}}
    te_odds = odds[odds["race_date"] > va_end] if not odds.empty else odds
    if not te_odds.empty:
        mw = market_win(te_odds)
        rep["win"]["market_final_odds"] = win_summary(mw.rename(columns={"mkt_prob": "pred_prob"}), "pred_prob", te_res)
        mr = market_rt3(te_odds)
        rep["rt3_rank"]["market_final_odds"] = rt3_topn(mr, te_res)
        rep["rt3_metrics"]["market_final_odds"] = evaluate_predictions(pd.DataFrame(columns=RACE_KEYS + ["car_no", "pred_prob"]), mr, te_res)
        te_ref = refund[refund["race_date"] > va_end] if not refund.empty else refund
        roi, _ = evaluate_ev_strategies(v1_rt3, te_odds, te_ref, te_res, [1.0, 1.2, 1.5], max_picks=5)
        rep["ev_bets_final_odds_v1"] = roi  # 確定オッズで見た参考値（実際より良く見える）
    for m in rep["rt3_metrics"].values():
        m.pop("ece_bins", None)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

# autorace/scripts/eval_blend.py （Gemini作成の backtest_ev_grid を使い、全体の流れは Claude が quick_eval_v1 に合わせて作成）
# 日付で 学習60%／合成の当てはめ15%／テスト25% に分け、v1単独・市場・合成(a)・合成(b) を比べる。
# 期待値買いは 確定オッズ版（参考：良く見えすぎる）と 試走後オッズ版（odds_pre がある期間）。出力は集計値だけ。
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autorace.model.blend import BenterBlend
from autorace.model.evaluate_v1 import compute_win_logloss, evaluate_predictions  # noqa: F401（テストから参照）
from autorace.model.features import RACE_KEYS
from autorace.model.features_v1 import build_features_v1
from autorace.model.market import fit_market_gamma, market_rt3_probs, market_win_probs, pick_snapshot
from autorace.model.v1 import ThreeStageModel
from autorace.scripts.backtest_v0 import load_parquet_table
from autorace.scripts.quick_eval_v1 import _norm, rt3_topn, win_summary
from autorace.scripts.train_v1 import attach_targets

K = list(RACE_KEYS)


def _max_zero_streak(hit_flags: np.ndarray) -> int:
    max_s = cur = 0
    for h in hit_flags:
        cur = 0 if h > 0 else cur + 1
        max_s = max(max_s, cur)
    return int(max_s)


def _empty_row(th: float, mo: float, mk: int) -> dict[str, Any]:
    return {"ev_threshold": th, "max_odds": mo, "max_k": mk, "n_races_bet": 0, "n_bets": 0, "n_hits": 0,
            "hit_rate": 0.0, "roi": 0.0, "roi_pct": 0.0, "max_losing_streak": 0, "max_losing_streak_bets": 0,
            "monthly_roi": {}}


def backtest_ev_grid(rt3_pred_df: pd.DataFrame, odds_rt3_df: pd.DataFrame, refund_df: pd.DataFrame,
                     thresholds=(1.0, 1.1, 1.2, 1.3, 1.5), min_prob: float = 0.005,
                     max_odds_list=(100.0, 200.0, 500.0), max_k_list=(1, 3, 5)) -> list[dict[str, Any]]:
    """（Gemini作成）3連単の期待値買い。確率×オッズ≥閾値、確率≥min_prob、オッズ≤上限、1レース期待値の高い順に最大k点。
    払戻は refund の payout（100円あたり）で判定。集計だけ返す。"""
    results: list[dict[str, Any]] = []
    if rt3_pred_df is None or rt3_pred_df.empty or odds_rt3_df is None or odds_rt3_df.empty:
        return results
    o = odds_rt3_df.copy()
    if "bet_type" in o.columns:
        o = o.loc[o["bet_type"] == "rt3"].copy()
    o["combo"] = o["combo"].astype(str).str.strip()
    o["odds"] = pd.to_numeric(o["odds"], errors="coerce")
    o = o.loc[o["odds"].notna() & (o["odds"] > 0)].drop_duplicates(subset=K + ["combo"], keep="last")
    p = rt3_pred_df[K + ["combo", "combo_prob"]].copy()
    p["combo"] = p["combo"].astype(str).str.strip()
    cand = p.loc[p["combo_prob"] >= float(min_prob)].merge(o[K + ["combo", "odds"]], on=K + ["combo"], how="inner")
    if cand.empty:
        return [_empty_row(float(th), float(mo), int(mk)) for th in thresholds for mo in max_odds_list for mk in max_k_list]
    cand["ev"] = cand["combo_prob"] * cand["odds"]
    cand = cand.loc[(cand["odds"] <= float(max(max_odds_list))) & (cand["ev"] >= float(min(thresholds)))].copy()

    ref = pd.DataFrame(columns=K + ["combo", "payout"])
    if refund_df is not None and not refund_df.empty:
        r = refund_df.copy()
        if "bet_type" in r.columns:
            r = r.loc[r["bet_type"] == "rt3"].copy()
        if not r.empty:
            r["combo"] = r["combo"].astype(str).str.strip()
            r["payout"] = pd.to_numeric(r["payout"], errors="coerce").fillna(0.0)
            ref = r.groupby(K + ["combo"], as_index=False)["payout"].sum()
    cand = cand.merge(ref, on=K + ["combo"], how="left")
    cand["payout"] = pd.to_numeric(cand["payout"], errors="coerce").fillna(0.0)
    cand["month"] = cand["race_date"].astype(str).str[:7]
    cand = cand.sort_values(["race_date", "place_code", "race_no", "ev", "combo_prob"],
                            ascending=[True, True, True, False, False]).reset_index(drop=True)

    for th in thresholds:
        for mo in max_odds_list:
            sub = cand.loc[(cand["ev"] >= float(th)) & (cand["odds"] <= float(mo))]
            ranks = sub.groupby(["race_date", "place_code", "race_no"], sort=False).cumcount() + 1
            for mk in max_k_list:
                picks = sub.loc[ranks <= int(mk)]
                n_bets = int(len(picks))
                if n_bets == 0:
                    results.append(_empty_row(float(th), float(mo), int(mk)))
                    continue
                pay = picks["payout"].to_numpy(dtype=np.float64)
                hit = (pay > 0).astype(np.int64)
                race_hits = picks.assign(_h=hit).groupby(["race_date", "place_code", "race_no"], sort=False)["_h"].max().to_numpy()
                roi = float(pay.sum() / (n_bets * 100.0))
                monthly = {}
                for m, g in picks.groupby("month", sort=True):
                    mp = g["payout"].to_numpy(dtype=np.float64)
                    monthly[str(m)] = {"n_bets": int(len(g)), "n_hits": int((mp > 0).sum()),
                                       "roi_pct": round(float(mp.sum() / (len(g) * 100.0)) * 100, 1)}
                results.append({"ev_threshold": float(th), "max_odds": float(mo), "max_k": int(mk),
                                "n_races_bet": int(len(race_hits)), "n_bets": n_bets, "n_hits": int(hit.sum()),
                                "hit_rate": round(hit.sum() / n_bets, 6), "roi": round(roi, 6), "roi_pct": round(roi * 100, 2),
                                "max_losing_streak": _max_zero_streak(race_hits), "max_losing_streak_bets": _max_zero_streak(hit),
                                "monthly_roi": monthly})
    return results


def _variants(model: ThreeStageModel, blend: BenterBlend, df: pd.DataFrame, odds: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """1つの期間・1つのオッズ（確定 or 試走後）について、4つの予想（1着・3連単）を作る"""
    act = df[~df["is_absent"].fillna(False).astype(bool)]
    v1_win = model.predict_win(act)
    v1_rt3 = model.predict_rt3(act)
    mw = market_win_probs(odds)
    mr = market_rt3_probs(odds)
    bw = blend.predict_win(v1_win, mw)
    return {
        "v1": {"win": v1_win, "rt3": v1_rt3},
        "market": {"win": mw.rename(columns={"market_prob": "pred_prob"}), "rt3": mr.rename(columns={"market_prob": "combo_prob"})},
        "blend_a": {"win": bw, "rt3": blend.predict_rt3_direct(v1_rt3, mr)},
        "blend_b": {"win": bw, "rt3": blend.predict_rt3_conditional(model, act, mw, v1_win)},
    }


def _summ(v: dict[str, pd.DataFrame], res: pd.DataFrame) -> dict[str, Any]:
    w = win_summary(v["win"], "pred_prob", res)
    m = evaluate_predictions(v["win"], v["rt3"], res)
    return {"win": w, "rt3_rank": rt3_topn(v["rt3"], res),
            "rt3_logloss": round(float(m["rt3_logloss"]), 4), "rt3_ece": round(float(m["rt3_ece"]), 5)}


def _best(grid: list[dict[str, Any]], min_bets: int = 300, n: int = 5) -> list[dict[str, Any]]:
    ok = [g for g in grid if g["n_bets"] >= min_bets]
    return sorted(ok, key=lambda g: -g["roi"])[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--end-date", default="2099-12-31")
    ap.add_argument("--out", default="out/eval_blend.json")
    ap.add_argument("--pre-how", default="min_before_close")
    ap.add_argument("--pre-minutes", type=float, default=5.0)
    a = ap.parse_args()
    d = Path(a.data_dir)
    program = _norm(load_parquet_table(d, "program", a.end_date))
    result = _norm(load_parquet_table(d, "race_result", a.end_date))
    info = load_parquet_table(d, "race_info", a.end_date)
    odds = _norm(load_parquet_table(d, "odds", a.end_date))
    refund = _norm(load_parquet_table(d, "refund", a.end_date))
    pre = load_parquet_table(d, "odds_pre", a.end_date)
    pre = _norm(pre) if not pre.empty else pre
    odds = odds[odds["bet_type"].isin(["tns", "rt3"])]

    dates = sorted(set(program["race_date"]) & set(result["race_date"]))
    n = len(dates)
    tr_end, bl_end = dates[int(n * 0.60) - 1], dates[int(n * 0.75) - 1]
    print(f"[Data] days={n} {dates[0]}..{dates[-1]} odds_rows={len(odds)} odds_pre_rows={len(pre)} | "
          f"train..{tr_end} blend..{bl_end} test..{dates[-1]}")

    full = attach_targets(build_features_v1(program, info if not info.empty else None, result), result)
    tr = full[full["race_date"] <= tr_end]
    bl = full[(full["race_date"] > tr_end) & (full["race_date"] <= bl_end)]
    te = full[full["race_date"] > bl_end]
    bl_res = result[(result["race_date"] > tr_end) & (result["race_date"] <= bl_end)]
    te_res = result[result["race_date"] > bl_end]
    bl_odds = odds[(odds["race_date"] > tr_end) & (odds["race_date"] <= bl_end)]
    te_odds = odds[odds["race_date"] > bl_end]
    te_ref = refund[refund["race_date"] > bl_end]

    model = ThreeStageModel().fit(tr, bl)  # 温度は当てはめ期間で決める
    bl_act = bl[~bl["is_absent"].fillna(False).astype(bool)]
    blend = BenterBlend()
    blend.fit(model.predict_win(bl_act), market_win_probs(bl_odds), bl_res)
    blend.fit_rt3(model.predict_rt3(bl_act), market_rt3_probs(bl_odds), bl_res)
    gamma = fit_market_gamma(bl_odds, bl_res)
    print(f"[Blend] alpha={blend.alpha:.3f} beta={blend.beta:.3f} alpha3={blend.alpha3:.3f} beta3={blend.beta3:.3f} "
          f"market_gamma={gamma:.3f}")

    rep: dict[str, Any] = {"period": {"train_end": tr_end, "blend_end": bl_end, "test_end": dates[-1]},
                           "temperatures": model.temperatures, "blend": blend.to_dict(), "market_gamma": gamma}
    var = _variants(model, blend, te, te_odds)
    rep["test_final_odds"] = {k: _summ(v, te_res) for k, v in var.items()}
    rep["ev_final_odds"] = {k: _best(backtest_ev_grid(var[k]["rt3"], te_odds, te_ref)) for k in ("v1", "blend_a", "blend_b")}

    rep["ev_pre_odds"] = {"available": False}
    if not pre.empty:
        snap = pick_snapshot(pre[pre["bet_type"].isin(["tns", "rt3"])], how=a.pre_how, minutes=a.pre_minutes)
        races = snap[K].drop_duplicates()
        te_p = te.merge(races, on=K)
        if not te_p.empty:
            ref_p = te_ref.merge(races, on=K)
            res_p = te_res.merge(races, on=K)
            vp = _variants(model, blend, te_p, snap)
            rep["ev_pre_odds"] = {
                "available": True, "races": int(len(races)),
                "metrics": {k: _summ(v, res_p) for k, v in vp.items()},
                "grid": {k: backtest_ev_grid(vp[k]["rt3"], snap, ref_p, max_k_list=(1, 3, 5)) for k in ("v1", "blend_a", "blend_b")},
            }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    # 標準出力は要点だけ
    for k, s in rep["test_final_odds"].items():
        print(f"[Test {k}] fav_win={s['win'].get('favorite_win_rate')} win_ll={s['win'].get('win_logloss')} "
              f"rt3_ll={s['rt3_logloss']} top1/5/10/20={[s['rt3_rank'].get(f'top{x}_hit_rate') for x in (1, 5, 10, 20)]}")
    for k, rows in rep["ev_final_odds"].items():
        for g in rows[:3]:
            print(f"[EV final {k}] th={g['ev_threshold']} max_odds={g['max_odds']} k={g['max_k']} bets={g['n_bets']} "
                  f"hits={g['n_hits']} roi={g['roi_pct']}% lose_streak={g['max_losing_streak']}")
    if rep["ev_pre_odds"].get("available"):
        print(f"[Pre] races={rep['ev_pre_odds']['races']}")
        for k, rows in rep["ev_pre_odds"]["grid"].items():
            tot = [g for g in rows if g["n_bets"] > 0]
            best = sorted(tot, key=lambda g: -g["roi"])[:2]
            for g in best:
                print(f"[EV pre {k}] th={g['ev_threshold']} max_odds={g['max_odds']} k={g['max_k']} bets={g['n_bets']} "
                      f"hits={g['n_hits']} roi={g['roi_pct']}%")


if __name__ == "__main__":
    main()

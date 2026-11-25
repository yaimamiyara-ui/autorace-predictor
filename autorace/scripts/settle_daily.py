"""毎朝の答え合わせ（Gemini作成、Claude確認・修正）。

前日に記録した直前データ（program_pre / odds_pre）で予想と推奨（記録のみ、購入はしない）を作り、
結果・払戻・確定オッズと照らし合わせて成績表とダッシュボードを更新する。

予想は data/models/live に本番用の v1 と合成の重み（fit_live_model.py が作る）があれば
「v1 と市場（直前の単勝オッズ）の合成(b)」を使い、推奨は 期待値1.3以上・オッズ100倍以下・1レース1点。
無い（または予想を作れない）ときは従来の v0 と 期待値1.2以上・500倍以下・1レース5点 を使う。
どちらを使ったかは成績表の model 列に残す。

修正点（Claudeのレビュー）：
- 場コードと場名の対応を修正（1=船橋, 2=川口, 3=伊勢崎, 4=浜松, 5=飯塚, 6=山陽）
- 結果の着順の列名 order を使う（元は見つけられず的中判定ができなかった）
- 精度指標の名前を evaluate.py に合わせる（元は LogLoss が空になっていた）
- ダッシュボードを簡潔な形に作り直し
"""
from __future__ import annotations

import argparse
import html
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from autorace.model.blend import BenterBlend
from autorace.model.combos import generate_rt3_combos
from autorace.model.evaluate import evaluate_predictions
from autorace.model.features_v1 import build_features_v1
from autorace.model.market import market_win_probs, pick_snapshot
from autorace.model.v0 import BaselineModelV0
from autorace.model.v1 import ThreeStageModel
from autorace.scripts.backtest_v0 import actual_rt3, load_parquet_table

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
RACE_KEYS = ["place_code", "race_date", "race_no"]
MIN_EV, MIN_PROB, MAX_ODDS, MAX_RECS_PER_RACE = 1.2, 0.01, 500.0, 5
# 本番用 v1＋合成(b) の推奨条件（eval_blend の試算で回収率が最も良かった組み合わせ）
V1_MIN_EV, V1_MIN_PROB, V1_MAX_ODDS, V1_MAX_RECS_PER_RACE = 1.3, 0.005, 100.0, 1
FEATURE_LOOKBACK_DAYS = 400
PLACE_NAMES = {1: "船橋", 2: "川口", 3: "伊勢崎", 4: "浜松", 5: "飯塚", 6: "山陽"}


def default_target_date() -> str:
    return (datetime.now(JST) - timedelta(days=1)).strftime("%Y-%m-%d")


def normalize_race_keys(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    if "race_date" in out.columns:
        out["race_date"] = pd.to_datetime(out["race_date"]).dt.strftime("%Y-%m-%d")
    for c in ("place_code", "race_no"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def select_earliest_snapshot(df: pd.DataFrame, dedup_keys: list[str]) -> pd.DataFrame:
    """同じレースが複数回記録されていたら、最も早い時刻のものを使う"""
    if df.empty:
        return df.copy()
    out = df.copy()
    if "fetched_at" in out.columns:
        ts = pd.to_datetime(out["fetched_at"], errors="coerce", utc=True)
        out = out[ts.isna() | (ts == ts.groupby([out[k] for k in RACE_KEYS]).transform("min"))]
    keys = [k for k in dedup_keys if k in out.columns]
    return out.drop_duplicates(subset=keys, keep="first").reset_index(drop=True) if keys else out.reset_index(drop=True)


def select_recommendations(combos_df: pd.DataFrame, odds_pre_df: pd.DataFrame, min_ev: float = MIN_EV,
                           min_prob: float = MIN_PROB, max_odds: float = MAX_ODDS,
                           max_per_race: int = MAX_RECS_PER_RACE) -> pd.DataFrame:
    """期待値（確率×直前オッズ）が閾値以上の3連単を、各レース最大 max_per_race 点推奨にする（記録のみ）"""
    cols = RACE_KEYS + ["bet_type", "combo", "combo_prob", "pre_odds", "ev"]
    if combos_df.empty or odds_pre_df.empty:
        return pd.DataFrame(columns=cols)
    combos = normalize_race_keys(combos_df)
    combos = combos[combos["bet_type"] == "rt3"]
    odds = normalize_race_keys(odds_pre_df)
    odds = odds[(odds["snapshot"] == "pre") & (odds["bet_type"] == "rt3")]
    odds = select_earliest_snapshot(odds, RACE_KEYS + ["combo"])[RACE_KEYS + ["combo", "odds"]].rename(columns={"odds": "pre_odds"})
    m = combos.merge(odds, on=RACE_KEYS + ["combo"], how="inner")
    m["pre_odds"] = pd.to_numeric(m["pre_odds"], errors="coerce")
    m["ev"] = m["combo_prob"] * m["pre_odds"]
    m = m[(m["combo_prob"] >= min_prob) & (m["pre_odds"] > 0) & (m["pre_odds"] <= max_odds) & (m["ev"] >= min_ev)]
    if m.empty:
        return pd.DataFrame(columns=cols)
    m = m.sort_values(RACE_KEYS + ["ev", "combo_prob", "combo"], ascending=[True, True, True, False, False, True], kind="mergesort")
    return m.groupby(RACE_KEYS, sort=False).head(int(max_per_race))[cols].reset_index(drop=True)


def settle_recommendations(recs_df, result_df, refund_df, final_odds_df) -> pd.DataFrame:
    """推奨の的中（結果の着順で判定）と戻り（払戻/100、無ければ確定オッズ、外れは0）"""
    out_cols = RACE_KEYS + ["bet_type", "combo", "combo_prob", "pre_odds", "final_odds", "odds_change_rate", "ev", "is_hit", "return_amount"]
    if recs_df.empty:
        return pd.DataFrame(columns=out_cols)
    recs = normalize_race_keys(recs_df)
    act = (normalize_race_keys(actual_rt3(result_df))[RACE_KEYS + ["combo"]].rename(columns={"combo": "actual_combo"})
           if result_df is not None and not result_df.empty else pd.DataFrame(columns=RACE_KEYS + ["actual_combo"]))
    if refund_df is not None and not refund_df.empty:
        ref = normalize_race_keys(refund_df)
        ref = ref[ref["bet_type"] == "rt3"][RACE_KEYS + ["combo", "payout"]].drop_duplicates(subset=RACE_KEYS + ["combo"])
    else:
        ref = pd.DataFrame(columns=RACE_KEYS + ["combo", "payout"])
    if final_odds_df is not None and not final_odds_df.empty:
        fin = normalize_race_keys(final_odds_df)
        fin = fin[(fin["snapshot"] == "final") & (fin["bet_type"] == "rt3")]
        fin = fin.assign(final_odds=pd.to_numeric(fin["odds"], errors="coerce"))[RACE_KEYS + ["combo", "final_odds"]].drop_duplicates(subset=RACE_KEYS + ["combo"], keep="last")
    else:
        fin = pd.DataFrame(columns=RACE_KEYS + ["combo", "final_odds"])

    # 同着で当たり目が複数あるレースでも推奨の行が増えないよう、当たり目の集合で判定する
    act_hit = act.rename(columns={"actual_combo": "combo"}).drop_duplicates()
    act_hit["actual_combo"] = act_hit["combo"]
    s = recs.merge(act_hit, on=RACE_KEYS + ["combo"], how="left").merge(ref, on=RACE_KEYS + ["combo"], how="left").merge(fin, on=RACE_KEYS + ["combo"], how="left")
    s["final_odds"] = pd.to_numeric(s["final_odds"], errors="coerce")
    s["payout"] = pd.to_numeric(s["payout"], errors="coerce")
    s["odds_change_rate"] = np.where((s["pre_odds"] > 0) & s["final_odds"].notna(), (s["final_odds"] - s["pre_odds"]) / s["pre_odds"], np.nan)
    s["is_hit"] = (s["actual_combo"].notna() & (s["combo"] == s["actual_combo"])).astype(bool)
    ratio = (s["payout"] / 100.0).combine_first(s["final_odds"]).fillna(0.0)
    s["return_amount"] = np.where(s["is_hit"], ratio, 0.0).astype(float)
    return s[out_cols].reset_index(drop=True)


def build_predictions_table(pred_df: pd.DataFrame, result_df: pd.DataFrame) -> pd.DataFrame:
    pred = normalize_race_keys(pred_df)
    cols = RACE_KEYS + ["car_no", "player_code", "is_absent", "pred_prob"]
    out = pred[[c for c in cols if c in pred.columns]].copy()
    if result_df is not None and not result_df.empty:
        res = normalize_race_keys(result_df)[RACE_KEYS + ["car_no", "order"]].drop_duplicates(subset=RACE_KEYS + ["car_no"])
        out = out.merge(res.rename(columns={"order": "finish_order"}), on=RACE_KEYS + ["car_no"], how="left")
    else:
        out["finish_order"] = np.nan
    out["finish_order"] = pd.to_numeric(out["finish_order"], errors="coerce")
    out["is_win"] = np.where(out["finish_order"].notna(), (out["finish_order"] == 1).astype(float), np.nan)
    return out.sort_values(RACE_KEYS + ["car_no"]).reset_index(drop=True)


def upsert_monthly(base_dir: Path, target_date: str, new_df: pd.DataFrame, sort_keys: list[str]) -> None:
    """同じ日を再実行したら、その日の分を上書きする"""
    path = base_dir / target_date[:4] / f"{target_date[:7]}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = normalize_race_keys(pd.read_parquet(path))
        old = old[old["race_date"] != target_date]
        new_df = pd.concat([old, new_df], ignore_index=True) if not old.empty else new_df
    keys = [k for k in sort_keys if k in new_df.columns]
    if keys and not new_df.empty:
        new_df = new_df.sort_values(keys, kind="mergesort")
    new_df.reset_index(drop=True).to_parquet(path, index=False)


def upsert_daily_summary(tracker_dir: Path, row: dict) -> pd.DataFrame:
    path = tracker_dir / "daily_summary.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_parquet(path)
        old = old[old["race_date"] != row["race_date"]]
        df = pd.concat([old, df], ignore_index=True)
    df = df.sort_values("race_date").reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df


def _load_all(subdir: Path) -> pd.DataFrame:
    files = sorted(subdir.glob("*/*.parquet")) if subdir.exists() else []
    dfs = [d for d in (pd.read_parquet(f) for f in files) if not d.empty]
    return normalize_race_keys(pd.concat(dfs, ignore_index=True)) if dfs else pd.DataFrame()


def _calibration_svg(pred: pd.DataFrame) -> str:
    """20区間：予想した1着確率（横）と実際の1着率（縦）。点が斜めの線に近いほど確率が正確"""
    w, h, pl, pb, pt, pr = 520, 320, 50, 40, 15, 15
    pw, ph = w - pl - pr, h - pt - pb
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg">',
             f'<rect x="{pl}" y="{pt}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#ddd"/>',
             f'<line x1="{pl}" y1="{pt+ph}" x2="{pl+pw}" y2="{pt}" stroke="#bbb" stroke-dasharray="4 4"/>']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{pl+t*pw:.0f}" y="{h-18}" font-size="11" text-anchor="middle">{int(t*100)}%</text>')
        parts.append(f'<text x="{pl-6}" y="{pt+(1-t)*ph+4:.0f}" font-size="11" text-anchor="end">{int(t*100)}%</text>')
    parts.append(f'<text x="{pl+pw/2}" y="{h-2}" font-size="11" text-anchor="middle">予想した1着確率</text>')
    v = pred.dropna(subset=["pred_prob", "is_win"]) if not pred.empty else pred
    if not v.empty:
        v = v[~v.get("is_absent", pd.Series(False, index=v.index)).fillna(False).astype(bool)]
        b = np.clip((v["pred_prob"].astype(float) * 20).astype(int), 0, 19)
        g = v.assign(_b=b).groupby("_b").agg(p=("pred_prob", "mean"), y=("is_win", "mean"), n=("is_win", "size"))
        for _, r in g.iterrows():
            rad = 3 + min(8, np.sqrt(r["n"]) / 3)
            parts.append(f'<circle cx="{pl+r["p"]*pw:.1f}" cy="{pt+(1-r["y"])*ph:.1f}" r="{rad:.1f}" fill="#2563eb" fill-opacity="0.7"><title>{r["p"]:.1%}→{r["y"]:.1%}（{int(r["n"])}件）</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


# 成績表の「予想」列に出す名前（記録上の名前 → 画面の表記）
MODEL_LABELS = {"v0": "旧（基本）", "v1_blend": "新（合成）"}


def generate_dashboard_html(tracker_dir: Path) -> Path:
    summary = pd.read_parquet(tracker_dir / "daily_summary.parquet") if (tracker_dir / "daily_summary.parquet").exists() else pd.DataFrame()
    recs = _load_all(tracker_dir / "recommendations")
    preds = _load_all(tracker_dir / "predictions")
    n_rec = len(recs)
    hits = int(recs["is_hit"].sum()) if n_rec else 0
    roi = float(recs["return_amount"].sum() / n_rec) if n_rec else 0.0

    def tr(cells):
        return "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>"

    rows = []
    for _, r in summary.sort_values("race_date", ascending=False).head(30).iterrows():
        ll = "" if pd.isna(r["log_loss"]) else f"{r['log_loss']:.3f}"
        mdl = r.get("model") if isinstance(r.get("model"), str) and r.get("model") else "v0"
        rows.append(tr([r["race_date"], MODEL_LABELS.get(mdl, mdl), int(r["num_races"]), ll, int(r["num_recs"]), int(r["num_hits"]), f"{r['roi']:.0%}" if r["num_recs"] else "-"]))
    place_rows = []
    if n_rec:
        for pc, g in recs.groupby("place_code"):
            place_rows.append(tr([PLACE_NAMES.get(int(pc), pc), len(g), int(g["is_hit"].sum()), f"{g['return_amount'].sum()/len(g):.0%}"]))
    chg = recs["odds_change_rate"].dropna() if n_rec else pd.Series(dtype=float)
    page = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>オートレース予想 答え合わせ</title><style>body{{font-family:sans-serif;max-width:760px;margin:auto;padding:12px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:8px 0 20px}}td,th{{border-bottom:1px solid #eee;padding:4px 6px;text-align:right;font-size:14px}}
td:first-child,th:first-child{{text-align:left}}.k{{display:inline-block;margin:0 18px 10px 0}}.k b{{font-size:22px;display:block}}</style></head><body>
<h1>オートレース予想 答え合わせ</h1><p>推奨は記録のみ（購入していません）。1点100円で買ったと仮定した成績です。</p>
<div class="k">累計の推奨<b>{n_rec}点</b></div><div class="k">的中<b>{hits}点</b></div><div class="k">回収率<b>{roi:.0%}</b></div>
<div class="k">推奨時→確定のオッズ変化<b>{(chg.median() if len(chg) else 0):+.0%}</b></div>
<h2>日別</h2><table><tr><th>日付</th><th>予想</th><th>レース</th><th>外れ具合</th><th>推奨</th><th>的中</th><th>回収率</th></tr>{''.join(rows)}</table>
<h2>場別</h2><table><tr><th>場</th><th>推奨</th><th>的中</th><th>回収率</th></tr>{''.join(place_rows)}</table>
<h2>確率の正確さ（20区間）</h2>{_calibration_svg(preds)}
<p style="color:#666;font-size:13px">予想：新（合成）＝予想と市場のオッズを合わせたもの、旧（基本）＝前からの簡単な予想。外れ具合は小さいほど確率が正確です。</p>
<p style="color:#888;font-size:12px">更新：{datetime.now(JST):%Y-%m-%d %H:%M}</p></body></html>"""
    out = tracker_dir / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    return out


def load_live_model(data_dir: Path) -> tuple[ThreeStageModel, BenterBlend] | None:
    """data/models/live に本番用の v1 と合成の重みがあれば読む。無ければ None。"""
    base = Path(data_dir) / "models" / "live"
    if not (base / "blend.json").exists() or not (base / "model" / "meta.json").exists():
        return None
    return ThreeStageModel.load(base / "model"), BenterBlend.load(base / "blend.json")


def predict_v1_blend(data_dir: Path, target_date: str, races: pd.DataFrame, program_day: pd.DataFrame,
                     odds_pre_day: pd.DataFrame, model: ThreeStageModel, blend: BenterBlend) -> tuple[pd.DataFrame, pd.DataFrame]:
    """v1 と直前の単勝オッズの合成(b)で、1着確率と3連単の確率を作る。特徴量は前日までの結果だけを使う。"""
    start = (pd.Timestamp(target_date) - pd.Timedelta(days=FEATURE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    program = normalize_race_keys(load_parquet_table(data_dir, "program", end_date=target_date, start_date=start))
    if program.empty:
        raise ValueError("program is empty")
    program = program[program["race_date"] <= target_date]
    if program[program["race_date"] == target_date].empty:
        # 当日の出走表が無ければ直前に記録した出走表を使う
        program = pd.concat([program, program_day], ignore_index=True)
    info = load_parquet_table(data_dir, "race_info", end_date=target_date, start_date=start)
    res = normalize_race_keys(load_parquet_table(data_dir, "race_result", end_date=target_date, start_date=start))
    hist = res[res["race_date"] < target_date] if not res.empty else res
    feat = normalize_race_keys(build_features_v1(program, info if not info.empty else None, hist if not hist.empty else None))
    day = feat[feat["race_date"] == target_date].merge(races, on=RACE_KEYS)
    act = day[~day["is_absent"].fillna(False).astype(bool)]
    if act.empty:
        raise ValueError("no runners for the target day")
    pre = normalize_race_keys(odds_pre_day)
    pre = pre[pre["snapshot"] == "pre"] if "snapshot" in pre.columns else pre
    mw = market_win_probs(pick_snapshot(pre, how="earliest"))
    if mw.empty:
        raise ValueError("no pre-race win odds")
    v1_win = model.predict_win(act)
    win = normalize_race_keys(blend.predict_win(v1_win, mw))
    rt3 = normalize_race_keys(blend.predict_rt3_conditional(model, act, mw, v1_win))
    if win.empty or rt3.empty:
        raise ValueError("blend produced no predictions")
    return win, rt3


def run_settle_daily(data_dir: Path, target_date: str) -> int:
    pp = normalize_race_keys(load_parquet_table(data_dir, "program_pre", end_date=target_date, start_date=target_date))
    pp = pp[pp["race_date"] == target_date] if not pp.empty else pp
    if pp.empty:
        logger.info("date=%s: 直前データの記録がないため終了", target_date)
        return 0

    program_day = select_earliest_snapshot(pp, RACE_KEYS + ["car_no"]).drop(columns=["fetched_at", "minutes_to_close"], errors="ignore")
    races = program_day[RACE_KEYS].drop_duplicates().reset_index(drop=True)

    results = normalize_race_keys(load_parquet_table(data_dir, "race_result", end_date=target_date))
    hist = results[results["race_date"] < target_date] if not results.empty else results
    day_res = results[results["race_date"] == target_date].merge(races, on=RACE_KEYS) if not results.empty else results

    if day_res.empty:
        # 結果がまだ取得されていない日は答え合わせしない（外れ扱いで記録してしまうのを防ぐ）
        logger.warning("date=%s: 結果が未取得のため答え合わせを見送り", target_date)
        return 0

    def day_table(name):
        df = normalize_race_keys(load_parquet_table(data_dir, name, end_date=target_date, start_date=target_date))
        return df[df["race_date"] == target_date].merge(races, on=RACE_KEYS) if not df.empty else df

    odds_pre_day = day_table("odds_pre")
    model_name = "v0"
    live = load_live_model(data_dir)
    pred = rt3 = None
    if live is not None and not odds_pre_day.empty:
        try:
            pred, rt3 = predict_v1_blend(data_dir, target_date, races, program_day, odds_pre_day, *live)
            pred = pred.merge(races, on=RACE_KEYS)
            model_name = "v1_blend"
        except Exception as e:  # 本番用の予想を作れない日は v0 で続ける（記録を止めない）
            logger.warning("date=%s: v1 合成の予想を作れないため v0 を使う（%s: %s）", target_date, type(e).__name__, e)
            pred = rt3 = None
    if model_name == "v1_blend":
        recs = select_recommendations(rt3, odds_pre_day, min_ev=V1_MIN_EV, min_prob=V1_MIN_PROB,
                                      max_odds=V1_MAX_ODDS, max_per_race=V1_MAX_RECS_PER_RACE)
    else:
        pred = normalize_race_keys(BaselineModelV0().predict_historical(program_day, hist)).merge(races, on=RACE_KEYS)
        recs = select_recommendations(normalize_race_keys(generate_rt3_combos(pred)), odds_pre_day)
    settled = settle_recommendations(recs, day_res, day_table("refund"), day_table("odds"))
    preds_tbl = build_predictions_table(pred, day_res)

    ll = brier = float("nan")
    if not day_res.empty:
        m = evaluate_predictions(pred, day_res)["all_races"]
        ll = m.get("race_multiclass_logloss") if m.get("race_multiclass_logloss") is not None else float("nan")
        brier = m.get("brier_score") if m.get("brier_score") is not None else float("nan")

    n, hits = len(settled), int(settled["is_hit"].sum()) if len(settled) else 0
    roi = float(settled["return_amount"].sum() / n) if n else 0.0
    tracker = data_dir / "tracker"
    upsert_monthly(tracker / "predictions", target_date, preds_tbl, RACE_KEYS + ["car_no"])
    upsert_monthly(tracker / "recommendations", target_date, settled, RACE_KEYS + ["combo"])
    upsert_daily_summary(tracker, {"race_date": target_date, "num_races": len(races), "log_loss": ll,
                                   "brier_score": brier, "num_recs": n, "num_hits": hits, "roi": roi, "model": model_name})
    generate_dashboard_html(tracker)
    logger.info("date=%s model=%s races=%d logloss=%.3f recs=%d hits=%d roi=%.2f", target_date, model_name, len(races), ll, n, hits, roi)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Settle daily autorace predictions.")
    p.add_argument("--date", default=None, help="YYYY-MM-DD（省略時はJSTの前日）")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    a = p.parse_args(argv)
    return run_settle_daily(a.data_dir, a.date or default_target_date())


if __name__ == "__main__":
    sys.exit(main())

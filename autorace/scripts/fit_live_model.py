# autorace/scripts/fit_live_model.py （Claude作成）
# 毎日の答え合わせ（settle_daily）で使う本番用の予想を作る。
# 直近 blend_months か月より前の train_months か月で v1（3段階モデル）を学習し、
# 直近 blend_months か月で、v1 の1着確率と市場（単勝オッズ）の合成の重み（Benter 合成）を当てはめる。
# 出力：<out_dir>/live/model/（v1 のモデル）、<out_dir>/live/blend.json（合成の重み）、<out_dir>/live/meta.json（期間と件数）
# eval_blend の試算（2026-03〜11、約5700レース）で、合成(b)・期待値1.3以上・オッズ100倍以下・1レース1点が
# 回収率 約80%（v1 単独や v0 より良い）だったので、settle_daily はこの組み合わせを使う。
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from autorace.model.blend import BenterBlend
from autorace.model.features_v1 import build_features_v1
from autorace.model.market import market_win_probs
from autorace.model.v1 import ThreeStageModel
from autorace.scripts.backtest_v0 import load_parquet_table
from autorace.scripts.quick_eval_v1 import _norm
from autorace.scripts.train_v1 import attach_targets


def split_periods(end_date: str, train_months: int, blend_months: int) -> dict[str, str]:
    end = pd.Timestamp(end_date)
    blend_start = end - pd.DateOffset(months=blend_months) + pd.Timedelta(days=1)
    train_start = blend_start - pd.DateOffset(months=train_months)
    f = "%Y-%m-%d"
    return {"train_start": train_start.strftime(f), "blend_start": blend_start.strftime(f), "end": end.strftime(f)}


def fit_live_model(data_dir: str | Path, out_dir: str | Path, end_date: str, train_months: int = 12,
                   blend_months: int = 3) -> dict[str, Any]:
    d = Path(data_dir)
    program = load_parquet_table(d, "program", end_date)
    result = load_parquet_table(d, "race_result", end_date)
    info = load_parquet_table(d, "race_info", end_date)
    odds = load_parquet_table(d, "odds", end_date)
    if program.empty or result.empty or odds.empty:
        raise RuntimeError("program / race_result / odds is empty.")
    program, result, odds = _norm(program), _norm(result), _norm(odds)
    odds = odds[odds["bet_type"].isin(["tns", "rt3"])]
    per = split_periods(end_date, train_months, blend_months)

    full = attach_targets(build_features_v1(program, info if not info.empty else None, result), result)
    tr = full[(full["race_date"] >= per["train_start"]) & (full["race_date"] < per["blend_start"])]
    bl = full[(full["race_date"] >= per["blend_start"]) & (full["race_date"] <= per["end"])]
    if tr.empty or bl.empty:
        raise RuntimeError(f"not enough rows: train={len(tr)} blend={len(bl)}")
    bl_res = result[(result["race_date"] >= per["blend_start"]) & (result["race_date"] <= per["end"])]
    bl_odds = odds[(odds["race_date"] >= per["blend_start"]) & (odds["race_date"] <= per["end"])]

    model = ThreeStageModel().fit(tr, bl)  # 温度は合成の期間で決める（eval_blend と同じ）
    bl_act = bl[~bl["is_absent"].fillna(False).astype(bool)]
    blend = BenterBlend()
    blend.fit(model.predict_win(bl_act), market_win_probs(bl_odds), bl_res)

    live = Path(out_dir) / "live"
    model.save(live / "model")
    blend.save(live / "blend.json")
    meta = {"periods": per, "train_rows": int(len(tr)), "blend_rows": int(len(bl)),
            "temperatures": model.temperatures, "blend": blend.to_dict()}
    (live / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[Live] train {per['train_start']}..{per['blend_start']} rows={len(tr)} | blend ..{per['end']} rows={len(bl)} "
          f"| alpha={blend.alpha:.3f} beta={blend.beta:.3f} | fit={blend.fit_info_}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit the live v1 + market blend model used by settle_daily.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--blend-months", type=int, default=3)
    a = ap.parse_args()
    fit_live_model(a.data_dir, a.out_dir, a.end_date, a.train_months, a.blend_months)


if __name__ == "__main__":
    main()

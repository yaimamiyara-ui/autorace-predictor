# autorace/scripts/backtest_v0.py（Gemini作成、Claude確認・修正）
# 修正点：的中判定を払戻データではなく結果の着順で行う（払戻の推測パーサが読めなかったレースを外れ扱いにしないため）。
#         戻りは払戻があればそれ、無ければ確定オッズで代用する。
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field

from autorace.model.combos import generate_rt3_combos
from autorace.model.evaluate import evaluate_predictions
from autorace.model.features import RACE_KEYS
from autorace.model.v0 import BaselineModelV0, V0Params

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BacktestConfig(BaseModel):
    data_dir: Path = Field(default=Path("data"))
    output_dir: Path = Field(default=Path("data/backtest/v0"))
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    ev_thresholds: list[float] = Field(default=[1.0, 1.2, 1.4])
    max_picks_per_race: int = Field(default=5, ge=1)


def load_parquet_table(data_dir: Path, table_name: str, end_date: str, start_date: str | None = None) -> pd.DataFrame:
    """data/{table}/YYYY/YYYY-MM.parquet を期間で絞って結合する"""
    base = data_dir / table_name
    if not base.exists():
        return pd.DataFrame()
    dfs = []
    for f in sorted(base.glob("*/*.parquet")):
        ym = f.stem
        if ym > end_date[:7] or (start_date and ym < start_date[:7]):
            continue
        dfs.append(pd.read_parquet(f))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["race_date"] <= end_date]
    if start_date:
        df = df[df["race_date"] >= start_date]
    return df.reset_index(drop=True)


def actual_rt3(result_df: pd.DataFrame) -> pd.DataFrame:
    """結果から各レースの実際の1-2-3着（3連単の正解）を作る"""
    r = result_df[result_df["order"].isin([1, 2, 3])].copy()
    r["order"] = r["order"].astype(int)
    piv = r.pivot_table(index=RACE_KEYS, columns="order", values="car_no", aggfunc="first")
    piv = piv.dropna(subset=[c for c in (1, 2, 3) if c in piv.columns])
    if not all(c in piv.columns for c in (1, 2, 3)):
        return pd.DataFrame(columns=RACE_KEYS + ["combo"])
    piv["combo"] = piv[1].astype(int).astype(str) + "-" + piv[2].astype(int).astype(str) + "-" + piv[3].astype(int).astype(str)
    return piv.reset_index()[RACE_KEYS + ["combo"]]


def evaluate_ev_strategies(combos_df, odds_df, refund_df, result_df, thresholds, max_picks=5):
    empty = {"num_races_bet": 0, "num_combos_selected": 0, "num_hits": 0,
             "combo_hit_rate": 0.0, "race_hit_rate": 0.0, "avg_return_multiplier": 0.0}
    if combos_df.empty or odds_df.empty:
        return {f"ev_{th:.1f}": {"threshold": th, **empty} for th in thresholds}, pd.DataFrame()

    odds_rt3 = odds_df[odds_df["bet_type"] == "rt3"][RACE_KEYS + ["combo", "odds"]].copy()
    odds_rt3["odds"] = pd.to_numeric(odds_rt3["odds"], errors="coerce")
    odds_rt3 = odds_rt3[odds_rt3["odds"] > 0].dropna(subset=["odds"])
    merged = combos_df.merge(odds_rt3, on=RACE_KEYS + ["combo"], how="inner")

    hits = actual_rt3(result_df)
    hits["is_hit"] = 1
    merged = merged.merge(hits, on=RACE_KEYS + ["combo"], how="left")
    merged["is_hit"] = merged["is_hit"].fillna(0).astype(int)

    if not refund_df.empty:
        ref = refund_df[refund_df["bet_type"] == "rt3"][RACE_KEYS + ["combo", "payout"]].copy()
        ref["payout"] = pd.to_numeric(ref["payout"], errors="coerce")
        ref = ref.groupby(RACE_KEYS + ["combo"], as_index=False)["payout"].sum()
        merged = merged.merge(ref, on=RACE_KEYS + ["combo"], how="left")
    else:
        merged["payout"] = float("nan")
    # 的中なら払戻/100、払戻が読めていなければ確定オッズで代用。外れは0。
    merged["return_mult"] = merged["is_hit"] * merged["payout"].div(100.0).fillna(merged["odds"])

    merged["ev"] = merged["combo_prob"] * merged["odds"]
    merged = merged.sort_values(RACE_KEYS + ["ev", "combo_prob"], ascending=[True, True, True, False, False])

    summary, selected = {}, []
    for th in thresholds:
        cand = merged[merged["ev"] >= th].copy()
        if cand.empty:
            summary[f"ev_{th:.1f}"] = {"threshold": th, **empty}
            continue
        cand["rank_in_race"] = cand.groupby(RACE_KEYS, sort=False).cumcount() + 1
        picked = cand[cand["rank_in_race"] <= max_picks].copy()
        picked["threshold"] = th
        selected.append(picked)
        n_combos, n_hits = len(picked), int(picked["is_hit"].sum())
        n_races = int(picked.groupby(RACE_KEYS, sort=False).ngroups)
        race_hits = int(picked.groupby(RACE_KEYS, sort=False)["is_hit"].max().sum())
        summary[f"ev_{th:.1f}"] = {
            "threshold": th, "num_races_bet": n_races, "num_combos_selected": n_combos, "num_hits": n_hits,
            "combo_hit_rate": n_hits / n_combos if n_combos else 0.0,
            "race_hit_rate": race_hits / n_races if n_races else 0.0,
            "avg_return_multiplier": float(picked["return_mult"].mean()) if n_combos else 0.0,
        }
    return summary, (pd.concat(selected, ignore_index=True) if selected else pd.DataFrame())


def run_backtest(config: BacktestConfig) -> dict:
    program_all = load_parquet_table(config.data_dir, "program", end_date=config.end_date)
    result_all = load_parquet_table(config.data_dir, "race_result", end_date=config.end_date)
    odds_eval = load_parquet_table(config.data_dir, "odds", end_date=config.end_date, start_date=config.start_date)
    refund_eval = load_parquet_table(config.data_dir, "refund", end_date=config.end_date, start_date=config.start_date)

    pred_all = BaselineModelV0(V0Params()).predict_historical(program_all, result_all)
    in_period = lambda df: df[(df["race_date"] >= config.start_date) & (df["race_date"] <= config.end_date)].reset_index(drop=True)
    pred_eval, result_eval = in_period(pred_all), in_period(result_all)

    eval_metrics = evaluate_predictions(pred_eval, result_eval, n_bins=20)
    combos_eval = generate_rt3_combos(pred_eval)
    ev_summary, picked = evaluate_ev_strategies(combos_eval, odds_eval, refund_eval, result_eval,
                                                config.ev_thresholds, config.max_picks_per_race)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pred_eval.to_parquet(config.output_dir / "predictions.parquet", index=False)
    if not picked.empty:
        picked.to_parquet(config.output_dir / "selected_combos.parquet", index=False)
    summary = {"period": {"start_date": config.start_date, "end_date": config.end_date},
               "accuracy": eval_metrics, "ev_reference": ev_summary}
    with open(config.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Backtest Summary: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest Baseline Model v0")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="data/backtest/v0")
    a = p.parse_args()
    run_backtest(BacktestConfig(data_dir=Path(a.data_dir), output_dir=Path(a.output_dir),
                                start_date=a.start_date, end_date=a.end_date))


if __name__ == "__main__":
    main()

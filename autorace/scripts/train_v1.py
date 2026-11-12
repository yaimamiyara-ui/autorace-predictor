# autorace/scripts/train_v1.py （Gemini作成、Claude確認・修正）
# 修正点：data_dir を Path にする（load_parquet_table は Path 前提）、ログを件数と指標だけにする。
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autorace.model.combos import generate_rt3_combos
from autorace.model.evaluate_v1 import evaluate_predictions
from autorace.model.features import RACE_KEYS
from autorace.model.features_v1 import build_features_v1
from autorace.model.v0 import BaselineModelV0
from autorace.model.v1 import ThreeStageModel
from autorace.scripts.backtest_v0 import load_parquet_table


def generate_walk_forward_splits(min_date_str: str, end_date_str: str, train_months: int = 12, valid_months: int = 2,
                                 test_months: int = 2, step_months: int = 2) -> list[dict[str, str]]:
    """学習→検証→テストの期間を step_months ずつずらして作る。"""
    min_dt, end_dt = pd.Timestamp(min_date_str), pd.Timestamp(end_date_str)
    splits: list[dict[str, str]] = []
    cursor = min_dt
    while True:
        train_end = cursor + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        valid_start = train_end + pd.Timedelta(days=1)
        valid_end = valid_start + pd.DateOffset(months=valid_months) - pd.Timedelta(days=1)
        test_start = valid_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_start > end_dt:
            break
        test_end = min(test_end, end_dt)
        splits.append({k: v.strftime("%Y-%m-%d") for k, v in (
            ("train_start", cursor), ("train_end", train_end), ("valid_start", valid_start),
            ("valid_end", valid_end), ("test_start", test_start), ("test_end", test_end))})
        if test_end >= end_dt:
            break
        cursor = cursor + pd.DateOffset(months=step_months)
    return splits


def attach_targets(feat_df: pd.DataFrame, race_result: pd.DataFrame) -> pd.DataFrame:
    """特徴量に着順・事故情報を結合する（学習の目的変数用。特徴量には使わない）。"""
    target_cols = [c for c in ["order", "accident_code", "foul_code"] if c in race_result.columns]
    res_sub = race_result[RACE_KEYS + ["car_no"] + target_cols].drop_duplicates(subset=RACE_KEYS + ["car_no"]).copy()
    res_sub["race_date"] = res_sub["race_date"].astype(str)
    res_sub["place_code"] = pd.to_numeric(res_sub["place_code"], errors="coerce").fillna(0).astype(np.int32)
    res_sub["race_no"] = pd.to_numeric(res_sub["race_no"], errors="coerce").fillna(0).astype(np.int32)
    drop_existing = [c for c in target_cols if c in feat_df.columns]
    base = feat_df.drop(columns=drop_existing) if drop_existing else feat_df
    return base.merge(res_sub, on=RACE_KEYS + ["car_no"], how="left")


def _in(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df["race_date"] >= start) & (df["race_date"] <= end)]


def _fmt(x: Any) -> str:
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def run_training_pipeline(data_dir: str | Path, out_dir: str | Path, end_date: str, train_months: int = 12,
                          valid_months: int = 2, test_months: int = 2, step_months: int = 2) -> dict[str, Any]:
    data_dir = Path(data_dir)
    program = load_parquet_table(data_dir, "program", end_date=end_date)
    race_result = load_parquet_table(data_dir, "race_result", end_date=end_date)
    race_info = load_parquet_table(data_dir, "race_info", end_date=end_date)
    if program.empty or race_result.empty:
        raise RuntimeError("program or race_result is empty.")
    for df in (program, race_result):
        df["race_date"] = df["race_date"].astype(str)
    print(f"[Data] program_rows={len(program)} result_rows={len(race_result)} race_info_rows={len(race_info)}")

    # 全期間の特徴量を一括で作る（内部で前日以前の結果だけを参照）
    full_df = attach_targets(build_features_v1(program, race_info, race_result), race_result)
    v0_pred_all = BaselineModelV0().predict_historical(program, race_result)
    v0_pred_all["race_date"] = v0_pred_all["race_date"].astype(str)

    splits = generate_walk_forward_splits(str(full_df["race_date"].min()), end_date, train_months, valid_months,
                                          test_months, step_months)
    fold_reports: list[dict[str, Any]] = []
    for idx, sp in enumerate(splits, start=1):
        tr_df = _in(full_df, sp["train_start"], sp["train_end"])
        va_df = _in(full_df, sp["valid_start"], sp["valid_end"])
        te_df = _in(full_df, sp["test_start"], sp["test_end"])
        te_res = _in(race_result, sp["test_start"], sp["test_end"])
        if tr_df.empty or va_df.empty or te_df.empty or te_res.empty:
            continue

        model = ThreeStageModel().fit(tr_df, va_df)
        v1_win_df = model.predict_win(te_df)
        eval_v1_3stage = evaluate_predictions(v1_win_df, model.predict_rt3(te_df), te_res)
        eval_v1_harville = evaluate_predictions(v1_win_df, generate_rt3_combos(v1_win_df), te_res)
        v0_te = _in(v0_pred_all, sp["test_start"], sp["test_end"])
        eval_v0_harville = evaluate_predictions(v0_te, generate_rt3_combos(v0_te), te_res)

        fold_reports.append({
            "fold": idx, "period": sp,
            "counts": {"train_rows": int(len(tr_df)), "valid_rows": int(len(va_df)), "test_rows": int(len(te_df)),
                       "test_win_races": eval_v1_3stage["n_win_races"], "test_rt3_races": eval_v1_3stage["n_rt3_races"]},
            "temperatures": model.temperatures,
            "best_iterations": {s: int(b.best_iteration or b.current_iteration()) for s, b in model.boosters.items()},
            "models": {"v1_three_stage": eval_v1_3stage, "v1_m1_harville": eval_v1_harville, "v0_harville": eval_v0_harville},
        })
        print(f"[Fold {idx}] test={sp['test_start']}..{sp['test_end']} races={eval_v1_3stage['n_rt3_races']} | "
              f"win_ll v0={_fmt(eval_v0_harville['win_logloss'])} v1={_fmt(eval_v1_3stage['win_logloss'])} | "
              f"rt3_ll v0harv={_fmt(eval_v0_harville['rt3_logloss'])} v1harv={_fmt(eval_v1_harville['rt3_logloss'])} "
              f"v1_3stage={_fmt(eval_v1_3stage['rt3_logloss'])} | ece v1_3stage={_fmt(eval_v1_3stage['rt3_ece'])}")

    # 直近までのデータで本番モデル（最後の valid_months を検証に使う）
    end_dt = pd.Timestamp(end_date)
    prod_valid_start = (end_dt - pd.DateOffset(months=valid_months) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prod_train_start = (pd.Timestamp(prod_valid_start) - pd.DateOffset(months=train_months)).strftime("%Y-%m-%d")
    prod_tr = full_df[(full_df["race_date"] >= prod_train_start) & (full_df["race_date"] < prod_valid_start)]
    prod_va = _in(full_df, prod_valid_start, end_date)
    if prod_tr.empty:
        prod_tr = full_df[full_df["race_date"] < prod_valid_start]
    prod_model = ThreeStageModel().fit(prod_tr, prod_va)
    model_save_dir = Path(out_dir) / "models" / "v1" / end_date
    prod_model.save(model_save_dir)

    # 特徴量の重要度（gain、上位20）。効いていない特徴の見直しに使う
    importance = {}
    for stage, b in prod_model.boosters.items():
        gains = b.feature_importance(importance_type="gain")
        total = float(gains.sum()) or 1.0
        pairs = sorted(zip(b.feature_name(), gains), key=lambda x: -x[1])[:20]
        importance[f"m{stage}"] = {n: round(float(g) / total, 4) for n, g in pairs}

    summary_metrics: dict[str, Any] = {}
    for m_key in ("v1_three_stage", "v1_m1_harville", "v0_harville"):
        if fold_reports:
            summary_metrics[m_key] = {
                metric: float(np.nanmean([f["models"][m_key][metric] for f in fold_reports]))
                for metric in ("win_logloss", "rt3_logloss", "rt3_top1_hit_rate", "rt3_ece")}

    report_data = {
        "end_date": end_date,
        "config": {"train_months": train_months, "valid_months": valid_months, "test_months": test_months,
                   "step_months": step_months},
        "summary_mean": summary_metrics, "folds": fold_reports,
        "production_model": {"model_dir": str(model_save_dir), "train_start": prod_train_start,
                             "valid_start": prod_valid_start, "end_date": end_date, "train_rows": int(len(prod_tr)),
                             "valid_rows": int(len(prod_va)), "temperatures": prod_model.temperatures,
                             "feature_importance_gain": importance},
    }
    text = json.dumps(report_data, ensure_ascii=False, indent=2)
    (model_save_dir / "report.json").write_text(text, encoding="utf-8")
    (Path(out_dir) / "report.json").write_text(text, encoding="utf-8")
    for m_key, m in summary_metrics.items():
        print(f"[Mean] {m_key}: " + " ".join(f"{k}={_fmt(v)}" for k, v in m.items()))
    for st, imp in importance.items():
        print(f"[Importance {st}] " + " ".join(f"{k}={v}" for k, v in list(imp.items())[:8]))
    print(f"[Production] train_rows={len(prod_tr)} valid_rows={len(prod_va)} T={prod_model.temperatures}")
    return report_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate autorace v1 ThreeStageModel.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--valid-months", type=int, default=2)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--step-months", type=int, default=2)
    a = parser.parse_args()
    run_training_pipeline(a.data_dir, a.out_dir, a.end_date, a.train_months, a.valid_months, a.test_months, a.step_months)


if __name__ == "__main__":
    main()

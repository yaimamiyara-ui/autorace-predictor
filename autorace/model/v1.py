# autorace/model/v1.py （Gemini作成、Claude確認）
# 3段階モデル：M1(1着)・M2(1着を除いた2着)・M3(1・2着を除いた3着)。各段にグループsoftmaxのLightGBMと温度スケーリング。
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from autorace.model.features import RACE_KEYS
from autorace.model.features_v1 import CATEGORICAL_COLS, FEATURE_COLS


def compute_group_softmax(preds: np.ndarray, group_sizes: np.ndarray) -> np.ndarray:
    """グループ（レース）ごとの数値安定な softmax。"""
    if len(preds) == 0:
        return np.array([], dtype=np.float64)
    group_sizes = np.asarray(group_sizes, dtype=np.int64)
    group_starts = np.concatenate(([0], np.cumsum(group_sizes)[:-1]))
    max_per_group = np.maximum.reduceat(preds, group_starts)
    exp_preds = np.exp(preds - np.repeat(max_per_group, group_sizes))
    sum_exp = np.add.reduceat(exp_preds, group_starts)
    return exp_preds / np.repeat(sum_exp, group_sizes)


def group_softmax_objective(preds: np.ndarray, train_data: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """LightGBM 4系の params["objective"] 用。grad = p - y, hess = max(p(1-p), 1e-6)"""
    labels = train_data.get_label()
    probs = compute_group_softmax(preds, train_data.get_group())
    return probs - labels, np.maximum(probs * (1.0 - probs), 1e-6)


def group_softmax_eval(preds: np.ndarray, eval_data: lgb.Dataset) -> tuple[str, float, bool]:
    """正解（その段で決まる着の車）の平均負対数尤度。"""
    labels = eval_data.get_label()
    group_sizes = eval_data.get_group()
    if group_sizes is None or len(group_sizes) == 0:
        return "group_logloss", 0.0, False
    probs = compute_group_softmax(preds, group_sizes)
    win_probs = np.clip(probs[labels == 1.0], 1e-15, 1.0)
    if len(win_probs) == 0:
        return "group_logloss", 0.0, False
    return "group_logloss", float(-np.mean(np.log(win_probs))), False


def fit_temperature(logits: np.ndarray, labels: np.ndarray, group_sizes: np.ndarray) -> float:
    """検証データのグループ NLL を最小にする温度 T。"""
    if len(logits) == 0 or len(group_sizes) == 0 or np.sum(labels == 1.0) == 0:
        return 1.0

    def nll(temp: float) -> float:
        probs = compute_group_softmax(logits / temp, group_sizes)
        return float(-np.mean(np.log(np.clip(probs[labels == 1.0], 1e-15, 1.0))))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x) if res.success else 1.0


def _is_nonempty_code(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    return ~s.isin(["", "0", "0.0", "None", "nan", "<NA>"])


def prepare_stage_dataset(df: pd.DataFrame, stage: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """各段の学習・検証データ。欠車を除きレース順に並べる。
    M1: 1着が1人に決まるレース / M2: 事故なしで1・2着が決まるレースから1着の行を除く /
    M3: 事故なしで1〜3着が決まるレースから1・2着の行を除く。"""
    empty = (df.iloc[0:0].copy(), np.array([], dtype=np.float64), np.array([], dtype=np.int32))
    if df.empty:
        return empty

    work = df[~df["is_absent"].fillna(False).astype(bool)].copy()
    work["order_num"] = pd.to_numeric(work.get("order"), errors="coerce")

    has_acc = pd.Series(False, index=work.index)
    if "accident_code" in work.columns:
        has_acc = has_acc | _is_nonempty_code(work["accident_code"])
    if "foul_code" in work.columns:
        has_acc = has_acc | _is_nonempty_code(work["foul_code"])
    has_acc = has_acc | work["order_num"].isna()  # 出走して着順なしも事故扱い
    work["_has_acc"] = has_acc
    work["_is_1"] = (work["order_num"] == 1).astype(np.int32)
    work["_is_2"] = (work["order_num"] == 2).astype(np.int32)
    work["_is_3"] = (work["order_num"] == 3).astype(np.int32)

    r_grp = work.groupby(RACE_KEYS, sort=False)
    c1, c2, c3 = (r_grp[c].transform("sum") for c in ("_is_1", "_is_2", "_is_3"))
    race_acc = r_grp["_has_acc"].transform("any")
    n_active = r_grp["_is_1"].transform("count")

    if stage == 1:
        sub = work[(c1 == 1) & (n_active >= 3)].copy()
    elif stage == 2:
        valid_race = (c1 == 1) & (c2 == 1) & (~race_acc) & (n_active >= 3)
        sub = work[valid_race & (work["order_num"] != 1)].copy()
    elif stage == 3:
        valid_race = (c1 == 1) & (c2 == 1) & (c3 == 1) & (~race_acc) & (n_active >= 4)
        sub = work[valid_race & (~work["order_num"].isin([1, 2]))].copy()
    else:
        raise ValueError(f"Invalid stage: {stage}")
    if sub.empty:
        return empty

    sub = sub.sort_values(RACE_KEYS + ["car_no"]).reset_index(drop=True)
    labels = (sub["order_num"] == stage).astype(np.float64).to_numpy()
    group_sizes = sub.groupby(RACE_KEYS, sort=False).size().to_numpy(dtype=np.int32)
    return sub, labels, group_sizes


class ThreeStageModel:
    def __init__(self, params: dict[str, Any] | None = None, num_boost_round: int = 300,
                 early_stopping_rounds: int = 30, feature_cols: list[str] | None = None,
                 categorical_cols: list[str] | None = None) -> None:
        default_params: dict[str, Any] = {
            "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 30,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l1": 0.1, "lambda_l2": 1.0, "seed": 42, "deterministic": True,
            "force_col_wise": True, "verbosity": -1, "n_jobs": 2,
        }
        if params is not None:
            default_params.update(params)
        self.base_params = default_params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
        self.categorical_cols = list(categorical_cols) if categorical_cols is not None else list(CATEGORICAL_COLS)
        self.boosters: dict[int, lgb.Booster] = {}
        self.temperatures: dict[int, float] = {1: 1.0, 2: 1.0, 3: 1.0}

    def _extract_X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for col in self.feature_cols:
            X[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else 0.0
        for cat_col in self.categorical_cols:
            if cat_col in X.columns:
                X[cat_col] = X[cat_col].fillna(0).astype(np.int32)
        return X

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> "ThreeStageModel":
        for stage in (1, 2, 3):
            tr_sub, y_tr, g_tr = prepare_stage_dataset(train_df, stage=stage)
            va_sub, y_va, g_va = prepare_stage_dataset(valid_df, stage=stage)
            if len(tr_sub) == 0 or len(g_tr) == 0:
                raise ValueError(f"Stage {stage} training dataset is empty.")

            dtrain = lgb.Dataset(self._extract_X(tr_sub), label=y_tr, group=g_tr,
                                 categorical_feature=self.categorical_cols, free_raw_data=False)
            train_params = dict(self.base_params)
            train_params["objective"] = group_softmax_objective
            train_params["metric"] = "None"
            valid_sets, valid_names, callbacks = [dtrain], ["train"], []
            has_valid = len(va_sub) > 0 and len(g_va) > 0
            if has_valid:
                dvalid = lgb.Dataset(self._extract_X(va_sub), label=y_va, group=g_va, reference=dtrain,
                                     categorical_feature=self.categorical_cols, free_raw_data=False)
                valid_sets.append(dvalid)
                valid_names.append("valid")
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds, first_metric_only=True, verbose=False))

            booster = lgb.train(params=train_params, train_set=dtrain, num_boost_round=self.num_boost_round,
                                valid_sets=valid_sets, valid_names=valid_names, feval=group_softmax_eval,
                                callbacks=callbacks)
            self.boosters[stage] = booster
            self.temperatures[stage] = (fit_temperature(booster.predict(self._extract_X(va_sub)), y_va, g_va)
                                        if has_valid else 1.0)
        return self

    def predict_win(self, df: pd.DataFrame) -> pd.DataFrame:
        """各出走車の1着確率 pred_prob 列を付けて返す（欠車は0）。"""
        if 1 not in self.boosters:
            raise RuntimeError("Model has not been fitted or loaded.")
        out = df.copy()
        out["pred_prob"] = 0.0
        if out.empty:
            return out
        out["_orig_order"] = np.arange(len(out))
        out = out.sort_values(RACE_KEYS + ["car_no"]).reset_index(drop=True)
        active_mask = ~out["is_absent"].fillna(False).astype(bool)
        active_df = out[active_mask]
        if not active_df.empty:
            s1 = self.boosters[1].predict(self._extract_X(active_df)) / self.temperatures[1]
            g_sizes = active_df.groupby(RACE_KEYS, sort=False).size().to_numpy(dtype=np.int32)
            out.loc[active_mask, "pred_prob"] = compute_group_softmax(s1, g_sizes)
        out = out.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)
        out.index = df.index
        return out

    def predict_rt3(self, df: pd.DataFrame) -> pd.DataFrame:
        """P(i,j,k) = softmax(s1/T1)_i × softmax_{除i}(s2/T2)_j × softmax_{除i,j}(s3/T3)_k（各レース合計1）"""
        for stage in (1, 2, 3):
            if stage not in self.boosters:
                raise RuntimeError("Model has not been fitted or loaded.")
        cols_out = RACE_KEYS + ["bet_type", "combo", "combo_prob"]
        work = df[~df["is_absent"].fillna(False).astype(bool)].copy() if not df.empty else df
        if work.empty:
            return pd.DataFrame(columns=cols_out)

        work["car_no"] = pd.to_numeric(work["car_no"], errors="coerce").fillna(0).astype(np.int32)
        work = work.sort_values(RACE_KEYS + ["car_no"]).reset_index(drop=True)
        X_all = self._extract_X(work)
        s1_all = self.boosters[1].predict(X_all) / self.temperatures[1]
        s2_all = self.boosters[2].predict(X_all) / self.temperatures[2]
        s3_all = self.boosters[3].predict(X_all) / self.temperatures[3]

        car_nos_all = work["car_no"].to_numpy(dtype=np.int32)
        place_all, date_all, rno_all = (work[c].to_numpy() for c in RACE_KEYS)
        group_sizes = work.groupby(RACE_KEYS, sort=False).size().to_numpy(dtype=np.int64)
        group_starts = np.concatenate(([0], np.cumsum(group_sizes)[:-1]))

        rec_place, rec_date, rec_rno, rec_combo, rec_prob = [], [], [], [], []
        for start, n in zip(group_starts, group_sizes):
            if n < 3:
                continue
            end = start + n
            cars = car_nos_all[start:end]
            w1 = np.exp(s1_all[start:end] - np.max(s1_all[start:end]))
            w2 = np.exp(s2_all[start:end] - np.max(s2_all[start:end]))
            w3 = np.exp(s3_all[start:end] - np.max(s3_all[start:end]))
            p1 = w1 / np.sum(w1)
            p2 = w2[None, :] / np.maximum((np.sum(w2) - w2)[:, None], 1e-15)
            np.fill_diagonal(p2, 0.0)
            denom3 = np.sum(w3) - w3[:, None] - w3[None, :]
            p3 = w3[None, None, :] / np.maximum(denom3[:, :, None], 1e-15)
            idx = np.arange(n)
            valid_3d = ((idx[:, None, None] != idx[None, :, None]) & (idx[:, None, None] != idx[None, None, :])
                        & (idx[None, :, None] != idx[None, None, :]))
            p_ijk = np.where(valid_3d, p1[:, None, None] * p2[:, :, None] * p3, 0.0)
            total_p = float(np.sum(p_ijk))
            if total_p > 0:
                p_ijk = p_ijk / total_p
            ii, jj, kk = np.where(valid_3d)
            cnt = len(ii)
            rec_place.extend([place_all[start]] * cnt)
            rec_date.extend([date_all[start]] * cnt)
            rec_rno.extend([rno_all[start]] * cnt)
            rec_combo.extend(f"{cars[i]}-{cars[j]}-{cars[k]}" for i, j, k in zip(ii, jj, kk))
            rec_prob.extend(p_ijk[ii, jj, kk].tolist())

        return pd.DataFrame({"place_code": rec_place, "race_date": rec_date, "race_no": rec_rno,
                             "bet_type": ["rt3"] * len(rec_prob), "combo": rec_combo, "combo_prob": rec_prob})

    def save(self, model_dir: str | Path) -> None:
        out_path = Path(model_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for stage, booster in self.boosters.items():
            booster.save_model(str(out_path / f"model_m{stage}.txt"))
        meta = {
            "temperatures": {str(k): float(v) for k, v in self.temperatures.items()},
            "feature_cols": self.feature_cols,
            "categorical_cols": self.categorical_cols,
            "base_params": {k: v for k, v in self.base_params.items() if not callable(v)},
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        (out_path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | Path) -> "ThreeStageModel":
        in_path = Path(model_dir)
        meta = json.loads((in_path / "meta.json").read_text(encoding="utf-8"))
        obj = cls(params=meta.get("base_params"), num_boost_round=meta.get("num_boost_round", 300),
                  early_stopping_rounds=meta.get("early_stopping_rounds", 30),
                  feature_cols=meta.get("feature_cols"), categorical_cols=meta.get("categorical_cols"))
        obj.temperatures = {int(k): float(v) for k, v in meta.get("temperatures", {}).items()}
        for stage in (1, 2, 3):
            m_file = in_path / f"model_m{stage}.txt"
            if m_file.exists():
                obj.boosters[stage] = lgb.Booster(model_file=str(m_file))
        return obj

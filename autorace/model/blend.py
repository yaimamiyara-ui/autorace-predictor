# autorace/model/blend.py （Gemini作成、Claude確認）
# Benter 方式：モデル確率と市場（オッズ）確率の対数線形合成。
#  1着: p_i ∝ exp(α·log p_model + β·log p_market)
#  3連単(a): v1 の3連単確率と市場の3連単確率を α3,β3 で合成
#  3連単(b): 合成した1着確率 × v1 の M2・M3 の条件付き確率
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from autorace.model.features import RACE_KEYS
from autorace.scripts.backtest_v0 import actual_rt3

_PERM_IDX_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_COMBO_STR_CACHE: dict[tuple[int, ...], np.ndarray] = {}
K = list(RACE_KEYS)


def _get_perm_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n not in _PERM_IDX_CACHE:
        trip = [(i, j, k) for i in range(n) for j in range(n) if j != i for k in range(n) if k != i and k != j]
        a = np.asarray(trip, dtype=np.int64).reshape(-1, 3)
        _PERM_IDX_CACHE[n] = (a[:, 0], a[:, 1], a[:, 2])
    return _PERM_IDX_CACHE[n]


def _get_combo_strings(cars: tuple[int, ...]) -> np.ndarray:
    if cars not in _COMBO_STR_CACHE:
        ii, jj, kk = _get_perm_indices(len(cars))
        _COMBO_STR_CACHE[cars] = np.array([f"{cars[i]}-{cars[j]}-{cars[k]}" for i, j, k in zip(ii, jj, kk)], dtype=object)
    return _COMBO_STR_CACHE[cars]


def _extract_stage_scaled_scores(v1_model: Any, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ThreeStageModel から (s1/T1, s2/T2, s3/T3)"""
    X = v1_model._extract_X(df)
    b, t = v1_model.boosters, v1_model.temperatures
    return tuple(np.asarray(b[s].predict(X), dtype=np.float64) / float(t[s]) for s in (1, 2, 3))  # type: ignore[return-value]


def _race_bounds(merged: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    codes, _ = pd.factorize(merged[K].astype(str).agg("|".join, axis=1), sort=False)
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    return starts, np.diff(np.r_[starts, len(codes)])


def _optimize_benter_weights(log_pm, log_pmkt, target, starts, counts, eps: float = 1e-15):
    """レース内 softmax の交差エントロピーを L-BFGS-B で最小化"""
    n_races = len(starts)

    def loss_and_grad(params: np.ndarray):
        u = params[0] * log_pm + params[1] * log_pmkt
        u_max = np.repeat(np.maximum.reduceat(u, starts), counts)
        exp_u = np.exp(u - u_max)
        sum_exp = np.repeat(np.add.reduceat(exp_u, starts), counts)
        log_p = (u - u_max) - np.log(np.maximum(sum_exp, eps))
        p = exp_u / np.maximum(sum_exp, eps)
        diff = p - target
        return float(-np.sum(target * log_p) / n_races), np.array(
            [np.sum(diff * log_pm) / n_races, np.sum(diff * log_pmkt) / n_races], dtype=np.float64)

    cands = [np.array(x, dtype=np.float64) for x in ([0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.25, 0.75])]
    best_init = min(cands, key=lambda x: loss_and_grad(x)[0])
    res = minimize(loss_and_grad, x0=best_init, method="L-BFGS-B", jac=True, bounds=[(-5.0, 10.0), (-5.0, 10.0)],
                   options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500})
    info = {"n_races": int(n_races), "logloss_blend": float(res.fun),
            "logloss_model_only": float(loss_and_grad(cands[1])[0]),
            "logloss_market_only": float(loss_and_grad(cands[2])[0]),
            "converged": bool(res.success), "nit": int(res.nit)}
    return float(res.x[0]), float(res.x[1]), info


def _renorm(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] / df.groupby(K, sort=False)[col].transform("sum")


def _softmax_col(df: pd.DataFrame, u: np.ndarray) -> np.ndarray:
    d = df[K].copy()
    d["_u"] = u
    d["_e"] = np.exp(d["_u"] - d.groupby(K, sort=False)["_u"].transform("max"))
    return (d["_e"] / d.groupby(K, sort=False)["_e"].transform("sum")).to_numpy()


class BenterBlend:
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, alpha3: float = 0.5, beta3: float = 0.5, eps: float = 1e-15) -> None:
        self.alpha, self.beta, self.alpha3, self.beta3, self.eps = float(alpha), float(beta), float(alpha3), float(beta3), float(eps)
        self.fit_info_: dict[str, Any] = {}
        self.fit_rt3_info_: dict[str, Any] = {}

    @staticmethod
    def _resolve_market_col(df: pd.DataFrame, fallback_col: str) -> str:
        if "market_prob" in df.columns:
            return "market_prob"
        if fallback_col in df.columns:
            return fallback_col
        raise KeyError(f"market_df must contain 'market_prob' or '{fallback_col}'")

    def _logs(self, merged: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (np.log(np.clip(merged["_mod_prob"].to_numpy(dtype=np.float64), self.eps, 1.0)),
                np.log(np.clip(merged["_mkt_prob"].to_numpy(dtype=np.float64), self.eps, 1.0)))

    # ---- 1着 ----
    def _merge_win_inputs(self, win_pred_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        mkt_col = self._resolve_market_col(market_df, "pred_prob")
        m = market_df[K + ["car_no", mkt_col]].rename(columns={mkt_col: "_mkt_prob"}).copy()
        p = win_pred_df[K + ["car_no", "pred_prob"]].rename(columns={"pred_prob": "_mod_prob"}).copy()
        m["car_no"] = pd.to_numeric(m["car_no"], errors="coerce").astype(int)
        p["car_no"] = pd.to_numeric(p["car_no"], errors="coerce").astype(int)
        merged = p.merge(m, on=K + ["car_no"], how="inner")
        if merged.empty:
            return merged
        merged = merged.sort_values(K + ["car_no"]).reset_index(drop=True)
        g = merged.groupby(K, sort=False)
        merged = merged.loc[(g["_mod_prob"].transform("sum") > 0) & (g["_mkt_prob"].transform("sum") > 0)].copy()
        if merged.empty:
            return merged
        merged["_mod_prob"] = _renorm(merged, "_mod_prob")  # 欠車を除いた出走車で再正規化
        merged["_mkt_prob"] = _renorm(merged, "_mkt_prob")
        return merged.reset_index(drop=True)

    def fit(self, win_pred_df: pd.DataFrame, market_df: pd.DataFrame, race_result: pd.DataFrame) -> "BenterBlend":
        merged = self._merge_win_inputs(win_pred_df, market_df)
        if merged.empty:
            raise ValueError("No overlapping races/cars between win_pred_df and market_df.")
        res = race_result[K + ["car_no", "order"]].copy()
        res["car_no"] = pd.to_numeric(res["car_no"], errors="coerce")
        res["order"] = pd.to_numeric(res["order"], errors="coerce")
        res = res.dropna(subset=["car_no"]).copy()
        res["car_no"] = res["car_no"].astype(int)
        res["is_win"] = (res["order"] == 1).astype(np.float64)
        merged = merged.merge(res[K + ["car_no", "is_win"]], on=K + ["car_no"], how="inner")
        merged = merged.sort_values(K + ["car_no"]).reset_index(drop=True)
        g = merged.groupby(K, sort=False)
        merged = merged.loc[(g["is_win"].transform("sum") > 0) & (g["car_no"].transform("count") >= 2)].copy()
        if merged.empty:
            raise ValueError("No valid races with 1st-place finisher found for fitting.")
        merged["target"] = _renorm(merged, "is_win")
        starts, counts = _race_bounds(merged)
        lpm, lmk = self._logs(merged)
        self.alpha, self.beta, self.fit_info_ = _optimize_benter_weights(lpm, lmk, merged["target"].to_numpy(dtype=np.float64), starts, counts, self.eps)
        return self

    def predict_win(self, win_pred_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        out_cols = K + ["car_no", "pred_prob"]
        merged = self._merge_win_inputs(win_pred_df, market_df)
        if merged.empty:
            return pd.DataFrame(columns=out_cols)
        lpm, lmk = self._logs(merged)
        merged["pred_prob"] = _softmax_col(merged, self.alpha * lpm + self.beta * lmk)
        return merged[out_cols].reset_index(drop=True)

    blended_win = predict_win

    # ---- 3連単 (a) ----
    def _merge_rt3_inputs(self, rt3_pred_df: pd.DataFrame, market_rt3_df: pd.DataFrame) -> pd.DataFrame:
        mkt_col = self._resolve_market_col(market_rt3_df, "combo_prob")
        p = rt3_pred_df[K + ["combo", "combo_prob"]].rename(columns={"combo_prob": "_mod_prob"}).copy()
        m = market_rt3_df[K + ["combo", mkt_col]].rename(columns={mkt_col: "_mkt_prob"}).copy()
        p["combo"] = p["combo"].astype(str).str.strip()
        m["combo"] = m["combo"].astype(str).str.strip()
        p = p.merge(m[K].drop_duplicates(), on=K, how="inner")  # 市場オッズのあるレースだけ
        if p.empty:
            return pd.DataFrame()
        # 市場の3連単に現れる車番＝出走車。欠車を含む組を除く
        parts = m["combo"].str.split("-", expand=True)
        active = pd.concat([m[K].assign(_car=parts[c]) for c in range(min(3, parts.shape[1]))], ignore_index=True).drop_duplicates()
        pp = p["combo"].str.split("-", expand=True)
        p["_c1"], p["_c2"], p["_c3"] = pp[0], pp[1], pp[2]
        for c in ("_c1", "_c2", "_c3"):
            p = p.merge(active.rename(columns={"_car": c}), on=K + [c], how="inner")
        p = p.drop(columns=["_c1", "_c2", "_c3"])
        if p.empty:
            return pd.DataFrame()
        merged = p.merge(m, on=K + ["combo"], how="left")
        if merged["_mkt_prob"].isna().any():  # 売れていない組は最小値の1/10で埋める
            min_mkt = merged.groupby(K, sort=False)["_mkt_prob"].transform("min")
            merged["_mkt_prob"] = merged["_mkt_prob"].fillna(np.maximum(min_mkt.fillna(1e-6) * 0.1, self.eps))
        merged = merged.sort_values(K + ["combo"]).reset_index(drop=True)
        g = merged.groupby(K, sort=False)
        merged = merged.loc[(g["_mod_prob"].transform("sum") > 0) & (g["_mkt_prob"].transform("sum") > 0)].copy()
        if merged.empty:
            return pd.DataFrame()
        merged["_mod_prob"] = _renorm(merged, "_mod_prob")
        merged["_mkt_prob"] = _renorm(merged, "_mkt_prob")
        return merged.reset_index(drop=True)

    def fit_rt3(self, rt3_pred_df: pd.DataFrame, market_rt3_df: pd.DataFrame, race_result: pd.DataFrame) -> "BenterBlend":
        merged = self._merge_rt3_inputs(rt3_pred_df, market_rt3_df)
        if merged.empty:
            raise ValueError("No overlapping rt3 races/combos between model and market.")
        act = actual_rt3(race_result).copy()
        if act.empty:
            raise ValueError("No actual rt3 results available for fit_rt3.")
        act["combo"] = act["combo"].astype(str).str.strip()
        act["is_hit"] = 1.0
        merged = merged.merge(act[K + ["combo", "is_hit"]], on=K + ["combo"], how="left")
        merged["is_hit"] = merged["is_hit"].fillna(0.0)
        merged = merged.sort_values(K + ["combo"]).reset_index(drop=True)
        g = merged.groupby(K, sort=False)
        merged = merged.loc[(g["is_hit"].transform("sum") > 0) & (g["combo"].transform("count") >= 6)].copy()
        if merged.empty:
            raise ValueError("No valid rt3 races with matching winning combo found.")
        merged["target"] = _renorm(merged, "is_hit")
        starts, counts = _race_bounds(merged)
        lpm, lmk = self._logs(merged)
        self.alpha3, self.beta3, self.fit_rt3_info_ = _optimize_benter_weights(lpm, lmk, merged["target"].to_numpy(dtype=np.float64), starts, counts, self.eps)
        return self

    def predict_rt3_direct(self, rt3_pred_df: pd.DataFrame, market_rt3_df: pd.DataFrame) -> pd.DataFrame:
        out_cols = K + ["bet_type", "combo", "combo_prob"]
        merged = self._merge_rt3_inputs(rt3_pred_df, market_rt3_df)
        if merged.empty:
            return pd.DataFrame(columns=out_cols)
        lpm, lmk = self._logs(merged)
        merged["combo_prob"] = _softmax_col(merged, self.alpha3 * lpm + self.beta3 * lmk)
        merged["bet_type"] = "rt3"
        return merged[out_cols].reset_index(drop=True)

    # ---- 3連単 (b) ----
    def predict_rt3_conditional(self, v1_model: Any, df: pd.DataFrame, market_win_df: pd.DataFrame,
                                win_pred_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """合成1着確率 × v1 の M2・M3 の条件付き確率。df は欠車を除いた出走行。"""
        out_cols = K + ["bet_type", "combo", "combo_prob"]
        if df is None or df.empty or market_win_df is None or market_win_df.empty:
            return pd.DataFrame(columns=out_cols)
        z1, z2, z3 = _extract_stage_scaled_scores(v1_model, df)
        stage_df = df[K + ["car_no"]].copy()
        stage_df["car_no"] = pd.to_numeric(stage_df["car_no"], errors="coerce").astype(int)
        stage_df["_z1"], stage_df["_z2"], stage_df["_z3"] = z1, z2, z3
        if win_pred_df is None:
            win_pred_df = stage_df[K + ["car_no"]].copy()
            win_pred_df["pred_prob"] = _softmax_col(stage_df, z1)
        bw = self.predict_win(win_pred_df, market_win_df)
        if bw.empty:
            return pd.DataFrame(columns=out_cols)
        merged = bw.merge(stage_df[K + ["car_no", "_z2", "_z3"]], on=K + ["car_no"], how="inner")
        if merged.empty:
            return pd.DataFrame(columns=out_cols)
        merged = merged.sort_values(K + ["car_no"]).reset_index(drop=True)
        merged = merged.loc[merged.groupby(K, sort=False)["pred_prob"].transform("sum") > 0].copy()
        merged["pred_prob"] = _renorm(merged, "pred_prob")
        merged["_n_cars"] = merged.groupby(K, sort=False)["car_no"].transform("count")
        merged = merged.loc[merged["_n_cars"] >= 3].copy()
        if merged.empty:
            return pd.DataFrame(columns=out_cols)

        frames: list[pd.DataFrame] = []
        for n_cars, grp in merged.groupby("_n_cars", sort=False):  # 出走台数ごとにまとめてベクトル計算
            n = int(n_cars)
            r = len(grp) // n
            ii, jj, kk = _get_perm_indices(n)
            p1 = grp["pred_prob"].to_numpy(dtype=np.float64).reshape(r, n)
            z2m = grp["_z2"].to_numpy(dtype=np.float64).reshape(r, n)
            z3m = grp["_z3"].to_numpy(dtype=np.float64).reshape(r, n)
            cars = grp["car_no"].to_numpy(dtype=np.int64).reshape(r, n)
            w2 = np.exp(z2m - z2m.max(axis=1, keepdims=True))
            w3 = np.exp(z3m - z3m.max(axis=1, keepdims=True))
            d2 = np.maximum(w2.sum(axis=1, keepdims=True) - w2[:, ii], self.eps)
            d3 = np.maximum(w3.sum(axis=1, keepdims=True) - w3[:, ii] - w3[:, jj], self.eps)
            p = p1[:, ii] * (w2[:, jj] / d2) * (w3[:, kk] / d3)
            p = p / np.maximum(p.sum(axis=1, keepdims=True), self.eps)
            if np.all(cars == cars[0:1, :]):
                combo_arr = np.tile(_get_combo_strings(tuple(int(x) for x in cars[0])), r)
            else:
                combo_arr = np.concatenate([_get_combo_strings(tuple(int(x) for x in row)) for row in cars])
            head = grp.iloc[::n][K].reset_index(drop=True)
            chunk = pd.DataFrame({c: np.repeat(head[c].to_numpy(), len(ii)) for c in K})
            chunk["bet_type"] = "rt3"
            chunk["combo"] = combo_arr
            chunk["combo_prob"] = p.reshape(-1)
            frames.append(chunk)
        return pd.concat(frames, ignore_index=True).sort_values(K + ["combo"]).reset_index(drop=True)[out_cols]

    def blended_rt3(self, method: str = "a", *, rt3_pred_df=None, market_rt3_df=None, v1_model=None, df=None,
                    market_win_df=None, win_pred_df=None) -> pd.DataFrame:
        m = method.lower().strip()
        if m in ("a", "direct"):
            if rt3_pred_df is None:
                if v1_model is None or df is None:
                    raise ValueError("method='a' requires rt3_pred_df (or v1_model and df).")
                rt3_pred_df = v1_model.predict_rt3(df)
            if market_rt3_df is None:
                raise ValueError("method='a' requires market_rt3_df.")
            return self.predict_rt3_direct(rt3_pred_df, market_rt3_df)
        if m in ("b", "conditional"):
            if market_win_df is None or v1_model is None or df is None:
                raise ValueError("method='b' requires v1_model, df and market_win_df.")
            return self.predict_rt3_conditional(v1_model, df, market_win_df, win_pred_df)
        raise ValueError(f"Unknown blended_rt3 method: {method!r}. Expected 'a' or 'b'.")

    def to_dict(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta, "alpha3": self.alpha3, "beta3": self.beta3, "eps": self.eps,
                "fit_info": self.fit_info_, "fit_rt3_info": self.fit_rt3_info_}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BenterBlend":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(alpha=float(d.get("alpha", 0.5)), beta=float(d.get("beta", 0.5)), alpha3=float(d.get("alpha3", 0.5)),
                  beta3=float(d.get("beta3", 0.5)), eps=float(d.get("eps", 1e-15)))
        obj.fit_info_ = dict(d.get("fit_info", {}))
        obj.fit_rt3_info_ = dict(d.get("fit_rt3_info", {}))
        return obj


def blended_rt3(blend: BenterBlend, method: str = "a", **kwargs: Any) -> pd.DataFrame:
    return blend.blended_rt3(method=method, **kwargs)

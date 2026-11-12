# autorace/model/combos.py（Gemini作成、Claude確認：Harvilleの式は P(i)·P(j)/(1-P(i))·P(k)/(1-P(i)-P(j)) で正しい）
from __future__ import annotations

import numpy as np
import pandas as pd

from autorace.model.features import RACE_KEYS


def compute_harville_rt3_for_race(car_nos: np.ndarray, probs: np.ndarray) -> list[tuple[str, float]]:
    """1レース分の車番と1着確率から、Harville方式で3連単の全順列と確率を返す（確率>0が3人未満なら空）"""
    valid_mask = probs > 0.0
    c_valid = car_nos[valid_mask]
    p_valid = probs[valid_mask]
    n = len(c_valid)
    if n < 3:
        return []
    p_valid = p_valid / p_valid.sum()

    p_i = p_valid[:, None, None]
    p_j = p_valid[None, :, None]
    p_k = p_valid[None, None, :]
    prob_3d = (p_i * p_j * p_k) / (np.maximum(1.0 - p_i, 1e-12) * np.maximum(1.0 - p_i - p_j, 1e-12))

    idx = np.arange(n)
    mask_3d = ((idx[:, None, None] != idx[None, :, None]) & (idx[:, None, None] != idx[None, None, :])
               & (idx[None, :, None] != idx[None, None, :]))
    i_idx, j_idx, k_idx = np.where(mask_3d)
    combo_probs = prob_3d[i_idx, j_idx, k_idx]
    total_p = combo_probs.sum()
    if total_p > 0:
        combo_probs = combo_probs / total_p
    return [(f"{int(ci)}-{int(cj)}-{int(ck)}", float(cp))
            for ci, cj, ck, cp in zip(c_valid[i_idx], c_valid[j_idx], c_valid[k_idx], combo_probs)]


def generate_rt3_combos(pred_df: pd.DataFrame) -> pd.DataFrame:
    """pred_prob を持つ出走表から、全レースの3連単組み合わせ確率を作る"""
    cols = RACE_KEYS + ["bet_type", "combo", "combo_prob"]
    if pred_df.empty:
        return pd.DataFrame(columns=cols)
    records = []
    for (place_code, race_date, race_no), grp in pred_df.groupby(RACE_KEYS, sort=False):
        g = grp.sort_values("car_no")
        for combo_str, cp in compute_harville_rt3_for_race(g["car_no"].to_numpy(dtype=int), g["pred_prob"].to_numpy(dtype=float)):
            records.append({"place_code": place_code, "race_date": race_date, "race_no": race_no,
                            "bet_type": "rt3", "combo": combo_str, "combo_prob": cp})
    return pd.DataFrame.from_records(records, columns=cols)

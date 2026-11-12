"""オッズ・払戻のパース（Gemini作成、Claude確認）。

オッズは実際の応答の形（2026-11-12 確認）に合わせている。
払戻は形が未確認のため「推測パーサ」。取り出せなかった分は raw として返し、呼び出し側で保存する。
ログには件数だけを出し、オッズ値・買い目・金額は出さない。
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from autorace.common.schema import OddsEntry, OddsPlayerEntry, RefundEntry

logger = logging.getLogger(__name__)

# bet_type: (APIのキー, 段数, 連複か, 幅のあるオッズか)
BET_TYPE_KEYS = {
    "tns": ("tnsOddsList", 1, False, False),
    "fns": ("fnsOddsList", 1, False, True),
    "rtw": ("rtwOddsList", 2, False, False),
    "rfw": ("rfwOddsList", 2, True, False),
    "rt3": ("rt3OddsList", 3, False, False),
    "rf3": ("rf3OddsList", 3, True, False),
    "wid": ("widOddsList", 2, True, True),
}

REFUND_BET_TYPES = ("tns", "fns", "rtw", "rfw", "rt3", "rf3", "wid")


def _parse_float(val: Any) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val) if val > 0 else None
    if isinstance(val, str):
        s = val.strip().replace(",", "")
        if s in ("", "-", "---", "欠", "特払", "不成立"):
            return None
        try:
            f = float(s)
            return f if f > 0 else None
        except ValueError:
            return None
    return None


def _parse_int(val: Any) -> Optional[int]:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        s = val.strip().replace(",", "").replace("円", "").replace("番人気", "")
        if s in ("", "-", "---", "欠"):
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def _extract_range_leaf(node: Any) -> Tuple[Optional[float], Optional[float], bool]:
    """幅のあるオッズ(fns, wid)の末端から (最小, 最大, 読めたか) を取り出す"""
    if isinstance(node, (str, int, float)):
        return _parse_float(node), None, True
    if isinstance(node, list):
        if len(node) == 0:
            return None, None, True
        if len(node) == 1:
            return _parse_float(node[0]), None, True
        if len(node) == 2 and all(not isinstance(x, (dict, list)) for x in node):
            v1, v2 = _parse_float(node[0]), _parse_float(node[1])
            if v1 is not None and v2 is not None and v1 > v2:
                v1, v2 = v2, v1
            return v1, v2, True
        return None, None, False
    if isinstance(node, dict):
        if not node:
            return None, None, True
        if any(isinstance(v, (dict, list)) for v in node.values()):
            return None, None, False
        lower_map = {str(k).lower(): v for k, v in node.items()}
        min_v, max_v, matched = None, None, False
        for k, v in lower_map.items():
            if any(w in k for w in ("min", "low", "start", "from", "first")) or k == "1":
                min_v, matched = _parse_float(v), True
            elif any(w in k for w in ("max", "high", "end", "to", "second")) or k == "2":
                max_v, matched = _parse_float(v), True
        if matched:
            if min_v is not None and max_v is not None and min_v > max_v:
                min_v, max_v = max_v, min_v
            return min_v, max_v, True
        if len(node) <= 2:
            vals = sorted(v for v in (_parse_float(x) for x in node.values()) if v is not None)
            if len(vals) == 2:
                return vals[0], vals[1], True
            if len(vals) == 1:
                return vals[0], None, True
            return None, None, True
    return None, None, False


def _format_combo(cars: List[str], unordered: bool) -> Optional[str]:
    try:
        nums = [int(c) for c in cars]
    except ValueError:
        return None
    if any(n < 1 or n > 8 for n in nums) or len(set(nums)) != len(nums):
        return None
    if unordered:
        nums.sort()
    return "-".join(str(n) for n in nums)


def parse_odds_body(
    place_code: int, race_date: str, race_no: int, body: Dict[str, Any], snapshot: str = "final",
) -> Tuple[List[OddsEntry], List[OddsPlayerEntry], Dict[str, Any]]:
    """/race_info/Odds の body → (オッズ, 選手の追加項目, 読めなかった部分のraw)"""
    if not isinstance(body, dict):
        return [], [], {"_body": body}

    odds_entries: List[OddsEntry] = []
    unknown_raw: Dict[str, Any] = {}

    for bet_type, (api_key, depth, unordered, is_range) in BET_TYPE_KEYS.items():
        raw_node = body.get(api_key)
        if raw_node in (None, "", [], {}):
            continue
        if not isinstance(raw_node, dict):
            unknown_raw[api_key] = raw_node
            continue

        combo_map: Dict[str, OddsEntry] = {}
        unknown_count = 0

        def _traverse(curr: Any, path: List[str]) -> None:
            nonlocal unknown_count
            if len(path) == depth:
                combo = _format_combo(path, unordered)
                if not combo:
                    unknown_count += 1
                    return
                if is_range:
                    o_min, o_max, ok = _extract_range_leaf(curr)
                    if not ok:
                        unknown_count += 1
                        return
                else:
                    if isinstance(curr, (dict, list)):
                        unknown_count += 1
                        return
                    o_min, o_max = _parse_float(curr), None
                entry = OddsEntry(place_code=place_code, race_date=race_date, race_no=race_no,
                                  bet_type=bet_type, combo=combo, odds=o_min, odds_max=o_max, snapshot=snapshot)
                if combo not in combo_map or (combo_map[combo].odds is None and entry.odds is not None):
                    combo_map[combo] = entry
                return
            if not isinstance(curr, dict):
                unknown_count += 1
                return
            for k, v in curr.items():
                _traverse(v, path + [str(k)])

        _traverse(raw_node, [])
        odds_entries.extend(combo_map.values())
        if unknown_count > 0:
            logger.warning("Odds parser: %d unrecognized nodes (place=%d date=%s R%d bet=%s)",
                           unknown_count, place_code, race_date, race_no, bet_type)
            unknown_raw[api_key] = raw_node

    player_entries: List[OddsPlayerEntry] = []
    player_list = body.get("playerList")
    if isinstance(player_list, list):
        for item in player_list:
            if not isinstance(item, dict) or not item.get("playerCode"):
                continue
            data = dict(item)
            data.pop("placeCode", None)  # 所属場コード。開催場と取り違えないよう外す（レビューで追加）
            data.update(place_code=place_code, race_date=race_date, race_no=race_no)
            try:
                player_entries.append(OddsPlayerEntry.model_validate(data))
            except Exception:
                unknown_raw.setdefault("playerList_errors", []).append(item)

    return odds_entries, player_entries, unknown_raw


# ---------------- 払戻（推測パーサ） ----------------

def _extract_combo(data: Dict[str, Any], bet_type: str) -> Optional[str]:
    unordered = bet_type in ("rfw", "rf3", "wid")
    expected_len = 1 if bet_type in ("tns", "fns") else (2 if bet_type in ("rtw", "rfw", "wid") else 3)

    for k, v in data.items():
        kl = k.lower()
        if any(w in kl for w in ("combo", "kumiban", "kumi", "number", "num", "select", "focus", "target")):
            if isinstance(v, str) and v.strip():
                nums = re.findall(r"[1-8]", v)
                if len(nums) == expected_len:
                    return _format_combo(nums, unordered)
            elif isinstance(v, int) and expected_len == 1 and 1 <= v <= 8:
                return str(v)

    pos: List[Tuple[str, str]] = []
    for k, v in data.items():
        kl = k.lower()
        if any(w in kl for w in ("pay", "money", "refund", "price", "amount", "pop", "rank", "ninki", "odds")):
            continue
        if any(w in kl for w in ("car", "no", "order", "1st", "2nd", "3rd", "first", "second", "third")):
            iv = _parse_int(v)
            if iv is not None and 1 <= iv <= 8:
                pos.append((kl, str(iv)))
    if len(pos) == expected_len:
        pos.sort(key=lambda x: x[0])
        combo = _format_combo([c[1] for c in pos], unordered)
        if combo:
            return combo

    for v in data.values():
        if isinstance(v, str):
            s = v.strip()
            if expected_len == 1 and re.fullmatch(r"[1-8]", s):
                return s
            if expected_len >= 2 and re.fullmatch(r"[1-8]([-=_/][1-8]){1,2}", s):
                nums = re.findall(r"[1-8]", s)
                if len(nums) == expected_len:
                    return _format_combo(nums, unordered)
    return None


def _parse_refund_record(place_code: int, race_date: str, race_no: int, bet_type: str,
                         rec: Dict[str, Any], slot: int) -> List[RefundEntry]:
    payout_c: List[int] = []
    pop_c: List[int] = []
    for k, v in rec.items():
        kl = k.lower()
        iv = _parse_int(v)
        if iv is None:
            continue
        if any(w in kl for w in ("refund", "pay", "money", "price", "amount", "dividend")):
            if iv >= 100:
                payout_c.append(iv)
        elif any(w in kl for w in ("pop", "ninki", "rank", "fav")):
            if 1 <= iv <= 500:
                pop_c.append(iv)
    if not payout_c:
        return []
    combo = _extract_combo(rec, bet_type)
    if not combo:
        return []
    return [RefundEntry(place_code=place_code, race_date=race_date, race_no=race_no, bet_type=bet_type,
                        slot=slot, combo=combo, payout=payout_c[0], popularity=pop_c[0] if pop_c else None)]


def _parse_bet_node(place_code: int, race_date: str, race_no: int, bet_type: str, node: Any) -> Tuple[List[RefundEntry], bool]:
    if node in (None, "", [], {}):
        return [], False
    entries: List[RefundEntry] = []
    if isinstance(node, list):
        unparsed = False
        for idx, item in enumerate(node, start=1):
            if isinstance(item, dict):
                sub = _parse_refund_record(place_code, race_date, race_no, bet_type, item, idx)
                if sub:
                    entries.extend(sub)
                elif any(_parse_int(v) not in (None, 0) for v in item.values() if not isinstance(v, (dict, list))):
                    unparsed = True
            elif item not in (None, "", "-", "---"):
                unparsed = True
        return entries, unparsed
    if isinstance(node, dict):
        if any(isinstance(v, (dict, list)) for v in node.values()):
            unparsed, slot = False, 1
            for k, v in node.items():
                if isinstance(v, dict):
                    merged = dict(v)
                    if re.fullmatch(r"[1-8]([-=_/][1-8]){0,2}", str(k).strip()):
                        merged.setdefault("_key_combo", str(k).strip())
                    sub = _parse_refund_record(place_code, race_date, race_no, bet_type, merged, slot)
                    if not sub:
                        sub, u = _parse_bet_node(place_code, race_date, race_no, bet_type, v)
                        unparsed = unparsed or u or not sub
                    for e in sub:
                        e.slot = slot
                        slot += 1
                    entries.extend(sub)
                elif isinstance(v, list):
                    sub, u = _parse_bet_node(place_code, race_date, race_no, bet_type, v)
                    entries.extend(sub)
                    unparsed = unparsed or u
            return entries, unparsed
        if not any(_parse_int(v) not in (None, 0) for v in node.values()):
            return [], False
        single = _parse_refund_record(place_code, race_date, race_no, bet_type, node, 1)
        return (single, False) if single else ([], True)
    return [], True


def parse_refund_body(place_code: int, race_date: str, body: Any) -> Tuple[List[RefundEntry], List[Dict[str, Any]]]:
    """/race_info/RaceRefund の body → (払戻, 読めなかった部分のraw)"""
    raw_list = body if isinstance(body, list) else next((v for v in body.values() if isinstance(v, list)), None) if isinstance(body, dict) else None
    if raw_list is None:
        return ([], [{"place_code": place_code, "race_date": race_date, "raw": body}]) if body else ([], [])

    entries: List[RefundEntry] = []
    unparsed: List[Dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            unparsed.append({"place_code": place_code, "race_date": race_date, "raw": item})
            continue
        race_no = _parse_int(item.get("raceNo"))
        info = item.get("refundInfo")
        if not race_no or not isinstance(info, dict):
            if info:
                unparsed.append({"place_code": place_code, "race_date": race_date, "raw": item})
            continue
        bad = False
        for bt in REFUND_BET_TYPES:
            if bt in info:
                sub, u = _parse_bet_node(place_code, race_date, race_no, bt, info[bt])
                entries.extend(sub)
                bad = bad or u
        if bad:
            unparsed.append({"place_code": place_code, "race_date": race_date, "race_no": race_no, "raw": item})
    if unparsed:
        logger.warning("Refund parser: %d race blocks kept as raw (place=%d date=%s)", len(unparsed), place_code, race_date)
    return entries, unparsed

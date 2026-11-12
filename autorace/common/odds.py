import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from autorace.common.schema import OddsEntry, OddsPlayerEntry, RefundEntry

logger = logging.getLogger(__name__)

# bet_type: (APIキー, 組み合わせの段数, 連複(順不同)か, 幅のあるオッズか)
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
    """幅のあるオッズ(fns, wid)の末端ノードから (min_odds, max_odds, is_recognized) を抽出する"""
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
        min_v, max_v = None, None
        matched_keys = False
        for k, v in lower_map.items():
            if any(w in k for w in ("min", "low", "start", "from", "1", "first")):
                min_v = _parse_float(v)
                matched_keys = True
            elif any(w in k for w in ("max", "high", "end", "to", "2", "second")):
                max_v = _parse_float(v)
                matched_keys = True

        if matched_keys:
            if min_v is not None and max_v is not None and min_v > max_v:
                min_v, max_v = max_v, min_v
            return min_v, max_v, True

        if len(node) <= 2:
            vals = [_parse_float(v) for v in node.values()]
            non_null = sorted(v for v in vals if v is not None)
            if len(non_null) == 2:
                return non_null[0], non_null[1], True
            if len(non_null) == 1:
                return non_null[0], (non_null[0] if len(vals) == 2 and vals[0] == vals[1] else None), True
            return None, None, True

    return None, None, False


def _format_combo(cars: List[str], unordered: bool) -> Optional[str]:
    try:
        nums = [int(c) for c in cars]
    except ValueError:
        return None
    if any(n < 1 or n > 8 for n in nums):
        return None
    if len(set(nums)) != len(nums):
        return None
    if unordered:
        nums.sort()
    return "-".join(str(n) for n in nums)


def parse_odds_body(
    place_code: int,
    race_date: str,
    race_no: int,
    body: Dict[str, Any],
    snapshot: str = "final",
) -> Tuple[List[OddsEntry], List[OddsPlayerEntry], Dict[str, Any]]:
    """
    /race_info/Odds の body をパースし、(OddsEntryリスト, OddsPlayerEntryリスト, 未知構造のraw辞書) を返す。
    ログには件数のみ出力し、具体的なオッズ値や買い目は出力しない。
    """
    if not isinstance(body, dict):
        return [], [], {"_body": body}

    odds_entries: List[OddsEntry] = []
    unknown_raw: Dict[str, Any] = {}

    for bet_type, (api_key, depth, unordered, is_range) in BET_TYPE_KEYS.items():
        raw_node = body.get(api_key)
        if raw_node is None or raw_node == "" or raw_node == [] or raw_node == {}:
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
                    o_min = _parse_float(curr)
                    o_max = None

                entry = OddsEntry(
                    place_code=place_code,
                    race_date=race_date,
                    race_no=race_no,
                    bet_type=bet_type,
                    combo=combo,
                    odds=o_min,
                    odds_max=o_max,
                    snapshot=snapshot,
                )
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
            logger.warning(
                "Odds parser encountered %d unrecognized nodes (place=%d date=%s R%d bet=%s)",
                unknown_count, place_code, race_date, race_no, bet_type,
            )
            unknown_raw[api_key] = raw_node

    player_entries: List[OddsPlayerEntry] = []
    player_list = body.get("playerList")
    if isinstance(player_list, list):
        for item in player_list:
            if not isinstance(item, dict) or not item.get("playerCode"):
                continue
            data = dict(item)
            data["place_code"] = place_code
            data["race_date"] = race_date
            data["race_no"] = race_no
            try:
                player_entries.append(OddsPlayerEntry.model_validate(data))
            except Exception:
                unknown_raw.setdefault("playerList_errors", []).append(item)

    return odds_entries, player_entries, unknown_raw


def _extract_combo_from_dict_or_str(data: Dict[str, Any], bet_type: str) -> Optional[str]:
    unordered = bet_type in ("rfw", "rf3", "wid")
    expected_len = 1 if bet_type in ("tns", "fns") else (2 if bet_type in ("rtw", "rfw", "wid") else 3)

    # 1) 明示的な組番キー
    for k, v in data.items():
        kl = k.lower()
        if any(w in kl for w in ("combo", "kumiban", "kumi", "number", "num", "select", "focus", "target")):
            if isinstance(v, str) and v.strip():
                nums = re.findall(r"[1-8]", v)
                if len(nums) == expected_len:
                    return _format_combo(nums, unordered)
            elif isinstance(v, int) and expected_len == 1 and 1 <= v <= 8:
                return str(v)

    # 2) carNo1, car1, first, second, third などの分割キー
    pos_candidates: List[Tuple[str, str]] = []
    for k, v in data.items():
        kl = k.lower()
        if any(w in kl for w in ("pay", "money", "refund", "price", "amount", "pop", "rank", "ninki", "odds")):
            continue
        if any(w in kl for w in ("car", "no", "order", "1st", "2nd", "3rd", "first", "second", "third", "uma", "waku")):
            iv = _parse_int(v)
            if iv is not None and 1 <= iv <= 8:
                pos_candidates.append((kl, str(iv)))

    if len(pos_candidates) == expected_len:
        pos_candidates.sort(key=lambda x: x[0])
        combo = _format_combo([c[1] for c in pos_candidates], unordered)
        if combo:
            return combo

    # 3) 文字列値の中で "1-2-3" や "1=2" の形式にマッチするもの
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


def _parse_single_refund_dict(
    place_code: int,
    race_date: str,
    race_no: int,
    bet_type: str,
    flat_dict: Dict[str, Any],
    default_slot: int,
) -> List[RefundEntry]:
    """平坦化された辞書から1件のRefundEntryを推測抽出する"""
    payout_candidates: List[Tuple[str, int]] = []
    pop_candidates: List[Tuple[str, int]] = []

    for k, v in flat_dict.items():
        kl = k.lower()
        iv = _parse_int(v)
        if iv is None:
            continue
        if any(w in kl for w in ("refund", "pay", "money", "price", "amount", "kin", "dividend")):
            if iv >= 100:
                payout_candidates.append((k, iv))
        elif any(w in kl for w in ("pop", "ninki", "rank", "fav")):
            if 1 <= iv <= 500:
                pop_candidates.append((k, iv))

    # キー名から判別できない場合、値の範囲（100円以上かつ10円単位）から払戻金を推測
    if not payout_candidates:
        for k, v in flat_dict.items():
            kl = k.lower()
            if any(w in kl for w in ("car", "no", "combo", "kumi", "pop", "ninki", "rank")):
                continue
            iv = _parse_int(v)
            if iv is not None and iv >= 100 and iv % 10 == 0:
                payout_candidates.append((k, iv))

    if not payout_candidates:
        return []

    combo = _extract_combo_from_dict_or_str(flat_dict, bet_type)
    if not combo:
        return []

    payout = payout_candidates[0][1]
    popularity = pop_candidates[0][1] if pop_candidates else None

    return [
        RefundEntry(
            place_code=place_code,
            race_date=race_date,
            race_no=race_no,
            bet_type=bet_type,
            slot=default_slot,
            combo=combo,
            payout=payout,
            popularity=popularity,
        )
    ]


def _parse_bet_refund_node(
    place_code: int,
    race_date: str,
    race_no: int,
    bet_type: str,
    node: Any,
) -> Tuple[List[RefundEntry], bool]:
    """賭式ごとのノードを再帰的に走査し、(抽出されたRefundEntryリスト, 未知・未抽出があったか) を返す"""
    if node is None or node == "" or node == [] or node == {}:
        return [], False

    entries: List[RefundEntry] = []

    # パターンA: リスト [{"combo": "1-2", "money": "1230", "pop": "4"}, ...]
    if isinstance(node, list):
        unparsed = False
        for idx, item in enumerate(node, start=1):
            if not isinstance(item, dict):
                if _parse_int(item) is not None or (isinstance(item, str) and item.strip() not in ("", "-", "---")):
                    unparsed = True
                continue
            sub_entries = _parse_single_refund_dict(place_code, race_date, race_no, bet_type, item, idx)
            if sub_entries:
                entries.extend(sub_entries)
            else:
                has_meaningful = any(
                    _parse_int(v) not in (None, 0) or (isinstance(v, str) and v.strip() not in ("", "-", "---", "0"))
                    for v in item.values()
                )
                if has_meaningful:
                    unparsed = True
        return entries, unparsed

    if isinstance(node, dict):
        # パターンB: ネスト辞書 {"1": {"combo": ...}, ...} または組番キー {"1-2": {"money": 1200, "pop": 3}}
        child_dicts = [v for v in node.values() if isinstance(v, (dict, list))]
        if child_dicts:
            unparsed = False
            slot_idx = 1
            for k, v in node.items():
                if v is None or v == "" or v == [] or v == {}:
                    continue
                if isinstance(v, dict):
                    merged = dict(v)
                    if re.fullmatch(r"[1-8]([-=_/][1-8]){0,2}", str(k).strip()):
                        merged.setdefault("_key_combo", str(k).strip())
                    sub = _parse_single_refund_dict(place_code, race_date, race_no, bet_type, merged, slot_idx)
                    if sub:
                        entries.extend(sub)
                        slot_idx += 1
                    else:
                        sub_rec, sub_unp = _parse_bet_refund_node(place_code, race_date, race_no, bet_type, v)
                        if sub_rec:
                            for e in sub_rec:
                                e.slot = slot_idx
                                slot_idx += 1
                            entries.extend(sub_rec)
                        if sub_unp or not sub_rec:
                            has_meaningful = any(
                                _parse_int(val) not in (None, 0)
                                for val in v.values()
                                if not isinstance(val, (dict, list))
                            )
                            if has_meaningful:
                                unparsed = True
                elif isinstance(v, list):
                    sub_rec, sub_unp = _parse_bet_refund_node(place_code, race_date, race_no, bet_type, v)
                    entries.extend(sub_rec)
                    if sub_unp:
                        unparsed = True
            return entries, unparsed

        # パターンC: フラット辞書（単一または番号サフィックス付きキー）
        has_meaningful = any(
            _parse_int(v) not in (None, 0) or (isinstance(v, str) and v.strip() not in ("", "-", "---", "0"))
            for v in node.values()
        )
        if not has_meaningful:
            return [], False

        grouped: Dict[int, Dict[str, Any]] = {}
        for k, v in node.items():
            m = re.search(r"(?:_|^|[a-zA-Z])([1-9])$", str(k))
            if m and not re.fullmatch(r"car(?:no)?[1-3]", str(k).lower()):
                grouped.setdefault(int(m.group(1)), {})[k] = v

        if len(grouped) >= 2:
            for idx in sorted(grouped.keys()):
                entries.extend(_parse_single_refund_dict(place_code, race_date, race_no, bet_type, grouped[idx], idx))
            if entries:
                return entries, False

        single = _parse_single_refund_dict(place_code, race_date, race_no, bet_type, node, 1)
        if single:
            return single, False

        return [], True

    return [], True


def parse_refund_body(
    place_code: int,
    race_date: str,
    body: Any,
) -> Tuple[List[RefundEntry], List[Dict[str, Any]]]:
    """
    /race_info/RaceRefund の body をパースし、(RefundEntryリスト, 未知・未抽出箇所のrawリスト) を返す。
    ※払戻の形は未確認のため推測パーサ。生JSONは常に別途保存し、形が分かったら確定版に置き換える。
    """
    raw_list: Optional[List[Any]] = None
    if isinstance(body, list):
        raw_list = body
    elif isinstance(body, dict):
        for v in body.values():
            if isinstance(v, list):
                raw_list = v
                break

    if raw_list is None:
        if body:
            logger.warning("Refund body has unexpected root type (place=%d date=%s)", place_code, race_date)
            return [], [{"place_code": place_code, "race_date": race_date, "raw": body}]
        return [], []

    entries: List[RefundEntry] = []
    unparsed_raws: List[Dict[str, Any]] = []

    for item in raw_list:
        if not isinstance(item, dict):
            unparsed_raws.append({"place_code": place_code, "race_date": race_date, "raw": item})
            continue

        race_no = _parse_int(item.get("raceNo"))
        refund_info = item.get("refundInfo")
        if not race_no or not isinstance(refund_info, dict):
            if refund_info:
                unparsed_raws.append({"place_code": place_code, "race_date": race_date, "raw": item})
            continue

        race_unparsed = False
        for bet_type in REFUND_BET_TYPES:
            if bet_type not in refund_info:
                continue
            bet_entries, has_unparsed = _parse_bet_refund_node(
                place_code, race_date, race_no, bet_type, refund_info[bet_type]
            )
            entries.extend(bet_entries)
            if has_unparsed:
                race_unparsed = True

        if race_unparsed:
            unparsed_raws.append({"place_code": place_code, "race_date": race_date, "race_no": race_no, "raw": item})

    if unparsed_raws:
        logger.warning(
            "Refund parser saved %d unparsed race blocks to raw (place=%d date=%s)",
            len(unparsed_raws), place_code, race_date,
        )

    return entries, unparsed_raws

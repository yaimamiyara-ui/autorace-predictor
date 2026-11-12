"""試走発表後のオッズを記録し続ける（Gemini作成、Claude確認・修正）。

修正点（Claudeのレビュー）：場コードを文字ではなく数字で扱う（ほかのデータと結合できるように）。
1回の実行はすぐ終わる。GitHub Actionsから10分おきに起動する前提。ログには件数だけを出す。
"""
from __future__ import annotations

import argparse
import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from autorace.common.odds import parse_odds_body
from autorace.common.schema import ProgramEntry, RaceInfoEntry
from autorace.fetch.client import AutoraceAPIError, AutoraceClient

logger = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MAX_REQUESTS = 30
MAX_RACE_NO = 12
LOOKAHEAD_RACES = 2

ODDS_DEDUP_KEYS = ["place_code", "race_date", "race_no", "bet_type", "combo"]
PROGRAM_DEDUP_KEYS = ["place_code", "race_date", "race_no", "car_no"]
META_DEDUP_KEYS = ["place_code", "race_date", "race_no"]


def _parse_optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _is_absent(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "none", "null")
    if isinstance(value, float) and math.isnan(value):
        return False
    return bool(value)


def _has_trial_run_time(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(f) and f > 0


def is_trial_announced(entries: list[ProgramEntry]) -> bool:
    """欠場以外の全車に試走タイムが入っていれば発表済み"""
    active = [e for e in entries if not _is_absent(e.absent)]
    return bool(active) and all(_has_trial_run_time(e.trial_run_time) for e in active)


def resolve_race_date(now_jst: datetime) -> str:
    """開催日。ミッドナイト開催で日付をまたいでも開催日で記録する（0:00〜4:59 は前日扱い）"""
    d = now_jst.date() - timedelta(days=1) if now_jst.hour < 5 else now_jst.date()
    return d.isoformat()


def compute_minutes_to_close(telvote_time: Any, race_date: str, now_jst: datetime) -> float | None:
    if telvote_time is None or ":" not in str(telvote_time):
        return None
    try:
        hh, mm = (int(x) for x in str(telvote_time).strip().split(":")[:2])
        d = date.fromisoformat(race_date)
    except ValueError:
        return None
    base = datetime(d.year, d.month, d.day, tzinfo=JST)
    if 0 <= hh < 5:  # 翌日0時台〜4時台の締切（ミッドナイト）
        hh += 24
    close_dt = base + timedelta(hours=hh, minutes=mm)
    return round((close_dt - now_jst).total_seconds() / 60.0, 2)


def _parquet_path(data_dir: Path, category: str, race_date: str) -> Path:
    return data_dir / category / race_date[:4] / f"{race_date[:7]}.parquet"


def _load_recorded_races(odds_path: Path) -> set[tuple[int, str, int]]:
    if not odds_path.exists():
        return set()
    df = pd.read_parquet(odds_path, columns=["place_code", "race_date", "race_no"])
    return {(int(p), str(d), int(r)) for p, d, r in zip(df["place_code"], df["race_date"], df["race_no"])}


def _append_dedup_parquet(path: Path, new_rows: list[dict[str, Any]], dedup_keys: list[str]) -> None:
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([pd.read_parquet(path), new_df], ignore_index=True) if path.exists() else new_df
    combined = combined.drop_duplicates(subset=[k for k in dedup_keys if k in combined.columns], keep="first")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(path)


def run_once(client: AutoraceClient | None = None, data_dir: Path | str = DEFAULT_DATA_DIR,
             now: datetime | None = None, max_requests: int = DEFAULT_MAX_REQUESTS) -> dict[str, int]:
    api = client if client is not None else AutoraceClient()
    base_dir = Path(data_dir)
    clock = (lambda: now.astimezone(JST) if now.tzinfo else now.replace(tzinfo=JST)) if now else (lambda: datetime.now(JST))
    stats = {"requests": 0, "venues_processed": 0, "recorded_races": 0, "odds_rows": 0, "program_rows": 0, "errors": 0}

    try:
        hold_resp = api.hold_today()
        stats["requests"] += 1
    except Exception as exc:
        stats["errors"] += 1
        logger.error("hold_today failed (%s)", type(exc).__name__)
        return stats

    venues = (hold_resp.get("body") or {}).get("today") or []
    recorded_cache: dict[Path, set[tuple[int, str, int]]] = {}

    for venue in venues:
        place_code = _parse_optional_int(venue.get("placeCode"))
        if place_code is None:
            continue
        try:
            if str(venue.get("cancelFlg", "")).strip() == "1":
                logger.info("place=%d skipped (cancelFlg=1)", place_code)
                continue
            race_date = resolve_race_date(clock())
            odds_path = _parquet_path(base_dir, "odds_pre", race_date)
            if odds_path not in recorded_cache:
                recorded_cache[odds_path] = _load_recorded_races(odds_path)
            recorded = recorded_cache[odds_path]

            result_no = max(_parse_optional_int(venue.get("resultRaceNo")) or 0, 0)
            now_no = _parse_optional_int(venue.get("nowRaceNo"))
            start = now_no if (now_no and now_no >= 1) else result_no + 1
            end = min(start + LOOKAHEAD_RACES, MAX_RACE_NO)

            odds_rows, prog_rows, meta_rows = [], [], []
            n_recorded = n_unannounced = 0
            for race_no in range(start, end + 1):
                if race_no <= result_no or (place_code, race_date, race_no) in recorded:
                    continue
                if stats["requests"] + 2 > max_requests:
                    logger.warning("request limit reached (%d)", stats["requests"])
                    break
                try:
                    prog_resp = api.program(place_code, race_date, race_no)
                    stats["requests"] += 1
                    entries = ProgramEntry.from_api_response(place_code, race_date, race_no, prog_resp.get("body") or {})
                    if not is_trial_announced(entries):
                        n_unannounced += 1
                        continue
                    odds_resp = api.odds(place_code, race_date, race_no)
                    stats["requests"] += 1
                    odds_entries, _, _ = parse_odds_body(place_code, race_date, race_no, odds_resp.get("body") or {}, snapshot="pre")
                    if not odds_entries:
                        logger.warning("place=%d R%d: 0 odds parsed", place_code, race_no)
                        continue
                    info = {}
                    if stats["requests"] + 1 <= max_requests:
                        try:
                            info_resp = api.other_race_info(place_code, race_date, race_no)
                            stats["requests"] += 1
                            ie = RaceInfoEntry.from_api_response(place_code, race_date, race_no, info_resp.get("body"))
                            if ie is not None:
                                # 試走ごろの距離・走路状態・気温など（race_* はまだ空のことが多い）
                                info = {f"ri_{k}": v for k, v in ie.model_dump().items()
                                        if k not in ("place_code", "race_date", "race_no")}
                        except AutoraceAPIError:
                            pass
                    t = clock()
                    fetched_at = t.isoformat(timespec="seconds")
                    is_now = now_no is not None and race_no == now_no
                    mtc = compute_minutes_to_close(venue.get("telvoteTime"), race_date, t) if is_now else None
                    for oe in odds_entries:
                        odds_rows.append({**oe.model_dump(), "fetched_at": fetched_at, "minutes_to_close": mtc})
                    for pe in entries:
                        prog_rows.append({**pe.model_dump(), "fetched_at": fetched_at, "minutes_to_close": mtc})
                    meta_rows.append({
                        "place_code": place_code, "place_key": venue.get("placeKey"), "race_date": race_date,
                        "race_no": race_no, "now_race_no": now_no, "result_race_no": result_no,
                        "race_start_time": venue.get("raceStartTime") if is_now else None,
                        "telvote_time": venue.get("telvoteTime") if is_now else None,
                        "period_start_date": venue.get("periodStartDate"), "roadtemp": venue.get("roadtemp"),
                        "weather": venue.get("weather"), "distance": venue.get("distance"),
                        "paragraphday": venue.get("paragraphday"), "nighter_code": venue.get("nighterCode"),
                        "fetched_at": fetched_at, "minutes_to_close": mtc, **info,
                    })
                    recorded.add((place_code, race_date, race_no))
                    n_recorded += 1
                except AutoraceAPIError:
                    # そのレースのデータが無い（開催終了・中止など）。エラーではないので静かに飛ばす
                    logger.info("place=%d R%d: no data", place_code, race_no)
                except Exception as exc:
                    stats["errors"] += 1
                    logger.error("place=%d R%d error (%s)", place_code, race_no, type(exc).__name__)

            if odds_rows:
                # 記録済み判定に使う odds_pre は最後に書く（途中で止まっても不整合にならない）
                _append_dedup_parquet(_parquet_path(base_dir, "program_pre", race_date), prog_rows, PROGRAM_DEDUP_KEYS)
                _append_dedup_parquet(_parquet_path(base_dir, "live_meta", race_date), meta_rows, META_DEDUP_KEYS)
                _append_dedup_parquet(odds_path, odds_rows, ODDS_DEDUP_KEYS)
            stats["venues_processed"] += 1
            stats["recorded_races"] += n_recorded
            stats["odds_rows"] += len(odds_rows)
            stats["program_rows"] += len(prog_rows)
            logger.info("place=%d date=%s recorded=%d unannounced=%d odds_rows=%d",
                        place_code, race_date, n_recorded, n_unannounced, len(odds_rows))
        except Exception as exc:
            stats["errors"] += 1
            logger.error("place=%s venue failed (%s)", place_code, type(exc).__name__)

    logger.info("done %s", stats)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Record pre-race Autorace odds after trial run.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_once(data_dir=a.data_dir, max_requests=a.max_requests)


if __name__ == "__main__":
    main()

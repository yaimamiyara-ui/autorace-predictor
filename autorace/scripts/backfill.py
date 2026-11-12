"""過去データの一括取得（Gemini作成、Claude確認・修正）。

修正点（Claudeのレビュー）：
- typing の Any が抜けていて起動時にエラーになるのを修正
- 場ごとにParquetへ書き込んでから完了印を付ける（強制終了時にデータだけ失うのを防ぐ）
"""
import argparse
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from autorace.common.odds import parse_odds_body, parse_refund_body
from autorace.common.schema import ProgramEntry, ResultEntry
from autorace.fetch.client import AutoraceAPIError, AutoraceClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_KINDS = ("program", "result", "odds", "refund")

DEDUP_KEYS_MAP = {
    "program": ["place_code", "race_date", "race_no", "player_code"],
    "race_result": ["place_code", "race_date", "race_no", "player_code"],
    "odds": ["place_code", "race_date", "race_no", "bet_type", "combo", "snapshot"],
    "odds_player": ["place_code", "race_date", "race_no", "player_code"],
    "refund": ["place_code", "race_date", "race_no", "bet_type", "slot", "combo"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Autorace data.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--places", nargs="+", default=["2", "3", "4", "5", "6"],
                        help="Place codes (space or comma separated)")
    parser.add_argument("--kinds", default="program,result,odds,refund",
                        help="Comma-separated kinds: program,result,odds,refund")
    parser.add_argument("--snapshot", default="final", help="Snapshot label for odds")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--save-raw", action="store_true", help="Save all raw JSON under data/raw/")
    parser.add_argument("--interval", type=float, default=3.0, help="Minimum interval between requests (sec)")
    parser.add_argument("--max-minutes", type=float, default=None, help="Stop gracefully after this many minutes")
    return parser.parse_args()


def parse_places(raw_places: List[str]) -> List[int]:
    return [int(p.strip()) for token in raw_places for p in token.split(",") if p.strip()]


def parse_kinds(raw_kinds: str) -> Set[str]:
    kinds = {k.strip().lower() for k in raw_kinds.split(",") if k.strip()}
    invalid = kinds - set(VALID_KINDS)
    if invalid:
        raise ValueError(f"Invalid --kinds: {invalid}. Allowed: {VALID_KINDS}")
    return kinds


def date_range(start_str: str, end_str: str) -> List[datetime.date]:
    start = datetime.date.fromisoformat(start_str)
    end = datetime.date.fromisoformat(end_str)
    if start > end:
        raise ValueError("--start must be on or before --end")
    return [start + datetime.timedelta(days=i) for i in range((end - start).days + 1)]


def get_parquet_path(base_dir: Path, folder: str, target_date: datetime.date) -> Path:
    return base_dir / folder / target_date.strftime("%Y") / f"{target_date.strftime('%Y-%m')}.parquet"


def get_done_path(base_dir: Path, target_date: datetime.date) -> Path:
    return base_dir / "_done" / target_date.strftime("%Y") / f"{target_date.strftime('%Y-%m')}.txt"


def load_done_markers(done_path: Path) -> Set[str]:
    if not done_path.exists():
        return set()
    with done_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(done_path: Path, cache: Set[str], date_str: str, place_code: int, kinds: Set[str]) -> None:
    done_path.parent.mkdir(parents=True, exist_ok=True)
    new_lines = []
    for k in sorted(kinds):
        marker = f"{date_str},{place_code},{k}"
        if marker not in cache:
            cache.add(marker)
            new_lines.append(marker)
    if new_lines:
        with done_path.open("a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")


def append_to_parquet(parquet_path: Path, records: List[dict], dedup_keys: List[str]) -> None:
    if not records:
        return
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(records)
    if parquet_path.exists():
        combined = pd.concat([pd.read_parquet(parquet_path), new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=dedup_keys, keep="last").sort_values(by=dedup_keys)
    combined.to_parquet(parquet_path, index=False)


def save_raw_json(base_dir: Path, kind: str, race_date: str, place_code: int, race_no: Optional[int], data: Any) -> None:
    suffix = f"r{race_no:02d}" if race_no is not None else "daily"
    out_path = base_dir / "raw" / kind / race_date[:4] / race_date / f"p{place_code}_{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def fetch_race_kinds(
    client: AutoraceClient, needed_kinds: Set[str], place_code: int, date_str: str, race_no: int,
    snapshot: str, data_dir: Path, save_raw: bool,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    prog_rows: List[dict] = []
    res_rows: List[dict] = []
    odds_rows: List[dict] = []
    odds_player_rows: List[dict] = []

    if "program" in needed_kinds:
        try:
            resp = client.program(place_code, date_str, race_no)
            if save_raw:
                save_raw_json(data_dir, "program", date_str, place_code, race_no, resp)
            prog_rows = [e.model_dump() for e in
                         ProgramEntry.from_api_response(place_code, date_str, race_no, resp.get("body") or {})]
        except AutoraceAPIError:
            pass

    if "result" in needed_kinds:
        try:
            resp = client.race_result(place_code, date_str, race_no)
            if save_raw:
                save_raw_json(data_dir, "race_result", date_str, place_code, race_no, resp)
            res_rows = [e.model_dump() for e in
                        ResultEntry.from_api_response(place_code, date_str, race_no, resp.get("body") or {})]
        except AutoraceAPIError:
            pass

    if "odds" in needed_kinds:
        # 出走表・結果を頼んでいて両方とも空なら、そのレースは無いとみなしてオッズは取りに行かない
        asked_any = bool(needed_kinds & {"program", "result"})
        if not asked_any or prog_rows or res_rows:
            try:
                resp = client.odds(place_code, date_str, race_no)
                if save_raw:
                    save_raw_json(data_dir, "odds", date_str, place_code, race_no, resp)
                o_entries, op_entries, unknown_raw = parse_odds_body(
                    place_code, date_str, race_no, resp.get("body") or {}, snapshot=snapshot)
                odds_rows = [e.model_dump() for e in o_entries]
                odds_player_rows = [e.model_dump() for e in op_entries]
                if unknown_raw and not save_raw:
                    save_raw_json(data_dir, "odds_unparsed", date_str, place_code, race_no, unknown_raw)
            except AutoraceAPIError:
                pass

    return prog_rows, res_rows, odds_rows, odds_player_rows


def flush(data_dir: Path, d: datetime.date, rows: Dict[str, List[dict]]) -> None:
    for folder, recs in rows.items():
        if recs:
            append_to_parquet(get_parquet_path(data_dir, folder, d), recs, DEDUP_KEYS_MAP[folder])


def main() -> None:
    start_monotonic = time.monotonic()
    args = parse_args()
    places = parse_places(args.places)
    kinds = parse_kinds(args.kinds)
    client = AutoraceClient(min_interval=args.interval)
    done_cache: Dict[Path, Set[str]] = {}

    for d in date_range(args.start, args.end):
        date_str = d.isoformat()
        done_path = get_done_path(args.data_dir, d)
        if done_path not in done_cache:
            done_cache[done_path] = load_done_markers(done_path)

        for place_code in places:
            if args.max_minutes is not None and (time.monotonic() - start_monotonic) / 60.0 >= args.max_minutes:
                logger.info("Reached --max-minutes. Stopping gracefully before %s place=%d", date_str, place_code)
                return

            needed_kinds = {k for k in kinds if f"{date_str},{place_code},{k}" not in done_cache[done_path]}
            if not needed_kinds:
                continue

            rows: Dict[str, List[dict]] = {k: [] for k in ("program", "race_result", "odds", "odds_player", "refund")}
            per_race_kinds = needed_kinds & {"program", "result", "odds"}
            race_held = False

            if per_race_kinds:
                for race_no in range(1, 13):
                    pr, rr, orows, oprows = fetch_race_kinds(
                        client, per_race_kinds, place_code, date_str, race_no, args.snapshot, args.data_dir, args.save_raw)
                    if not (pr or rr or orows or oprows):
                        if race_no == 1:
                            logger.info("No race held: date=%s place=%d", date_str, place_code)
                        break
                    race_held = True
                    rows["program"] += pr
                    rows["race_result"] += rr
                    rows["odds"] += orows
                    rows["odds_player"] += oprows

            if "refund" in needed_kinds and (race_held or not per_race_kinds):
                try:
                    resp = client.race_refund(place_code, date_str)
                    if args.save_raw:
                        save_raw_json(args.data_dir, "refund", date_str, place_code, None, resp)
                    ref_entries, unparsed = parse_refund_body(place_code, date_str, resp.get("body"))
                    rows["refund"] += [e.model_dump() for e in ref_entries]
                    if unparsed and not args.save_raw:
                        save_raw_json(args.data_dir, "refund_unparsed", date_str, place_code, None, unparsed)
                except AutoraceAPIError:
                    pass

            # 書き込んでから完了印（レビューで変更）
            flush(args.data_dir, d, rows)
            mark_done(done_path, done_cache[done_path], date_str, place_code, needed_kinds)
            if race_held:
                logger.info("Done date=%s place=%d (prog=%d res=%d odds=%d odds_p=%d refund=%d)",
                            date_str, place_code, len(rows["program"]), len(rows["race_result"]),
                            len(rows["odds"]), len(rows["odds_player"]), len(rows["refund"]))


if __name__ == "__main__":
    main()

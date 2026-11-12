import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from autorace.common.schema import ProgramEntry, ResultEntry
from autorace.fetch.client import AutoraceAPIError, AutoraceClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEDUP_KEYS = ["place_code", "race_date", "race_no", "player_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Autorace Program and RaceResult data.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--places",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6],
        help="Place codes (e.g. 2 3 4 5 6)",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Base directory for output data")
    parser.add_argument("--save-raw", action="store_true", help="Save raw JSON responses under data/raw/")
    parser.add_argument("--save-unverified", action="store_true", help="Also save raw Odds and RaceRefund JSON")
    parser.add_argument("--interval", type=float, default=3.0, help="Minimum interval between requests (sec)")
    parser.add_argument("--max-minutes", type=float, default=0, help="Stop cleanly after this many minutes (0 = no limit)")
    return parser.parse_args()


def date_range(start_str: str, end_str: str) -> List[datetime.date]:
    start = datetime.date.fromisoformat(start_str)
    end = datetime.date.fromisoformat(end_str)
    if start > end:
        raise ValueError("--start must be on or before --end")
    return [start + datetime.timedelta(days=i) for i in range((end - start).days + 1)]


def get_parquet_path(base_dir: Path, kind: str, target_date: datetime.date) -> Path:
    return base_dir / kind / target_date.strftime("%Y") / f"{target_date.strftime('%Y-%m')}.parquet"


# 取得済み（開催なしの日も含む）の (場, 日付) を月ごとに記録する。
# 開催のない日を毎回問い合わせ直さないため（レビューで追加）。
def get_done_path(base_dir: Path, target_date: datetime.date) -> Path:
    return base_dir / "_done" / target_date.strftime("%Y") / f"{target_date.strftime('%Y-%m')}.txt"


def load_done(done_path: Path) -> Set[Tuple[int, str]]:
    if not done_path.exists():
        return set()
    out: Set[Tuple[int, str]] = set()
    for line in done_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2:
            out.add((int(parts[0]), parts[1]))
    return out


def mark_done(done_path: Path, place_code: int, date_str: str, status: str) -> None:
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with done_path.open("a", encoding="utf-8") as f:
        f.write(f"{place_code},{date_str},{status}\n")


def append_to_parquet(parquet_path: Path, records: List[dict]) -> None:
    if not records:
        return
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(records)

    if parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    combined_df = combined_df.drop_duplicates(subset=DEDUP_KEYS, keep="last")
    combined_df = combined_df.sort_values(by=DEDUP_KEYS)
    combined_df.to_parquet(parquet_path, index=False)


def save_raw_json(base_dir: Path, kind: str, race_date: str, place_code: int, race_no: Optional[int], data: dict) -> None:
    suffix = f"r{race_no:02d}" if race_no is not None else "daily"
    out_path = base_dir / "raw" / kind / race_date[:4] / race_date / f"p{place_code}_{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def fetch_entries(
    client: AutoraceClient,
    kind: str,
    place_code: int,
    date_str: str,
    race_no: int,
    data_dir: Path,
    save_raw: bool,
) -> List[dict]:
    try:
        if kind == "program":
            resp = client.program(place_code, date_str, race_no)
            if save_raw:
                save_raw_json(data_dir, "program", date_str, place_code, race_no, resp)
            entries = ProgramEntry.from_api_response(place_code, date_str, race_no, resp.get("body") or {})
        else:
            resp = client.race_result(place_code, date_str, race_no)
            if save_raw:
                save_raw_json(data_dir, "race_result", date_str, place_code, race_no, resp)
            entries = ResultEntry.from_api_response(place_code, date_str, race_no, resp.get("body") or {})
        return [e.model_dump() for e in entries]
    except AutoraceAPIError:
        return []


def main() -> None:
    args = parse_args()
    dates = date_range(args.start, args.end)
    client = AutoraceClient(min_interval=args.interval)
    started = datetime.datetime.now()

    done_cache: Dict[Path, Set[Tuple[int, str]]] = {}

    for d in dates:
        if args.max_minutes and (datetime.datetime.now() - started).total_seconds() > args.max_minutes * 60:
            logger.info("Reached --max-minutes; stopping cleanly before %s", d.isoformat())
            break

        date_str = d.isoformat()
        prog_pq = get_parquet_path(args.data_dir, "program", d)
        res_pq = get_parquet_path(args.data_dir, "race_result", d)
        done_path = get_done_path(args.data_dir, d)
        if done_path not in done_cache:
            done_cache[done_path] = load_done(done_path)

        for place_code in args.places:
            key = (place_code, date_str)
            if key in done_cache[done_path]:
                continue

            place_programs: List[dict] = []
            place_results: List[dict] = []

            r1_prog = fetch_entries(client, "program", place_code, date_str, 1, args.data_dir, args.save_raw)
            r1_res = fetch_entries(client, "race_result", place_code, date_str, 1, args.data_dir, args.save_raw)

            if not r1_prog and not r1_res:
                logger.info("No race held: date=%s place=%d", date_str, place_code)
                mark_done(done_path, place_code, date_str, "empty")
                done_cache[done_path].add(key)
                continue

            place_programs.extend(r1_prog)
            place_results.extend(r1_res)

            if args.save_unverified:
                for fn, kind, rn in ((lambda: client.odds(place_code, date_str, 1), "odds", 1),
                                     (lambda: client.race_refund(place_code, date_str), "race_refund", None)):
                    try:
                        save_raw_json(args.data_dir, kind, date_str, place_code, rn, fn())
                    except AutoraceAPIError:
                        pass

            for race_no in range(2, 13):
                r_prog = fetch_entries(client, "program", place_code, date_str, race_no, args.data_dir, args.save_raw)
                r_res = fetch_entries(client, "race_result", place_code, date_str, race_no, args.data_dir, args.save_raw)
                if not r_prog and not r_res:
                    break
                place_programs.extend(r_prog)
                place_results.extend(r_res)
                if args.save_unverified:
                    try:
                        save_raw_json(args.data_dir, "odds", date_str, place_code, race_no,
                                      client.odds(place_code, date_str, race_no))
                    except AutoraceAPIError:
                        pass

            # 場ごとに書き込んでから完了印を付ける（途中で止まっても続きから再開できる）
            append_to_parquet(prog_pq, place_programs)
            append_to_parquet(res_pq, place_results)
            mark_done(done_path, place_code, date_str, "ok")
            done_cache[done_path].add(key)
            logger.info("Fetched place=%d date=%s (program=%d, result=%d rows)",
                        place_code, date_str, len(place_programs), len(place_results))


if __name__ == "__main__":
    main()

"""Historical backfill of NASA FIRMS VIIRS NOAA-20 (SP/NRT).

Run from the project root:
    python src/backfill_data.py --source VIIRS_NOAA20_SP --start 2023-09-01 --end 2023-09-05

This script reuses the existing ingestion configuration and functions from
src/ingest_data.py, and saves per-day CSV partitions without overwriting them.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from io import StringIO

import pandas as pd
import requests

from ingest_data import (  # run this script from src/ via python src/backfill_data.py
    AREA,
    BASE_URL,
    RAW_DIR,
    REQUIRED_COLUMNS,
    create_session,
    get_map_key,
)

MAX_DAYS_PER_REQUEST = 5
METADATA_DIR = RAW_DIR / "backfill_metadata"


def parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Tanggal harus YYYY-MM-DD") from exc


def date_batches(first: date, last: date):
    current = first
    while current <= last:
        batch_end = min(current + timedelta(days=MAX_DAYS_PER_REQUEST - 1), last)
        yield current, batch_end
        current = batch_end + timedelta(days=1)


def fetch_batch(session, key: str, source: str, start: date, end: date):
    num_days = (end - start).days + 1
    url = f"{BASE_URL}/{key}/{source}/{AREA}/{num_days}/{start.isoformat()}"
    response = session.get(url, timeout=(15, 90))
    response.raise_for_status()
    if not response.text.strip():
        raise ValueError("Respons API kosong; batch tidak akan disimpan")

    try:
        frame = pd.read_csv(StringIO(response.text), dtype={"acq_time": "string"})
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Respons bukan CSV yang valid") from exc

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Kolom wajib hilang: {sorted(missing)}")

    if not frame.empty:
        parsed = pd.to_datetime(frame["acq_date"], errors="coerce")
        if parsed.isna().any():
            raise ValueError("Ada tanggal observasi tidak valid")
        if not parsed.between(pd.Timestamp(start), pd.Timestamp(end)).all():
            raise ValueError("Ada observasi di luar periode permintaan")

    return frame, response.content


def save_batch(frame, payload: bytes, source: str, start: date, end: date):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    per_date = {}
    written_dates = []
    kept_dates = []
    current = start

    while current <= end:
        day = current.isoformat()
        output = RAW_DIR / f"{day}.csv"
        slice_df = frame.loc[frame["acq_date"] == day]
        per_date[day] = len(slice_df)
        if output.exists():
            kept_dates.append(day)
        else:
            temp_file = output.with_suffix(".csv.tmp")
            slice_df.to_csv(temp_file, index=False)
            temp_file.replace(output)
            written_dates.append(day)
        current += timedelta(days=1)

    metadata = {
        "source": source,
        "area": AREA,
        "request_start": start.isoformat(),
        "request_end": end.isoformat(),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "http_status": 200,
        "response_sha256": hashlib.sha256(payload).hexdigest(),
        "record_count": int(len(frame)),
        "columns": frame.columns.tolist(),
        "record_count_per_date": per_date,
        "new_partitions": written_dates,
        "existing_partitions_preserved": kept_dates,
    }
    metadata_path = METADATA_DIR / f"{source}_{start}_{end}.json"
    temp_metadata = metadata_path.with_suffix(".json.tmp")
    temp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temp_metadata.replace(metadata_path)

    print(f"  {len(frame):,} deteksi; {len(written_dates)} partisi baru, "
          f"{len(kept_dates)} partisi lama dipertahankan")
    print("  Kolom:", ", ".join(frame.columns))
    for day, count in per_date.items():
        print(f"    {day}: {count:,} deteksi")


def main():
    parser = argparse.ArgumentParser(description="Historical backfill NASA FIRMS")
    parser.add_argument("--start", type=parse_date, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, required=True, help="YYYY-MM-DD")
    parser.add_argument("--source", choices=["VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"],
                        default="VIIRS_NOAA20_SP")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Jeda dalam detik antar-request (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Tampilkan rencana tanpa mengunduh")
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("Tanggal --end harus >= --start")
    if args.delay < 0:
        parser.error("--delay tidak boleh negatif")

    batches = list(date_batches(args.start, args.end))
    print(f"Produk: {args.source}; periode: {args.start} sampai {args.end}")
    print(f"Total batch (maksimum 5 hari): {len(batches)}")
    print(f"Direktori output: {RAW_DIR}")
    if args.dry_run:
        for first, last in batches:
            print(f"  {first} sampai {last}")
        return 0

    key = get_map_key()
    session = create_session()
    downloaded = 0
    for first, last in batches:
        dates = [first + timedelta(days=d) for d in range((last - first).days + 1)]
        paths = [RAW_DIR / f"{d}.csv" for d in dates]
        if all(p.is_file() for p in paths):
            print(f"SKIP {first} sampai {last}: semua partisi sudah tersedia")
            continue
        print(f"FETCH {first} sampai {last}")
        try:
            frame, payload = fetch_batch(session, key, args.source, first, last)
            save_batch(frame, payload, args.source, first, last)
        except (requests.RequestException, ValueError, OSError) as exc:
            # Do NOT print exception details: requests errors can contain the API key in URLs.
            print(f"GAGAL {first} sampai {last}: {type(exc).__name__}", file=sys.stderr)
            print("Batch yang berhasil sebelumnya tetap tersimpan; jalankan ulang untuk melanjutkan.",
                  file=sys.stderr)
            return 1
        downloaded += 1
        if args.delay and first != batches[-1][0]:
            time.sleep(args.delay)
    print(f"Selesai. Batch diunduh: {downloaded}; sisanya dilewati bila sudah tersedia.")
    print("Catatan: dataset belum otomatis diproses atau di-versioning ulang oleh DVC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

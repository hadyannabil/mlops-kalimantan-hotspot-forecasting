"""Data ingestion untuk NASA FIRMS VIIRS NOAA-20 NRT."""

import hashlib
import json
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone
from pathlib import Path
from io import StringIO
import pandas as pd
import requests


BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
SOURCE = "VIIRS_NOAA20_NRT"
AREA = "108,-5,120,8"
DAY_RANGE = 3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "firms"
SNAPSHOT_DIR = RAW_DIR / "snapshots"
REQUIRED_COLUMNS = {
    "latitude",
    "longitude",
    "bright_ti4",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_ti5",
    "frp",
    "daynight",
}


def get_map_key() -> str:
    """Mengambil NASA FIRMS MAP_KEY dari environment variable."""
    map_key = os.getenv("MAP_KEY")

    if not map_key:
        raise ValueError(
            "MAP_KEY tidak ditemukan. "
            "Set MAP_KEY terlebih dahulu di environment variable."
        )

    return map_key


def create_session() -> requests.Session:
    """Membuat HTTP session dengan mekanisme retry."""
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)

    return session


def fetch_data(
    session: requests.Session,
    map_key: str
) -> requests.Response:
    """Mengambil tiga hari data terbaru dari NASA FIRMS."""
    url = (
        f"{BASE_URL}/{map_key}/{SOURCE}/"
        f"{AREA}/{DAY_RANGE}"
    )

    print("Mengambil data dari NASA FIRMS...")
    print(f"Source       : {SOURCE}")
    print(f"Bounding box : {AREA}")
    print(f"Day range    : {DAY_RANGE} hari")

    response = session.get(
        url,
        timeout=60
    )

    response.raise_for_status()

    return response


def parse_data(response: requests.Response) -> pd.DataFrame:
    """Membaca dan memvalidasi respons CSV NASA FIRMS."""
    if not response.text.strip():
        raise ValueError("Respons NASA FIRMS kosong.")

    data = pd.read_csv(
        StringIO(response.text),
        dtype={"acq_time": "string"}
    )

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Kolom FIRMS tidak lengkap: {sorted(missing_columns)}"
        )

    return data


def save_raw_data(
    response: requests.Response,
    data: pd.DataFrame
) -> None:
    """Menyimpan snapshot, metadata, dan partisi raw data per tanggal."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    ingestion_time = datetime.now(timezone.utc)

    run_id = ingestion_time.strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    # Menyimpan snapshot raw
    snapshot_path = SNAPSHOT_DIR / f"firms_{run_id}.csv"

    snapshot_path.write_text(
        response.text,
        encoding="utf-8"
    )

    print(f"\nSnapshot disimpan: {snapshot_path}")

    # Menghitung checksum snapshot
    checksum = hashlib.sha256(
        response.content
    ).hexdigest()

    # Menyimpan metadata ingestion
    metadata = {
        "run_id": run_id,
        "source": SOURCE,
        "bounding_box": AREA,
        "day_range": DAY_RANGE,
        "ingestion_timestamp": ingestion_time.isoformat(),
        "http_status": response.status_code,
        "record_count": len(data),
        "data_dates": sorted(data["acq_date"].unique().tolist()),
        "checksum_sha256": checksum,
        "status": "success"
    }

    metadata_path = SNAPSHOT_DIR / f"metadata_{run_id}.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Metadata disimpan: {metadata_path}")

    # Menyimpan partisi aktif per tanggal
    for data_date, daily_data in data.groupby("acq_date"):
        output_path = RAW_DIR / f"{data_date}.csv"

        daily_data.to_csv(
            output_path,
            index=False
        )

        print(
            f"Partisi {data_date}: "
            f"{len(daily_data)} record"
        )


def save_failure_metadata(error: Exception) -> None:
    """Menyimpan metadata ketika proses ingestion gagal."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    ingestion_time = datetime.now(timezone.utc)

    run_id = ingestion_time.strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    metadata = {
        "run_id": run_id,
        "source": SOURCE,
        "bounding_box": AREA,
        "day_range": DAY_RANGE,
        "ingestion_timestamp": ingestion_time.isoformat(),
        "status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error)
    }

    metadata_path = SNAPSHOT_DIR / f"metadata_{run_id}.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Metadata kegagalan disimpan: {metadata_path}")


if __name__ == "__main__":
    try:
        map_key = get_map_key()
        session = create_session()

        response = fetch_data(
            session,
            map_key
        )

        print(f"HTTP status  : {response.status_code}")

        data = parse_data(response)

        print(f"Jumlah record: {len(data)}")
        print(f"Jumlah kolom : {len(data.columns)}")

        print("\nKolom:")
        print(data.columns.tolist())

        print("\nJumlah record per tanggal:")
        print(data["acq_date"].value_counts().sort_index())

        save_raw_data(response, data)

    except Exception as error:
        print(f"\nIngestion gagal: {error}")
        save_failure_metadata(error)
        raise SystemExit(1) from error

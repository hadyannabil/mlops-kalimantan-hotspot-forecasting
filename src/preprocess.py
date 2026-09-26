"""Preprocessing data NASA FIRMS."""

from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "firms"
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "reference"
    / "geoBoundaries-IDN-ADM2.geojson"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_FILE = (
    PROCESSED_DIR
    / "daily_hotspot_activity.csv"
)

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

TARGET_REGIONS = {
    "Kalimantan Barat": [
        "Sambas",
        "Bengkayang",
        "Landak",
        "Mempawah",
        "Sanggau",
        "Ketapang",
        "Sintang",
        "Kapuas Hulu",
        "Sekadau",
        "Melawi",
        "Kayong Utara",
        "Kubu Raya",
        "Kota Pontianak",
        "Kota Singkawang",
    ],
    "Kalimantan Tengah": [
        "Kotawaringin Barat",
        "Kotawaringin Timur",
        "Kapuas",
        "Barito Selatan",
        "Barito Utara",
        "Sukamara",
        "Lamandau",
        "Seruyan",
        "Katingan",
        "Pulang Pisau",
        "Gunung Mas",
        "Barito Timur",
        "Murung Raya",
        "Kota Palangka Raya",
    ],
    "Kalimantan Selatan": [
        "Tanah Laut",
        "Kota Baru",
        "Banjar",
        "Barito Kuala",
        "Tapin",
        "Hulu Sungai Selatan",
        "Hulu Sungai Tengah",
        "Hulu Sungai Utara",
        "Tabalong",
        "Tanah Bumbu",
        "Balangan",
        "Kota Banjarmasin",
        "Kota Banjar Baru",
    ],
    "Kalimantan Timur": [
        "Paser",
        "Kutai Barat",
        "Kutai Kartanegara",
        "Kutai Timur",
        "Berau",
        "Penajam Paser Utara",
        "Mahakam Hulu",
        "Kota Balikpapan",
        "Kota Samarinda",
        "Kota Bontang",
    ],
    "Kalimantan Utara": [
        "Bulungan",
        "Malinau",
        "Nunukan",
        "Tana Tidung",
        "Kota Tarakan",
    ],
}


def load_raw_data() -> pd.DataFrame:
    """Membaca seluruh partisi raw FIRMS aktif."""
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "Tidak ada file CSV raw di data/raw/firms/."
        )

    dataframes = []

    for csv_file in csv_files:
        data = pd.read_csv(
            csv_file,
            dtype={"acq_time": "string"}
        )

        dataframes.append(data)

    combined_data = pd.concat(
        dataframes,
        ignore_index=True
    )

    return combined_data


def validate_and_normalize(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Memvalidasi kolom dan menormalisasi tipe data dasar."""
    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak lengkap: "
            f"{sorted(missing_columns)}"
        )

    data = data.copy()

    data["acq_date"] = pd.to_datetime(
        data["acq_date"],
        errors="coerce"
    )

    data["acq_time"] = (
        data["acq_time"]
        .astype("string")
        .str.zfill(4)
    )

    data["latitude"] = pd.to_numeric(
        data["latitude"],
        errors="coerce"
    )

    data["longitude"] = pd.to_numeric(
        data["longitude"],
        errors="coerce"
    )

    return data


def clean_invalid_records(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Menghapus record dengan tanggal, waktu, atau koordinat tidak valid."""
    data = data.copy()

    invalid_date = data["acq_date"].isna()

    invalid_time = ~data["acq_time"].str.match(
        r"^(?:[01]\d|2[0-3])[0-5]\d$",
        na=False
    )

    invalid_coordinate = (
        data["latitude"].isna()
        | data["longitude"].isna()
        | ~data["latitude"].between(-5, 8)
        | ~data["longitude"].between(108, 120)
    )

    invalid_mask = (
        invalid_date
        | invalid_time
        | invalid_coordinate
    )

    print("\nValidasi record:")
    print(
        f"Tanggal invalid   : "
        f"{invalid_date.sum()}"
    )
    print(
        f"Waktu invalid     : "
        f"{invalid_time.sum()}"
    )
    print(
        f"Koordinat invalid : "
        f"{invalid_coordinate.sum()}"
    )
    print(
        f"Total invalid     : "
        f"{invalid_mask.sum()}"
    )

    clean_data = data.loc[
        ~invalid_mask
    ].copy()

    return clean_data


def remove_duplicates(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Menghapus exact duplicate berdasarkan seluruh kolom raw."""
    duplicate_count = data.duplicated().sum()

    clean_data = (
        data
        .drop_duplicates()
        .copy()
    )

    print("\nPengecekan duplikat:")
    print(
        f"Exact duplicate : "
        f"{duplicate_count}"
    )
    print(
        f"Record tersisa  : "
        f"{len(clean_data)}"
    )

    return clean_data


def check_missing_values(
    data: pd.DataFrame
) -> None:
    """Menampilkan jumlah missing value pada setiap kolom."""
    missing_values = data.isna().sum()

    missing_values = missing_values[
        missing_values > 0
    ]

    print("\nPengecekan missing value:")

    if missing_values.empty:
        print("Tidak ada missing value.")
    else:
        print(missing_values)


def spatial_join_adm2(
    data: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Mencocokkan titik FIRMS dengan wilayah ADM2 Indonesia."""
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"File referensi tidak ditemukan: "
            f"{REFERENCE_FILE}"
        )

    boundaries = gpd.read_file(
        REFERENCE_FILE
    )

    hotspots = gpd.GeoDataFrame(
        data.copy(),
        geometry=gpd.points_from_xy(
            data["longitude"],
            data["latitude"]
        ),
        crs="EPSG:4326"
    )

    if boundaries.crs != hotspots.crs:
        boundaries = boundaries.to_crs(
            hotspots.crs
        )

    joined = gpd.sjoin(
        hotspots,
        boundaries[
            [
                "shapeName",
                "shapeID",
                "geometry"
            ]
        ],
        how="left",
        predicate="within"
    )

    matched_count = (
        joined["shapeID"]
        .notna()
        .sum()
    )

    unmatched_count = (
        joined["shapeID"]
        .isna()
        .sum()
    )

    print("\nSpatial join ADM2:")
    print(
        f"Total record       : "
        f"{len(joined)}"
    )
    print(
        f"Cocok dengan ADM2  : "
        f"{matched_count}"
    )
    print(
        f"Tidak cocok ADM2   : "
        f"{unmatched_count}"
    )

    return joined


def filter_target_regions(
    data: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Mempertahankan hanya 56 kabupaten/kota wilayah studi."""
    region_to_province = {
        region: province
        for province, regions in TARGET_REGIONS.items()
        for region in regions
    }

    target_names = set(
        region_to_province
    )

    filtered = data[
        data["shapeName"].isin(
            target_names
        )
    ].copy()

    filtered["province"] = (
        filtered["shapeName"]
        .map(region_to_province)
    )

    filtered["kabupaten_kota"] = (
        filtered["shapeName"]
    )

    print("\nFilter wilayah studi:")
    print(
        f"Record sebelum filter : "
        f"{len(data)}"
    )
    print(
        f"Record dalam 56 wilayah: "
        f"{len(filtered)}"
    )
    print(
        f"Record di luar scope   : "
        f"{len(data) - len(filtered)}"
    )
    print(
        f"Wilayah dengan deteksi : "
        f"{filtered['shapeID'].nunique()}"
    )

    return filtered


def create_daily_aggregation(
    data: gpd.GeoDataFrame,
    all_dates
) -> pd.DataFrame:
    """Membuat agregasi harian lengkap untuk 56 wilayah studi."""
    region_to_province = {
        region: province
        for province, regions in TARGET_REGIONS.items()
        for region in regions
    }

    boundaries = gpd.read_file(
        REFERENCE_FILE
    )

    target_boundaries = (
        boundaries[
            boundaries["shapeName"].isin(
                region_to_province
            )
        ][
            [
                "shapeID",
                "shapeName"
            ]
        ]
        .copy()
    )

    target_boundaries["province"] = (
        target_boundaries["shapeName"]
        .map(region_to_province)
    )

    target_boundaries = (
        target_boundaries
        .rename(
            columns={
                "shapeName": "kabupaten_kota"
            }
        )
    )

    aggregation = (
        data.groupby(
            [
                "acq_date",
                "shapeID",
                "province",
                "kabupaten_kota"
            ]
        )
        .agg(
            hotspot_count=(
                "shapeID",
                "size"
            ),
            mean_frp=(
                "frp",
                "mean"
            ),
            max_frp=(
                "frp",
                "max"
            ),
            total_frp=(
                "frp",
                "sum"
            )
        )
        .reset_index()
    )

    dates = sorted(
        pd.Series(all_dates)
        .dropna()
        .unique()
    )

    complete_grid = pd.MultiIndex.from_product(
        [
            dates,
            target_boundaries["shapeID"]
        ],
        names=[
            "date",
            "shapeID"
        ]
    ).to_frame(
        index=False
    )

    complete_grid = complete_grid.merge(
        target_boundaries,
        on="shapeID",
        how="left"
    )

    aggregation = aggregation.rename(
        columns={
            "acq_date": "date"
        }
    )

    daily_data = complete_grid.merge(
        aggregation,
        on=[
            "date",
            "shapeID",
            "province",
            "kabupaten_kota"
        ],
        how="left"
    )

    daily_data["hotspot_count"] = (
        daily_data["hotspot_count"]
        .fillna(0)
        .astype(int)
    )

    daily_data = (
        daily_data
        .sort_values(
            [
                "date",
                "province",
                "kabupaten_kota"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print("\nDaily aggregation:")
    print(
        f"Jumlah tanggal       : "
        f"{daily_data['date'].nunique()}"
    )
    print(
        f"Jumlah wilayah       : "
        f"{daily_data['shapeID'].nunique()}"
    )
    print(
        f"Total baris          : "
        f"{len(daily_data)}"
    )
    print(
        f"Total hotspot        : "
        f"{daily_data['hotspot_count'].sum()}"
    )

    return daily_data


def save_processed_data(
    daily_data: pd.DataFrame
) -> None:
    """Memperbarui historical processed dataset tanpa membuat duplikat."""
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_data = daily_data.copy()

    current_time = pd.Timestamp.now(tz="UTC")

    output_data["status"] = output_data["date"].apply(
        lambda date: (
            "ready"
            if current_time >= (
                pd.Timestamp(date, tz="UTC")
                + pd.Timedelta(days=1, hours=6)
            )
            else "partial"
        )
    )

    # Tanggal yang sedang diproses pada run ini.
    affected_dates = output_data[
        "date"
    ].unique()

    # Jika historical dataset sudah ada,
    # pertahankan tanggal lama yang tidak sedang diproses ulang.
    if PROCESSED_FILE.exists():
        existing_data = pd.read_csv(
            PROCESSED_FILE,
            dtype={
                "shapeID": "string"
            }
        )

        existing_data["date"] = pd.to_datetime(
            existing_data["date"],
            errors="coerce"
        )

        existing_data = existing_data[
            ~existing_data["date"].isin(
                affected_dates
            )
        ].copy()

        output_data = pd.concat(
            [
                existing_data,
                output_data
            ],
            ignore_index=True
        )

    output_data = (
        output_data
        .sort_values(
            [
                "date",
                "province",
                "kabupaten_kota"
            ]
        )
        .drop_duplicates(
            subset=[
                "date",
                "shapeID"
            ],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    output_data.to_csv(
        PROCESSED_FILE,
        index=False
    )

    print("\nProcessed data:")
    print(
        f"Disimpan ke : "
        f"{PROCESSED_FILE}"
    )
    print(
        f"Total baris historical : "
        f"{len(output_data)}"
    )

    print(
        "\nStatus per tanggal terbaru:"
    )

    print(
        output_data[
            output_data["date"].isin(
                affected_dates
            )
        ][
            [
                "date",
                "status"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "date"
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    data = load_raw_data()

    print(
        f"Raw data: "
        f"{len(data)} record"
    )

    data = validate_and_normalize(
        data
    )

    data = clean_invalid_records(
        data
    )

    data = remove_duplicates(
        data
    )

    check_missing_values(
        data
    )

    joined_data = spatial_join_adm2(
        data
    )

    kalimantan_data = filter_target_regions(
        joined_data
    )

    daily_data = create_daily_aggregation(
        kalimantan_data,
        data["acq_date"].unique()
    )

    save_processed_data(
        daily_data
    )

    print(
        "\nPreprocessing selesai."
    )

# Kalimantan Hotspot Forecasting

Proyek MLOps untuk mengolah data hotspot NASA FIRMS dan menyiapkan dataset harian kabupaten/kota di Kalimantan sebagai dasar prediksi lonjakan jumlah deteksi hotspot dalam tiga hari berikutnya.

Pipeline saat ini mencakup **data ingestion** dari NASA FIRMS Area API dan **preprocessing otomatis** hingga menghasilkan data aktivitas hotspot harian pada 56 kabupaten/kota di lima provinsi Kalimantan.

## Data Source

Sumber data utama:

- **NASA FIRMS Area API**
- Produk: **VIIRS NOAA-20 Near Real-Time (NRT)**
- Bounding box: `108,-5,120,8`
- Rentang pengambilan: 3 hari terbaru
- Format: CSV

Referensi batas administratif menggunakan **geoBoundaries ADM2 Indonesia** untuk memetakan koordinat hotspot ke kabupaten/kota.

## Project Structure

```text
mlops-kalimantan-hotspot-forecasting/
│
├── data/
│   ├── raw/
│   │   ├── firms/
│   │   │   ├── snapshots/
│   │   │   └── YYYY-MM-DD.csv
│   │   └── reference/
│   │       └── geoBoundaries-IDN-ADM2.geojson
│   │
│   └── processed/
│       └── daily_hotspot_activity.csv
│
├── src/
│   ├── ingest_data.py
│   ├── preprocess.py
│   └── initial_experiment.py
│
├── notebooks/
├── models/
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Proyek dikembangkan menggunakan Python 3.12 dan GitHub Codespaces.

Install seluruh dependency:

```bash
python -m pip install -r requirements.txt
```

Library utama yang digunakan antara lain:

- Pandas
- Requests
- GeoPandas
- Shapely
- NumPy
- Scikit-learn
- Matplotlib

## Data Ingestion

Script ingestion tersedia pada:

```text
src/ingest_data.py
```

Sebelum menjalankan script, simpan NASA FIRMS `MAP_KEY` sebagai environment variable:

```bash
export MAP_KEY="YOUR_MAP_KEY"
```

Jalankan ingestion:

```bash
python src/ingest_data.py
```

Script akan:

- mengambil data VIIRS NOAA-20 NRT untuk tiga hari terbaru;
- memvalidasi respons NASA FIRMS;
- menyimpan raw data berdasarkan tanggal;
- membuat snapshot baru untuk setiap ingestion run;
- menyimpan metadata ingestion;
- melakukan retry ketika terjadi gangguan request.

Contoh hasil:

```text
data/raw/firms/
├── 2026-09-23.csv
├── 2026-09-24.csv
├── 2026-09-25.csv
└── snapshots/
    ├── firms_<run_id>.csv
    └── metadata_<run_id>.json
```

Snapshot menggunakan timestamp sehingga data dari run sebelumnya tidak ditimpa.

## Preprocessing

Pastikan file geoBoundaries ADM2 Indonesia tersedia pada:

```text
data/raw/reference/geoBoundaries-IDN-ADM2.geojson
```

Kemudian jalankan:

```bash
python src/preprocess.py
```

Tahapan preprocessing meliputi:

- validasi struktur dan tipe data;
- pengecekan tanggal, waktu, dan koordinat;
- penghapusan exact duplicate;
- pengecekan missing value;
- spatial join dengan geoBoundaries ADM2;
- penyaringan 56 kabupaten/kota di Kalimantan;
- agregasi jumlah hotspot per wilayah dan tanggal;
- pembentukan complete daily grid untuk seluruh wilayah studi.

Hasil preprocessing disimpan pada:

```text
data/processed/daily_hotspot_activity.csv
```

Dataset utama memiliki kolom:

```text
date
shapeID
kabupaten_kota
province
hotspot_count
mean_frp
max_frp
total_frp
status
```

Wilayah tanpa deteksi pada tanggal yang valid tetap disimpan dengan `hotspot_count = 0`.

## Running the Pipeline

Urutan menjalankan pipeline:

```bash
python src/ingest_data.py
python src/preprocess.py
```

`ingest_data.py` dapat dijalankan kembali untuk mengambil data terbaru tanpa menghapus snapshot dari proses sebelumnya. `preprocess.py` juga memperbarui historical processed dataset berdasarkan kombinasi tanggal dan ID wilayah tanpa menambahkan baris duplikat.

## Next Steps

Tahap pengembangan berikutnya mencakup:

- feature engineering;
- pembentukan label SPIKE / NO SPIKE;
- temporal train-validation-test split;
- training Random Forest;
- daily inference;
- monitoring dan evaluasi kebutuhan retraining.

## License

Proyek ini menggunakan **MIT License**.
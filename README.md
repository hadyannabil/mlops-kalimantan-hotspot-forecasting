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
│   │   │   └── YYYY-MM-DD.csv
│   │   └── reference/
│   │       └── geoBoundaries-IDN-ADM2.geojson   # downloaded locally
│   └── processed/
│       └── daily_hotspot_activity.csv           # generated locally
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

Repository menyertakan sampel raw data FIRMS. File snapshot ingestion, metadata, processed dataset, dan geoBoundaries dihasilkan atau diunduh secara lokal dan tidak disimpan di repository.

## Setup

Proyek dikembangkan menggunakan Python 3.12 dan GitHub Codespaces.

Install dependency:

```bash
python -m pip install -r requirements.txt
```

Library utama yang digunakan meliputi Pandas, Requests, GeoPandas, Shapely, NumPy, Scikit-learn, dan Matplotlib.

## Data Ingestion

Data ingestion dijalankan melalui:

```text
src/ingest_data.py
```

Sebelum menjalankan script, siapkan NASA FIRMS `MAP_KEY` sebagai environment variable:

```bash
export MAP_KEY="YOUR_MAP_KEY"
```

Untuk GitHub Codespaces, `MAP_KEY` dapat disimpan sebagai **Codespaces Secret** agar tidak ditulis langsung pada source code atau repository.

Jalankan ingestion:

```bash
python src/ingest_data.py
```

Script akan:

- mengambil data VIIRS NOAA-20 NRT untuk tiga hari terbaru;
- memvalidasi respons dan struktur data;
- melakukan retry ketika terjadi gangguan request;
- menyimpan raw data berdasarkan tanggal;
- membuat snapshot untuk setiap ingestion run;
- menyimpan metadata ingestion.

Snapshot menggunakan timestamp sehingga data dari run sebelumnya tidak ditimpa.

## Preprocessing

Preprocessing menggunakan **geoBoundaries ADM2 Indonesia** sebagai referensi batas administratif.

File referensi tidak disimpan di repository karena ukurannya besar. Jika belum tersedia, jalankan:

```bash
mkdir -p data/raw/reference

wget -O data/raw/reference/geoBoundaries-IDN-ADM2.geojson \
"https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IDN/ADM2/geoBoundaries-IDN-ADM2.geojson"
```

Kemudian jalankan:

```bash
python src/preprocess.py
```

Tahapan preprocessing meliputi:

- validasi struktur dan tipe data;
- validasi tanggal, waktu, dan koordinat;
- penghapusan exact duplicate;
- pengecekan missing value;
- spatial join dengan geoBoundaries ADM2;
- penyaringan 56 kabupaten/kota di Kalimantan;
- agregasi hotspot per wilayah dan tanggal;
- pembentukan complete daily grid untuk seluruh wilayah studi.

Hasil preprocessing disimpan pada:

```text
data/processed/daily_hotspot_activity.csv
```

Dataset memiliki kolom:

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

`ingest_data.py` dapat dijalankan kembali untuk mengambil data terbaru tanpa menghapus snapshot sebelumnya.

`preprocess.py` memperbarui historical processed dataset berdasarkan kombinasi `date` dan `shapeID` sehingga pemrosesan ulang tidak menghasilkan baris duplikat.

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

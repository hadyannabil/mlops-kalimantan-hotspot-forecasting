# Kalimantan Hotspot Forecasting

Proyek MLOps untuk mengolah data hotspot NASA FIRMS dan menyiapkan dataset harian kabupaten/kota di Kalimantan sebagai dasar prediksi lonjakan jumlah deteksi hotspot dalam tiga hari berikutnya.

Pipeline saat ini mencakup **data ingestion** dari NASA FIRMS Area API dan **preprocessing otomatis** hingga menghasilkan data aktivitas hotspot harian pada 56 kabupaten/kota di lima provinsi Kalimantan. Versi raw data dikelola dengan **DVC** agar perubahan dataset dapat dilacak dan dibandingkan.

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
├── .dvc/
│   └── config                                   # konfigurasi DVC remote
├── data/
│   ├── raw/
│   │   ├── firms/                               # dilacak DVC, tidak masuk Git
│   │   │   └── YYYY-MM-DD.csv
│   │   ├── firms.dvc                            # penunjuk versi data (masuk Git)
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
├── .dvcignore
├── requirements.txt
└── README.md
```

Raw data FIRMS (partisi harian CSV, snapshot ingestion, dan metadata) dilacak dengan DVC dan tidak disimpan langsung di Git. Processed dataset dan geoBoundaries dihasilkan atau diunduh secara lokal dan tidak disimpan di repository.

## Setup

Proyek dikembangkan menggunakan Python 3.12 dan GitHub Codespaces.

Install dependency:

```bash
python -m pip install -r requirements.txt
```

Library utama yang digunakan meliputi Pandas, Requests, GeoPandas, Shapely, NumPy, Scikit-learn, Matplotlib, dan DVC untuk versioning data.

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

Setiap pertambahan data pada `data/raw/firms` dicatat sebagai versi baru dengan DVC (lihat bagian Data Versioning dengan DVC).

## Data Versioning dengan DVC

Raw data FIRMS (`data/raw/firms/`) dilacak dengan **DVC**, bukan Git. Git hanya menyimpan file penunjuk `data/raw/firms.dvc` (hash md5, ukuran, jumlah file), sedangkan isi data disimpan di DVC remote sehingga repository tetap kecil.

### Mengapa Versioning Data Penting

Hasil model Random Forest ditentukan oleh kode, konfigurasi, dan data. Git hanya menjamin dua yang pertama. Tanpa versioning data, kode yang sama pada dataset berbeda menghasilkan label, `threshold_spike`, model, dan metrik yang berbeda. Risikonya nyata di proyek ini karena data NRT dapat direvisi atau digantikan produk SP, ingestion mengambil ulang tiga hari terbaru sehingga isi partisi dapat berubah antar run, dan dataset terus bertambah.

Dengan DVC, setiap versi data punya hash md5 yang terikat ke commit Git. Model dapat dilatih ulang dengan data yang sama persis (`git checkout` lalu `dvc checkout` atau `dvc pull`), dan model kandidat dibandingkan dengan model aktif secara adil. Untuk audit, kapan dan apa yang berubah dapat ditelusuri lewat commit, tag, dan `dvc diff`, sedangkan perubahan tidak sengaja terdeteksi lewat `dvc status`.

### Versi Data

| Tag | Commit | File | Total baris CSV | md5 (awal) |
|---|---|---|---|---|
| data-v1 | 408d406 | 15 | 9208 | e913b246... |
| data-v2 | ccd5615 | 21 | 14155 | c272709a... |

data-v2 dibuat setelah ingestion dijalankan ulang secara manual (4.944 record baru, 9 sampai 11 Oktober 2026). `dvc diff data-v1 data-v2 --targets data/raw/firms` menunjukkan 6 file bertambah tanpa file lama yang berubah atau terhapus.

### Menambah Versi Data Baru

1. Jalankan ingestion ulang (`MAP_KEY` harus sudah di-export).
```bash
   python src/ingest_data.py
```
2. Cek perubahan data.
```bash
   dvc status data/raw/firms.dvc
```
3. Lacak versi baru dengan DVC.
```bash
   dvc add data/raw/firms
```
4. Commit file penunjuk dan beri tag versi.
```bash
   git add data/raw/firms.dvc data/raw/.gitignore
   git commit -m "Update raw FIRMS data v3"
   git tag data-v3
```
5. Kirim data ke remote dan kode ke GitHub.
```bash
   dvc push data/raw/firms.dvc
   git push origin main data-v3
```
6. Bandingkan dengan versi sebelumnya.
```bash
   dvc diff data-v2 data-v3 --targets data/raw/firms
```

Jangan gunakan `git add` pada `data/raw/firms`, selalu gunakan `dvc add`.

### Memulihkan Versi Data Tertentu

Versi lama dapat dipulihkan dengan mengambil file penunjuknya dari tag yang diinginkan lalu menjalankan `dvc checkout`.

```bash
git checkout data-v1 -- data/raw/firms.dvc   # ambil penunjuk versi lama
dvc checkout data/raw/firms.dvc              # pulihkan datanya
git checkout HEAD -- data/raw/firms.dvc      # kembali ke penunjuk terbaru
dvc checkout data/raw/firms.dvc
```

Pada pengujian, data-v1 dipulihkan menjadi 15 file (9208 baris CSV) dan versi terbaru kembali menjadi 21 file (14155 baris CSV).

### Remote Storage

Remote `storage` memakai direktori lokal `/workspaces/dvc-storage` di dalam Codespaces (konfigurasi pada `.dvc/config`). Keterbatasannya, remote ini belum merupakan penyimpanan eksternal, datanya hilang jika Codespace dihapus, dan tidak dapat diakses oleh orang yang meng-clone repository sehingga `dvc pull` gagal dan data dibuat ulang dengan menjalankan `ingest_data.py`. Untuk penggunaan nyata, remote dapat diganti ke S3, GCS, atau MinIO dengan `dvc remote add` atau `dvc remote modify`.

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
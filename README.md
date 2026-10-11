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
│   │   ├── .gitignore                           # dibuat DVC, memuat entri firms/
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

Raw data FIRMS (`data/raw/firms/`) dilacak dengan **DVC**, bukan Git. Git hanya menyimpan file penunjuk `data/raw/firms.dvc` (hash md5, ukuran, jumlah file), sedangkan isi data disimpan di DVC remote sehingga repository tetap kecil. Processed dataset dan geoBoundaries belum dilacak DVC.

### Mengapa Versioning Data Penting

Hasil model Random Forest ditentukan oleh kode, konfigurasi, dan data, sedangkan Git hanya menjamin dua yang pertama. Tanpa versioning data, kode yang sama pada dataset berbeda menghasilkan label, `threshold_spike`, model, dan metrik yang berbeda, dan risikonya nyata di proyek ini karena data NRT dapat direvisi atau digantikan produk SP, ingestion mengambil ulang tiga hari terbaru, dan dataset terus bertambah. Dengan DVC, setiap versi data punya hash md5 yang terikat ke commit Git, sehingga model dapat dilatih ulang dan dibandingkan dengan dataset yang sama persis selama datanya masih ada di cache atau remote. Untuk audit, perubahan ditelusuri lewat commit, tag, dan `dvc diff`, sedangkan perubahan tidak sengaja terdeteksi lewat `dvc status`.

### Setup DVC dan Versi Data

1. Inisialisasi DVC dan commit konfigurasinya.
```bash
   dvc init
   git commit -m "Initialize DVC"
```
2. Lepaskan sampel CSV yang sebelumnya dilacak Git. Pengecualian `!data/raw/*.dvc` dan `!data/raw/.gitignore` ditambahkan di `.gitignore` utama karena `data/raw/*` diabaikan, sehingga file penunjuk DVC dapat masuk ke Git.
```bash
   git rm -r --cached data/raw/firms
```
3. Lacak raw data, commit file penunjuknya, dan beri tag versi pertama.
```bash
   dvc add data/raw/firms
   git add .gitignore data/raw/firms.dvc data/raw/.gitignore
   git commit -m "Track raw FIRMS data v1 with DVC"
   git tag data-v1
```
4. Konfigurasi remote, lalu kirim data ke remote.
```bash
   dvc remote add -d storage /workspaces/dvc-storage
   git add .dvc/config
   git commit -m "Configure DVC remote storage"
   dvc push -T data/raw/firms.dvc
```

| Tag | Commit | File | Total baris CSV (`wc -l`) | md5 (awal) |
|---|---|---|---|---|
| data-v1 | 408d406 | 15 | 9.208 | e913b246... |
| data-v2 | ccd5615 | 21 | 14.155 | c272709a... |

data-v2 dibuat setelah ingestion dijalankan ulang secara manual (4.944 record baru, 9 sampai 11 Oktober 2026), dan `dvc diff data-v1 data-v2 --targets data/raw/firms` menunjukkan 6 file bertambah tanpa file lama yang berubah atau terhapus.

### Menambah dan Memulihkan Versi Data

Contoh berikut memakai `data-v3` untuk versi ketiga.

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
5. Kirim data ke remote dan kode ke GitHub. Opsi `-T` memastikan data pada semua tag ikut tersimpan di remote.
```bash
   dvc push -T data/raw/firms.dvc
   git push origin main data-v3
```
6. Bandingkan dengan versi sebelumnya.
```bash
   dvc diff data-v2 data-v3 --targets data/raw/firms
```

Jangan gunakan `git add` pada `data/raw/firms`, selalu gunakan `dvc add`.

Untuk memulihkan versi lama, ambil file penunjuknya dari tag yang diinginkan lalu jalankan `dvc checkout`.

```bash
git checkout data-v1 -- data/raw/firms.dvc   # ambil penunjuk versi lama
dvc checkout data/raw/firms.dvc              # pulihkan datanya
git checkout HEAD -- data/raw/firms.dvc      # kembali ke penunjuk terbaru
dvc checkout data/raw/firms.dvc
```

Hasilnya dapat diperiksa dengan `wc -l data/raw/firms/*.csv` (9.208 baris untuk data-v1 dan 14.155 untuk data-v2).

### Remote Storage

Remote `storage` berupa direktori lokal `/workspaces/dvc-storage` di dalam Codespaces (konfigurasi pada `.dvc/config`), jadi belum merupakan penyimpanan eksternal, datanya hilang jika Codespace dihapus, dan tidak dapat diakses orang yang meng-clone repository. Untuk penggunaan nyata, remote dapat diganti ke S3, GCS, atau MinIO dengan `dvc remote modify`.

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

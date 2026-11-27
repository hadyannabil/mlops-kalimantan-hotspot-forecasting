# Kalimantan Hotspot Forecasting

Proyek **MLOps** untuk memprediksi potensi **lonjakan aktivitas hotspot (SPIKE / NO SPIKE)** pada 56 kabupaten/kota di Kalimantan dalam horizon **1–3 hari ke depan**. Model yang digunakan adalah **Random Forest**, dengan **Persistence** sebagai baseline pembanding.

Proyek mencakup pengolahan data NASA FIRMS, eksperimen MLflow, versioning DVC, otomatisasi GitHub Actions, orkestrasi Docker Compose, horizontal scaling, serta observability dan deteksi data drift menggunakan Prometheus dan Grafana.

## Data Source

- **Sumber:** NASA FIRMS Area API, VIIRS NOAA-20; ingestion terkini menggunakan data *Near Real-Time* (NRT).
- **Wilayah:** 56 kabupaten/kota di lima provinsi Kalimantan; bounding box `108,-5,120,8`.
- **Pengambilan data:** tiga hari terbaru, format CSV. Dataset pemodelan berasal dari snapshot historis SP/NRT yang terdokumentasi.
- **Referensi administratif:** geoBoundaries ADM2 Indonesia untuk pemetaan koordinat ke kabupaten/kota.

## Project Structure

```text
mlops-kalimantan-hotspot-forecasting/
├── .github/workflows/mlops-automation.yaml   # CI/CD LK08
├── .dvc/config                                # konfigurasi remote DVC
├── api/app.py                                 # REST API inferensi & tracking
├── data/raw/                                  # data FIRMS dan referensi wilayah
├── data/processed/                            # data hasil preprocessing & DVC
├── docker_assets/model.dvc                    # pointer model Production v1
├── docker_assets/model/                       # artefak model (diambil via DVC)
├── deployment/
│   ├── mlflow/                                # MLflow Server
│   ├── nginx/                                 # Load balancer
│   ├── serving/                               # Model serving dan instrumentasi
│   ├── monitoring/                            # Prometheus dan Grafana
│   └── production_model.yaml                  # Metadata model Production
├── scripts/                                   # Baseline drift dan simulasi
├── src/                                       # ingestion, preprocessing, training, evaluasi
├── tests/                                     # pengujian otomatis
├── Dockerfile                                 # image Flask API
├── docker-compose.yaml                       # orkestrasi layanan LK09-LK11
├── .env.example                              # template environment
├── requirements.txt
└── README.md
```

Data dan artefak model berukuran besar dikelola dengan **DVC**, bukan disimpan langsung di Git. File `.env` berisi konfigurasi lokal dan tidak di-commit.

## Setup

Lingkungan pengembangan menggunakan **Python 3.12** dan GitHub Codespaces. Untuk pipeline Python dan DVC:

```bash
python -m pip install -r requirements.txt
```

Untuk menjalankan layanan LK09–LK11 diperlukan **Docker Engine** dan **Docker Compose**. Panduan tersedia pada bagian LK-09 hingga LK-11.

## Data Pipeline — LK04

### Data Ingestion

Siapkan NASA FIRMS `MAP_KEY` sebagai environment variable (misalnya melalui *Codespaces Secret*), kemudian jalankan:

```bash
export MAP_KEY="YOUR_MAP_KEY"
python src/ingest_data.py
```

Skrip mengambil data VIIRS NOAA-20 NRT untuk tiga hari terbaru, memvalidasi respons, melakukan retry bila diperlukan, dan menyimpan data harian beserta snapshot serta metadata ingestion. Snapshot sebelumnya tidak ditimpa.

### Preprocessing

File geoBoundaries tidak disimpan di Git. Jika belum tersedia, siapkan referensinya:

```bash
mkdir -p data/raw/reference
wget -O data/raw/reference/geoBoundaries-IDN-ADM2.geojson \
  "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IDN/ADM2/geoBoundaries-IDN-ADM2.geojson"
python src/preprocess.py
```

Preprocessing meliputi validasi data, penghapusan duplikat, *spatial join*, agregasi hotspot per kabupaten/kota dan tanggal, serta pembentukan *complete daily grid*. Hasil utama disimpan di `data/processed/daily_hotspot_activity.csv`; wilayah tanpa deteksi pada tanggal valid memiliki `hotspot_count = 0`.

Untuk memperbarui data, jalankan `ingest_data.py` lalu `preprocess.py` secara berurutan. Perubahan dataset selanjutnya dicatat melalui DVC.

## Data Versioning dengan DVC — LK05

DVC digunakan agar perubahan data dapat ditelusuri bersama versi kode. Pada repository ini, data mentah dilacak melalui `data/raw/firms.dvc`, sedangkan dataset pemodelan menggunakan `data/processed/hotspot_model_ready.csv.dvc`.

### Menambahkan Versi Data

Setelah ingestion menghasilkan data baru, jalankan dari root repository:

```bash
python src/ingest_data.py
dvc status data/raw/firms.dvc
dvc add data/raw/firms
git add data/raw/firms.dvc data/raw/.gitignore
git commit -m "Update raw FIRMS data"
dvc push -r dagshub data/raw/firms.dvc
```

Perintah `dvc push` di atas mengirim **versi baru** ke DagsHub jika akun memiliki akses. Bila menggunakan remote lokal historis, ganti `-r dagshub` dengan `-r storage`. File CSV tidak dimasukkan langsung dengan `git add`; yang dicatat oleh Git adalah pointer `.dvc`. Setelah meninjau perubahan, push commit ke branch yang sesuai. Perubahan yang masuk ke `main` akan memicu workflow LK08.

Remote yang tersedia:

| Remote | Kegunaan |
|---|---|
| `storage` (default) | Direktori lokal `/workspaces/dvc-storage` pada Codespaces; tidak otomatis tersedia di clone baru. |
| `dagshub` | Remote eksternal untuk dataset yang telah disinkronkan dan artefak model; mungkin memerlukan access token. |

Contoh mengambil artefak model Production untuk Docker:

```bash
dvc pull -r dagshub docker_assets/model.dvc
```

Versioning awal proyek juga menggunakan tag `data-v1` dan `data-v2`. Riwayat lengkap versi dapat ditelusuri melalui Git dan DVC tanpa perlu mengulang proses `dvc init`.

## LK-07: Model Registry, Versioning, dan Kesiapan Inferensi

Model Random Forest dari eksperimen LK06 dicatat pada **MLflow Model Registry** dengan nama `KalimantanHotspotSpikeRF`.

| Komponen | Version 1 | Version 2 |
|---|---|---|
| Kandidat | `rf_01_conservative` | `rf_02_moderate` |
| Validation Average Precision | **0,618774** | 0,576339 |
| Status pada LK07 | **Production** | None |

**Version 1 dipilih untuk inferensi** karena memperoleh Average Precision tertinggi pada validation set, bukan karena hasil test set. Versi ini telah melalui transisi `None → Staging → Production` pada LK07, dengan *run ID* `0c95822681c640168f647d6f3a8e623d`.

Model Production v1 berhasil diuji menggunakan data yang tersedia dan input sintetis baru dengan 13 fitur. Pengujian tersebut menunjukkan kesiapan inferensi fungsional, bukan jaminan performa operasional. Artefak model yang digunakan untuk Docker kini dilacak di `docker_assets/model.dvc` dan disimpan pada remote DagsHub.

## Docker: Model Inference REST API

Sebelum LK09, model Production v1 telah dikemas menjadi API menggunakan **Flask, Gunicorn, dan Docker**. Dokumentasi ini dipisahkan karena implementasi kontainerisasi dilakukan pada tahapan sebelumnya.

| Endpoint | Fungsi |
|---|---|
| `GET /health` | Status API dan versi model |
| `POST /predict` | Prediksi biner `SPIKE` / `NO SPIKE` dari 1–100 observasi, masing-masing 13 fitur |
| `GET /tracking-status` | Pemeriksaan koneksi MLflow, ditambahkan pada LK09 |

Untuk menjalankan **API saja** tanpa Compose, pastikan artefak model sudah ditarik melalui DVC, lalu:

```bash
docker build -t kalimantan-hotspot-api:v1 .
docker run --rm -p 8001:8000 kalimantan-hotspot-api:v1
```

API memuat artefak dari `docker_assets/model/` saat startup; prediksi **tidak mengambil model langsung dari MLflow Registry**. Jangan jalankan container standalone pada port `8001` bersamaan dengan API Compose.

## LK-08: Otomatisasi Pipeline dengan GitHub Actions

Workflow [`.github/workflows/mlops-automation.yaml`](.github/workflows/mlops-automation.yaml) berjalan saat ada **push ke `main`**. Pipeline memakai DVC/DagsHub untuk data, MLflow untuk pelacakan model, dan GitHub Actions Secrets untuk autentikasi.

| Job | Proses |
|---|---|
| `testing` | Menarik dataset DVC dan menjalankan Pytest. |
| `training` | Melatih kandidat setelah pengujian lulus dan mencatat eksperimen di MLflow. |
| `evaluation_registry` | Mengevaluasi kandidat dan mendaftarkan yang lolos validasi ke **Staging**. |

Pada pengujian LK08, **20 pengujian berhasil**. Registrasi kandidat ke Staging **bukan** deployment otomatis dan tidak mengganti artefak Production v1 yang digunakan oleh API.

## LK-09: Orkestrasi Layanan ML dengan Docker Compose

[`docker-compose.yaml`](docker-compose.yaml) pada tahap LK09 menjalankan tiga layanan pada jaringan internal `hotspot-network`:

| Service | Fungsi | Port host |
|---|---|---|
| `api-service` | Inferensi Random Forest dan koneksi tracking | `8001` |
| `mlflow-server` | Tracking eksperimen dan artefak MLflow | `5000` |
| `postgres` | Database metadata MLflow | Internal saja |

### Menjalankan Layanan Dasar

Siapkan **Docker Engine/Compose**, DVC, dan akses ke remote DagsHub. Dari root repository, lakukan persiapan awal:

```bash
# Jika belum menginstal DVC, jalankan langkah Setup terlebih dahulu
test -f .env || cp .env.example .env
# Edit .env dan ganti POSTGRES_PASSWORD dengan nilai acak yang kuat
# (gunakan huruf/angka agar aman pada URI koneksi database)
dvc pull -r dagshub docker_assets/model.dvc
```

Pastikan `docker_assets/model/MLmodel` tersedia. Kemudian jalankan tiga layanan dasar:

```bash
docker compose up -d postgres mlflow-server api-service
```

Verifikasi layanan dan endpoint:

```bash
docker compose ps
curl http://localhost:8001/health
curl http://localhost:8001/tracking-status
curl http://localhost:5000/health
```

Targetnya: ketiga layanan `healthy`, API mengembalikan model `KalimantanHotspotSpikeRF` v1, dan `/tracking-status` menampilkan `"mlflow_connected": true`.

### Integrasi dan Persistence

Compose menggunakan `depends_on` dan *healthcheck* agar PostgreSQL siap sebelum MLflow, kemudian MLflow siap sebelum API. Komunikasi antar-service menggunakan hostname internal seperti `mlflow-server:5000` dan `postgres:5432`.

Metadata MLflow disimpan dalam volume `postgres_data`, sedangkan artefaknya berada di `mlflow_artifacts`. Pada pengujian LK09, metadata eksperimen dan file uji tetap tersedia setelah container dibuat ulang. API juga berhasil melakukan prediksi `NO SPIKE` dan membaca metadata dari MLflow melalui SDK. **Model prediksi tetap dimuat dari artefak lokal DVC**, bukan dari registry server LK09.

Batas resource saat pengujian: API **1 CPU / 2 GB**, MLflow **0,6 CPU / 3 GB** (1 worker), dan PostgreSQL **0,4 CPU / 512 MB**. Batas MLflow disesuaikan setelah pengujian awal menemukan masalah kekurangan memori.

Untuk menghentikan layanan tanpa menghapus volume:

```bash
docker compose down
```

Hindari `docker compose down -v` jika data persisten masih diperlukan. Pastikan port yang dipublikasikan hanya diakses pada lingkungan praktikum yang tepercaya.

## LK-10: Model Serving dan Horizontal Scaling

Model Random Forest Production v1 dikemas menggunakan MLflow Models
dan dijalankan dalam tiga replika melalui Docker Compose dengan
NGINX sebagai load balancer.

### Mengakses Endpoint API

Pada LK10, image serving menggunakan tag `kalimantan-hotspot-mlflow-serving:lk10`. Konfigurasi Compose terbaru menggunakan image LK11 yang dibangun dari base image LK10. Lihat panduan menjalankan sistem pada bagian LK11.

```bash
docker compose up -d model-serving load-balancer
docker compose ps
```

| Endpoint | Fungsi |
|---|---|
| `GET http://localhost:8002/ping` | Health check |
| `POST http://localhost:8002/invocations` | Inferensi model |

Contoh health check:

```bash
curl -i http://localhost:8002/ping
```

Endpoint inferensi menerima JSON dengan format `dataframe_records`
berisi 13 fitur model. Prediksi `0` berarti NO SPIKE dan `1` berarti
SPIKE. Header `X-Backend` menunjukkan replika yang menangani request.

### Dynamic Scaling

```bash
# Menambah jumlah replika menjadi 4
docker compose up -d --scale model-serving=4 model-serving

# Memeriksa status replika
docker compose ps model-serving

# Mengembalikan jumlah replika menjadi 3
docker compose up -d --scale model-serving=3 model-serving
```

NGINX mendistribusikan request menggunakan Round Robin dan mendeteksi
perubahan jumlah replika melalui DNS internal Docker tanpa restart.

## LK-11: Implementasi Observability dan Dashboard

Prometheus dan Grafana diintegrasikan ke Docker Compose untuk
memantau tiga replika model serving. Scraping metrik dilakukan
setiap 5 detik melalui endpoint `/metrics`.

### Menjalankan Monitoring

Pastikan konfigurasi `.env` dan artefak model telah disiapkan
sesuai panduan LK-09. Jika base image LK-10 belum tersedia,
bangun terlebih dahulu menggunakan MLflow:

```bash
mlflow models build-docker \
  --model-uri "$PWD/docker_assets/model" \
  --name kalimantan-hotspot-mlflow-serving:lk10 \
  --env-manager local
```

Kemudian bangun image LK-11 dan jalankan layanan monitoring:

```bash
docker build -t kalimantan-hotspot-mlflow-serving:lk11 deployment/serving
docker compose up -d --scale model-serving=3 model-serving load-balancer prometheus grafana
```

| Layanan | URL |
|---|---|
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Dashboard Grafana tersedia otomatis melalui provisioning dan
menampilkan latensi, throughput, error rate, CPU, RAM,
distribusi prediksi, serta probabilitas SPIKE.

### Drift Detection dan Pengujian

Data drift diukur menggunakan PSI pada fitur `hotspot_count`,
`delta`, dan `rolling_mean_7` dengan baseline 30.016 observasi training.

Ambil dataset model-ready melalui DVC:

```bash
dvc pull -r dagshub data/processed/hotspot_model_ready.csv.dvc
```
Jalankan pengujian berikut secara berurutan:

```bash
python scripts/load_test.py --duration 120 --workers 4
python scripts/simulate_drift.py --mode normal --duration 180
python scripts/simulate_drift.py --mode drift --duration 180
```

Load test menghasilkan 1.679 request berhasil tanpa kegagalan.
Simulasi drift berhasil mendeteksi pergeseran distribusi
pada ketiga fitur dengan PSI di atas 0,2.

## Keterbatasan

- Sistem masih berupa prototipe praktikum, belum dilengkapi autentikasi API, alerting otomatis yang tervalidasi, dan deployment publik.
- Model menunjukkan *false positive* tinggi dan *distribution shift* pada evaluasi sebelumnya; status Production bukan jaminan siap digunakan secara operasional.
- Model perlu ditarik dari DagsHub sebelum build; database MLflow LK09 tidak otomatis memuat riwayat registry LK07–LK08, dan replikasi dari clone bersih belum diuji menyeluruh.
- Named volumes bertahan saat container dibuat ulang, tetapi tidak menggantikan mekanisme backup pada host lain.

- Deteksi data drift menggunakan PSI merupakan indikator awal perubahan distribusi input, bukan pengukuran langsung penurunan akurasi model.

## Next Steps

- Uji replikasi dari clone baru dan tambahkan pengujian integrasi Compose otomatis.
- Perkuat keamanan endpoint, pengelolaan secrets, backup, dan alerting otomatis.
- Evaluasi kualitas prediksi dan strategi pembaruan model berdasarkan data terbaru.

## License

Proyek menggunakan [MIT License](LICENSE).

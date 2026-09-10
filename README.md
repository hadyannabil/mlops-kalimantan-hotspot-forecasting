# Kalimantan Hotspot Forecasting

Repositori ini merupakan proyek pengembangan sistem machine learning untuk memprediksi tingkat aktivitas hotspot harian pada kabupaten/kota di Kalimantan berdasarkan data **NASA FIRMS (Fire Information for Resource Management System)**.

Proyek dirancang dengan pendekatan **MLOps** agar proses pengembangan machine learning dapat dilakukan secara terstruktur, reproducible, dan mudah dikembangkan pada tahap berikutnya. Repositori ini menjadi fondasi untuk proses pengumpulan data, preprocessing, eksplorasi data, pengembangan model, evaluasi, monitoring, hingga continuous training.

## Tentang Proyek

Aktivitas hotspot dapat mengalami perubahan dari waktu ke waktu sehingga sistem prediksi membutuhkan data yang diperbarui secara berkala. NASA FIRMS menyediakan data observasi active fire dan hotspot yang dapat dimanfaatkan untuk membangun sistem prediksi dengan karakteristik data yang terus bertambah.

Ruang lingkup proyek difokuskan pada wilayah kabupaten/kota di Kalimantan, Indonesia. Data yang diperoleh nantinya akan melalui tahapan pemrosesan sebelum digunakan untuk membangun model machine learning yang dapat memprediksi tingkat aktivitas hotspot harian.

Pada tahap pengembangan saat ini, repositori difokuskan pada penyediaan **infrastruktur dasar proyek**, meliputi konfigurasi lingkungan pengembangan, struktur direktori, dependency management, serta workflow pengembangan menggunakan GitHub.

## Tujuan Proyek

Proyek ini dikembangkan dengan beberapa tujuan utama:

1. Membangun lingkungan pengembangan machine learning yang konsisten dan reproducible menggunakan GitHub Codespaces.
2. Menyusun struktur repositori yang sistematis untuk mendukung pengembangan proyek machine learning.
3. Menerapkan GitHub Flow sebagai workflow pengembangan dan eksperimen.
4. Menyiapkan fondasi untuk proses pengumpulan dan pengolahan data NASA FIRMS.
5. Mengembangkan sistem prediksi tingkat aktivitas hotspot harian pada kabupaten/kota di Kalimantan.
6. Menyiapkan proyek agar dapat dikembangkan menuju implementasi pipeline MLOps seperti training, evaluation, monitoring, dan continuous training.

## Sumber Data

Sumber data utama yang digunakan dalam proyek adalah **NASA FIRMS (Fire Information for Resource Management System)**.

NASA FIRMS menyediakan data active fire dan hotspot yang diperoleh melalui observasi satelit. Data tersebut diperbarui secara berkala sehingga sesuai digunakan pada proyek yang membutuhkan data dinamis dan pengembangan model secara berkelanjutan.

Data mentah yang digunakan selama pengembangan ditempatkan pada direktori:

```text
data/raw/
```

Sedangkan data yang telah melalui tahap preprocessing atau transformasi ditempatkan pada:

```text
data/processed/
```

Dataset berukuran besar tidak disimpan secara langsung ke dalam repositori Git apabila tidak diperlukan. Pendekatan ini digunakan untuk menjaga repositori tetap ringan dan memisahkan source code dari data hasil pengolahan.

## Struktur Proyek

Struktur direktori proyek disusun agar setiap komponen memiliki fungsi yang jelas dan mudah dikembangkan pada tahap berikutnya.

```text
mlops-kalimantan-hotspot-forecasting/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── config/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── environment_test.py
│   └── initial_experiment.py
│
├── tests/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Penjelasan Struktur Direktori

**`.devcontainer/`**
Berisi konfigurasi development container yang digunakan oleh GitHub Codespaces. Konfigurasi ini memastikan proyek dapat dijalankan dengan versi Python, dependency, dan ekstensi pengembangan yang konsisten.

**`config/`**
Digunakan untuk menyimpan file konfigurasi proyek. Pada tahap pengembangan berikutnya folder ini dapat digunakan untuk konfigurasi data pipeline, model, training, maupun parameter lainnya.

**`data/raw/`**
Digunakan untuk menyimpan data mentah yang diperoleh langsung dari sumber data sebelum dilakukan preprocessing.

**`data/processed/`**
Digunakan untuk menyimpan data yang telah melalui proses pembersihan, transformasi, agregasi, atau preprocessing dan siap digunakan pada proses berikutnya.

**`docs/`**
Digunakan untuk menyimpan dokumentasi tambahan yang berkaitan dengan proyek.

**`models/`**
Digunakan sebagai lokasi penyimpanan artefak model machine learning yang dihasilkan dari proses training.

**`notebooks/`**
Digunakan untuk notebook eksperimen, Exploratory Data Analysis (EDA), visualisasi, dan pengujian awal sebelum implementasi dipindahkan menjadi kode yang lebih terstruktur.

**`src/`**
Berisi source code utama proyek seperti proses pengumpulan data, preprocessing, feature engineering, training model, evaluasi, dan komponen pipeline lainnya yang akan dikembangkan secara bertahap.

Saat ini folder `src/` juga memiliki:

* `environment_test.py` untuk memverifikasi lingkungan pengembangan dan dependency utama.
* `initial_experiment.py` sebagai eksperimen sederhana untuk memvalidasi workflow pengembangan proyek.

**`tests/`**
Digunakan untuk menyimpan pengujian terhadap fungsi atau komponen sistem yang dikembangkan.

**`requirements.txt`**
Berisi daftar library Python yang diperlukan oleh proyek.

**`.gitignore`**
Menentukan file atau direktori yang tidak perlu disimpan dan dilacak oleh Git.

## Teknologi yang Digunakan

Lingkungan pengembangan proyek saat ini menggunakan:

* Python 3.12
* Git dan GitHub
* GitHub Codespaces
* Git LFS
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter
* Requests

Konfigurasi GitHub Codespaces juga menyediakan ekstensi pendukung untuk Python, Jupyter, dan GitLens.

## Menjalankan Proyek Menggunakan GitHub Codespaces

GitHub Codespaces merupakan cara yang direkomendasikan untuk menjalankan proyek karena seluruh konfigurasi lingkungan telah didefinisikan pada file `.devcontainer/devcontainer.json`.

### 1. Buka Repository

Buka repository:

`hadyannabil/mlops-kalimantan-hotspot-forecasting`

### 2. Jalankan GitHub Codespaces

Pada halaman utama repository:

1. Pilih tombol **Code**.
2. Buka tab **Codespaces**.
3. Pilih **Create codespace on main**.
4. Tunggu hingga proses pembuatan development container selesai.

GitHub Codespaces akan membaca konfigurasi pada `.devcontainer/devcontainer.json`.

### 3. Instalasi Dependency Otomatis

Setelah Codespace dibuat, dependency proyek akan diinstal secara otomatis melalui perintah:

```bash
python -m pip install -r requirements.txt
```

Proses ini dilakukan oleh konfigurasi `postCreateCommand` sehingga pengguna tidak perlu melakukan instalasi library satu per satu.

### 4. Verifikasi Lingkungan

Untuk memastikan lingkungan pengembangan telah dikonfigurasi dengan benar, jalankan:

```bash
python src/environment_test.py
```

Program akan menampilkan informasi versi Python dan beberapa library utama yang digunakan oleh proyek.

Jika seluruh library dapat di-import tanpa error, lingkungan pengembangan siap digunakan.

### 5. Menjalankan Eksperimen Awal

Eksperimen awal dapat dijalankan menggunakan:

```bash
python src/initial_experiment.py
```

Eksperimen ini menggunakan data sederhana untuk memvalidasi bahwa lingkungan Python, dependency, serta workflow pengembangan dapat berjalan dengan benar.

Eksperimen tersebut belum merepresentasikan model prediksi hotspot final dan hanya digunakan sebagai validasi awal infrastruktur proyek.

## Menjalankan Proyek Secara Lokal

Selain menggunakan GitHub Codespaces, proyek dapat dijalankan pada komputer lokal selama Git dan Python tersedia.

Clone repository terlebih dahulu:

```bash
git clone https://github.com/hadyannabil/mlops-kalimantan-hotspot-forecasting.git
```

Masuk ke direktori proyek:

```bash
cd mlops-kalimantan-hotspot-forecasting
```

Disarankan menggunakan virtual environment agar dependency proyek tidak bercampur dengan environment Python lainnya.

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment pada Windows:

```bash
.venv\Scripts\activate
```

Aktifkan virtual environment pada Linux atau macOS:

```bash
source .venv/bin/activate
```

Kemudian instal seluruh dependency:

```bash
python -m pip install -r requirements.txt
```

Setelah instalasi selesai, lakukan verifikasi:

```bash
python src/environment_test.py
```

Jika tidak terdapat error, environment lokal siap digunakan untuk pengembangan.

## Dependency Proyek

Dependency Python dikelola melalui file `requirements.txt`.

Dependency utama yang tersedia saat ini meliputi:

```text
pandas
numpy
scikit-learn
matplotlib
jupyter
requests
```

Apabila library baru dibutuhkan selama pengembangan proyek, dependency tersebut perlu ditambahkan ke `requirements.txt` agar lingkungan pengembangan tetap reproducible.

## Workflow Pengembangan

Proyek menggunakan pendekatan **GitHub Flow** dalam proses pengembangan.

Branch `main` digunakan untuk menyimpan versi proyek yang stabil, sedangkan pengembangan fitur atau eksperimen dilakukan melalui branch terpisah.

Contoh pembuatan branch:

```bash
git checkout -b feat/nama-fitur
```

Setelah perubahan selesai dilakukan:

```bash
git add .
git commit -m "feat: deskripsi perubahan"
git push origin feat/nama-fitur
```

Selanjutnya perubahan dapat diajukan melalui **Pull Request** untuk ditinjau sebelum digabungkan ke branch `main`.

Pendekatan ini membantu menjaga riwayat pengembangan tetap terstruktur serta memisahkan pekerjaan eksperimen dari versi utama proyek.

## Alur Pengembangan Sistem

Secara umum, proyek direncanakan berkembang melalui alur berikut:

```text
NASA FIRMS
     │
     ▼
Data Ingestion
     │
     ▼
Data Validation
     │
     ▼
Data Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
Model Training
     │
     ▼
Model Evaluation
     │
     ▼
Prediction
     │
     ▼
Monitoring
     │
     ▼
Continuous Training
```

Implementasi setiap tahap akan dilakukan secara bertahap sesuai perkembangan proyek.

## Status Pengembangan

Saat ini proyek berada pada tahap **setup infrastruktur dasar**.

Komponen yang telah disiapkan meliputi:

* Repository GitHub sebagai version control.
* Struktur direktori proyek yang sistematis.
* GitHub Codespaces sebagai reproducible development environment.
* Python 3.12 sebagai lingkungan utama.
* Dependency management melalui `requirements.txt`.
* Dukungan Git LFS.
* Environment validation script.
* Initial experiment untuk validasi workflow.
* Penerapan workflow berbasis branch dan Pull Request.

Tahap berikutnya akan berfokus pada pengembangan komponen data dan machine learning secara bertahap.

## Pengembangan Selanjutnya

Beberapa komponen yang direncanakan untuk dikembangkan pada tahap berikutnya antara lain:

* Pengambilan data NASA FIRMS secara berkala.
* Pemrosesan dan validasi data hotspot.
* Exploratory Data Analysis.
* Feature engineering.
* Pengembangan model machine learning.
* Evaluasi performa model.
* Penyimpanan dan versioning model.
* Monitoring data dan performa model.
* Deteksi data drift.
* Mekanisme continuous training.
* Otomatisasi pipeline MLOps.

## Reproducibility

Salah satu tujuan utama struktur proyek ini adalah memastikan proses pengembangan dapat direproduksi.

Konfigurasi environment disimpan pada `.devcontainer/devcontainer.json`, sedangkan dependency Python disimpan pada `requirements.txt`. Dengan demikian, developer lain dapat membuat environment yang memiliki konfigurasi serupa tanpa melakukan setup secara manual dari awal.

Pendekatan ini juga mempermudah pengembangan proyek secara kolaboratif dan mengurangi perbedaan konfigurasi antar lingkungan.

## Lisensi

Proyek ini menggunakan **MIT License**. Informasi lengkap mengenai ketentuan lisensi tersedia pada file `LICENSE`.

## Pengembang

**Hadyan Nabil Sri Kaloko**

Program Studi Teknik Informatika
Fakultas Ilmu Komputer
Universitas Brawijaya

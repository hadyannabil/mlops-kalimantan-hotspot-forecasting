# Kalimantan Hotspot Forecasting

Proyek machine learning untuk memprediksi **lonjakan jumlah deteksi hotspot dalam tiga hari berikutnya** pada kabupaten/kota di Kalimantan menggunakan data **NASA FIRMS (Fire Information for Resource Management System)**.

Repositori ini dikembangkan dengan struktur yang sistematis dan lingkungan pengembangan yang konsisten menggunakan GitHub Codespaces. Pada tahap saat ini, proyek difokuskan pada perancangan dan pengembangan pipeline data sebagai fondasi untuk proses preprocessing, pemodelan, dan monitoring pada tahap berikutnya.

## Tujuan Proyek

Proyek ini bertujuan untuk:

* Memanfaatkan data hotspot yang diperbarui secara berkala untuk memantau perubahan aktivitas hotspot dari waktu ke waktu.
* Mengembangkan model machine learning yang mampu memprediksi kemungkinan lonjakan jumlah deteksi hotspot dalam tiga hari berikutnya pada setiap wilayah kabupaten/kota di Kalimantan.
* Mendukung proses pemantauan wilayah dengan memberikan informasi mengenai kondisi hotspot yang telah teramati serta indikasi peningkatan aktivitas.
* Menjaga model tetap relevan terhadap perubahan karakteristik data melalui proses monitoring dan pembaruan model secara berkelanjutan.

## Sumber Data

Data utama yang digunakan berasal dari **NASA FIRMS**, khususnya produk **VIIRS NOAA-20**, yang menyediakan data active fire dan thermal anomaly berdasarkan observasi satelit.

Data **Near Real-Time (NRT)** digunakan sebagai sumber data dinamis karena diperbarui secara berkala dan dapat diakses melalui NASA FIRMS Area API. Data historis dapat menggunakan produk Standard Processing (SP) ketika tersedia.

Koordinat setiap deteksi hotspot dipetakan ke wilayah kabupaten/kota menggunakan batas administratif **ADM2 dari geoBoundaries** sebagai data referensi.

Data mentah ditempatkan pada direktori `data/raw/`, sedangkan data yang telah melalui proses pengolahan ditempatkan pada `data/processed/`.

## Struktur Proyek

Struktur repositori disusun untuk memisahkan setiap komponen berdasarkan fungsinya sehingga proses pengembangan lebih terorganisasi dan mudah dipelihara.

```text
mlops-kalimantan-hotspot-forecasting/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── config/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── notebooks/
├── src/
│   ├── environment_test.py
│   └── initial_experiment.py
├── tests/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Penjelasan Direktori

* **`.devcontainer/`**  
  Berisi konfigurasi GitHub Codespaces untuk menyediakan lingkungan pengembangan yang konsisten.

* **`config/`**  
  Digunakan untuk menyimpan file konfigurasi yang diperlukan selama pengembangan sistem.

* **`data/raw/`**  
  Digunakan untuk menyimpan data mentah sebelum melalui proses preprocessing.

* **`data/processed/`**  
  Digunakan untuk menyimpan data yang telah dibersihkan, ditransformasikan, dan disiapkan untuk proses berikutnya.

* **`docs/`**  
  Digunakan untuk menyimpan dokumentasi tambahan proyek.

* **`models/`**  
  Digunakan untuk menyimpan model atau artefak hasil proses training.

* **`notebooks/`**  
  Digunakan untuk Exploratory Data Analysis, eksperimen, dan pengujian awal menggunakan Jupyter Notebook.

* **`src/`**  
  Berisi source code utama proyek. Saat ini terdapat `environment_test.py` untuk menguji kesiapan environment dan `initial_experiment.py` untuk validasi awal proses pengembangan.

* **`tests/`**  
  Digunakan untuk menyimpan pengujian terhadap fungsi atau komponen yang dikembangkan.

* **`requirements.txt`**  
  Berisi daftar dependency Python yang diperlukan oleh proyek.

## Teknologi yang Digunakan

Beberapa teknologi dan library utama yang digunakan atau direncanakan dalam proyek ini meliputi:

* Python 3.12
* GitHub
* GitHub Codespaces
* Git LFS
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter
* Requests
* GeoPandas

## Menjalankan Proyek

Penggunaan **GitHub Codespaces** direkomendasikan karena konfigurasi environment proyek telah tersedia pada `.devcontainer/devcontainer.json`. Dengan cara ini, pengguna tidak perlu melakukan konfigurasi environment secara manual dari awal.

### 1. Buka Repository

Buka repository `mlops-kalimantan-hotspot-forecasting` melalui GitHub.

### 2. Buat Codespace

Pada halaman repository:

1. Klik tombol **Code**.
2. Pilih tab **Codespaces**.
3. Klik **Create codespace on main**.
4. Tunggu hingga proses konfigurasi environment selesai.

GitHub Codespaces akan menggunakan konfigurasi yang terdapat pada `.devcontainer/devcontainer.json`.

### 3. Instalasi Dependency

Dependency proyek dikonfigurasi melalui file `requirements.txt`.

Apabila diperlukan instalasi manual, jalankan:

```bash
python -m pip install -r requirements.txt
```

### 4. Verifikasi Environment

Setelah Codespace siap, jalankan:

```bash
python src/environment_test.py
```

Script tersebut digunakan untuk memastikan Python dan library utama proyek dapat digunakan dengan benar.

Jika seluruh dependency berhasil di-import tanpa error, environment telah siap digunakan untuk pengembangan.

### 5. Menjalankan Eksperimen Awal

Untuk menjalankan eksperimen awal, gunakan:

```bash
python src/initial_experiment.py
```

Eksperimen ini digunakan sebagai validasi awal bahwa konfigurasi environment dan struktur proyek telah berjalan dengan baik. Script ini belum merupakan implementasi final pipeline maupun model prediksi hotspot.

## Pengembangan Selanjutnya

Setelah infrastruktur dasar proyek selesai disiapkan, pengembangan selanjutnya akan difokuskan pada implementasi pipeline data dan machine learning secara bertahap.

Tahapan yang direncanakan meliputi:

* Pengambilan data hotspot NASA FIRMS secara berkala melalui API.
* Pembersihan dan validasi data.
* Pemetaan titik hotspot ke kabupaten/kota menggunakan geoBoundaries ADM2.
* Agregasi data harian pada tingkat kabupaten/kota.
* Penyusunan fitur berdasarkan histori aktivitas hotspot.
* Pengembangan model untuk memprediksi lonjakan jumlah deteksi hotspot dalam tiga hari berikutnya.
* Evaluasi performa model menggunakan metrik yang sesuai.
* Penyimpanan hasil prediksi dan artefak model.
* Monitoring perubahan data dan performa model.
* Pengembangan mekanisme continuous training agar model dapat diperbarui ketika diperlukan.

Implementasi setiap tahap akan dilakukan secara bertahap sesuai perkembangan proyek dan kebutuhan sistem.

## Lisensi

Proyek ini menggunakan **MIT License**. Informasi lebih lanjut tersedia pada file `LICENSE`.

# Kalimantan Hotspot Forecasting

Proyek machine learning untuk memprediksi tingkat aktivitas hotspot harian pada kabupaten/kota di Kalimantan menggunakan data **NASA FIRMS (Fire Information for Resource Management System)**.

Repositori ini dikembangkan dengan struktur yang sistematis dan lingkungan pengembangan yang konsisten menggunakan GitHub Codespaces. Pada tahap saat ini, proyek difokuskan pada penyiapan infrastruktur dasar sebagai fondasi untuk pengembangan pipeline data dan model machine learning pada tahap berikutnya.

## Tujuan Proyek

Proyek ini bertujuan untuk:

* Membangun lingkungan pengembangan yang konsisten dan dapat direproduksi menggunakan GitHub Codespaces.
* Menyusun struktur repositori yang terorganisasi untuk mendukung pengembangan proyek machine learning.
* Menyiapkan fondasi untuk proses pengumpulan, pengolahan, pemodelan, dan evaluasi data hotspot.
* Mengembangkan sistem yang dapat digunakan untuk memprediksi tingkat aktivitas hotspot harian pada wilayah kabupaten/kota di Kalimantan.

## Sumber Data

Data utama yang akan digunakan berasal dari **NASA FIRMS**, yang menyediakan data active fire dan hotspot berdasarkan observasi satelit.

Data tersebut diperbarui secara berkala sehingga sesuai digunakan untuk pengembangan sistem machine learning dengan data yang terus bertambah dan berubah dari waktu ke waktu.

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
  Digunakan untuk menyimpan data yang telah dibersihkan atau ditransformasikan dan siap digunakan pada tahap berikutnya.

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

Beberapa teknologi dan library utama yang digunakan dalam proyek ini meliputi:

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

Eksperimen ini digunakan sebagai validasi awal bahwa konfigurasi environment dan struktur proyek telah berjalan dengan baik. Script ini belum merupakan implementasi model prediksi hotspot final.

## Pengembangan Selanjutnya

Setelah infrastruktur dasar proyek selesai disiapkan, pengembangan selanjutnya akan difokuskan pada implementasi pipeline data dan machine learning secara bertahap.

Tahapan yang direncanakan meliputi:

* Pengambilan data hotspot dari NASA FIRMS secara berkala.
* Pembersihan dan preprocessing data.
* Exploratory Data Analysis untuk memahami pola dan karakteristik data hotspot.
* Penyusunan fitur yang relevan untuk proses pemodelan.
* Pengembangan dan pelatihan model machine learning.
* Evaluasi performa model menggunakan metrik yang sesuai.
* Penyimpanan hasil prediksi dan artefak model.
* Monitoring perubahan data untuk mendeteksi data drift.
* Pengembangan mekanisme continuous training agar model dapat diperbarui ketika data baru tersedia.

Implementasi setiap tahap akan dilakukan secara bertahap sesuai perkembangan proyek dan kebutuhan sistem.

## Lisensi

Proyek ini menggunakan **MIT License**. Informasi lebih lanjut tersedia pada file `LICENSE`.

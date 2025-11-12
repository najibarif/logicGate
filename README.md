# TM Logic Gate (Expo + TensorFlow.js)

Aplikasi Expo React Native untuk klasifikasi gambar gerbang logika (contoh: AND, OR) menggunakan model Teachable Machine dan TensorFlow.js.

## Fitur
- **Kamera**: Ambil foto (native) atau menangkap frame video (web).
- **Inference on-device**: Memuat model `assets/model/model.json` + `weights.bin` untuk prediksi lokal.
- **Multi‑platform**: Android, iOS, dan Web (Expo Router tidak digunakan).

## Prasyarat
- Node.js LTS dan npm/yarn.
- Expo CLI (opsional, karena skrip `npm run` sudah tersedia).
- Perangkat dengan kamera untuk pengujian (atau emulator/simulator yang mendukung kamera).

## Instalasi
1. Install dependensi
   ```bash
   npm install
   ```
2. Pastikan berkas model berada pada lokasi berikut:
   - `assets/model/model.json`
   - `assets/model/weights.bin`
   - `assets/model/metadata.json` (opsional, hanya dokumentasi label)

   Label default di `App.js` saat ini:
   ```js
   const labels = ['OR', 'AND'];
   ```
   Sesuaikan dengan urutan output model jika berbeda.

## Menjalankan Aplikasi
- Jalankan server pengembangan:
  ```bash
  npm start
  ```
- Android (dev build):
  ```bash
  npm run android
  ```
- iOS (dev build, hanya macOS):
  ```bash
  npm run ios
  ```
- Web:
  ```bash
  npm run web
  ```

Saat di web, jika inisialisasi kamera gagal (izin/gesture), tombol "Aktifkan Kamera" akan muncul di preview untuk memulai stream secara manual.

## Struktur Penting
- `App.js` — UI kamera dan pipeline inferensi (resize 224x224, normalisasi 0..1, `model.predict`).
- `assets/model/` — berkas model Teachable Machine (`model.json`, `weights.bin`).
- `app.json` — konfigurasi Expo (ikon, splash, bundle id Android, dsb.).
- `package.json` — skrip dan dependensi.

## Dependensi Utama
- `expo`, `react`, `react-native`
- `expo-camera`, `expo-file-system`, `expo-image-picker`, `expo-gl`
- `@tensorflow/tfjs` dan `@tensorflow/tfjs-react-native`

Catatan: Backend TFJS akan dipilih otomatis oleh lingkungan (web/native). Pada native, paket `@tensorflow/tfjs-react-native` dibutuhkan untuk I/O bundel.

## Mengganti/Update Model
1. Ekspor ulang model dari Teachable Machine (Image project) dengan format TensorFlow.js.
2. Ganti berkas di `assets/model/` dengan nama yang sama: `model.json` dan `weights.bin`.
3. Jika jumlah/urutan label berubah, perbarui konstanta `labels` di `App.js` agar sesuai index output model.

## Troubleshooting
- **Model tidak termuat (Native/Web)**: Pastikan path `assets/model/model.json` dan `weights.bin` benar serta ikut ter-bundle. Periksa log: "Model loaded (web/native)" atau peringatan terkait.
- **Izin kamera ditolak**: Berikan izin di perangkat/emulator. Di web, domain harus HTTPS (kecuali localhost) untuk akses kamera.
- **Prediksi selalu unknown/0**: Pastikan ukuran input dan normalisasi sesuai (default 224x224, dibagi 255). Cek urutan label.
- **Web tidak menampilkan video**: Klik "Aktifkan Kamera" untuk memulai stream jika browser memerlukan user gesture.

## Lisensi
Lisensi: 0BSD (lihat `package.json`).

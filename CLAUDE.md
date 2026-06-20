# CLAUDE.md — Proyek VisionGuard+ AI
> Satria Data 2026 | Statistics Essay Competition (SEC) | Sub-tema: Kesehatan
> Fokus: VisionGuard+ — sistem prediksi risiko kebutaan berbasis AI

---

## 1. IDENTITAS PROYEK

| Item | Detail |
|---|---|
| Nama sistem | VisionGuard+ AI |
| Kompetisi | Statistics Essay Competition (SEC) — Satria Data 2026 |
| Penyelenggara | Kemdiktisaintek melalui BPTI |
| Sub-tema | **Kesehatan** |
| Tema besar | "Integrasi Sains, Teknologi, dan Industri dalam Menghadapi Tren Sains Data di Era Society 5.0" |
| Final | Universitas Diponegoro, Semarang |
| Output | Esai ilmiah PDF 2.000–3.000 kata |
| Deadline proyek | 20 Juni 2026 |
| Deadline esai | 30 Juni 2026 |

---

## 2. JUDUL

```
VisionGuard+: Sistem Prediksi Risiko Kebutaan Berbasis 
Explainable Multi-Disease CNN dan Survival Analysis 
untuk Pemetaan Epidemiologi dan Prioritas 
Skrining Penglihatan Nasional
```

---

## 3. LATAR BELAKANG MASALAH

- **10,2 juta** penderita diabetes Indonesia (IDF 2023), 35% berisiko Retinopati Diabetik (DR)
- **80% glaukoma** tidak terdeteksi dini (WHO 2022)
- **10.000+ Puskesmas** tanpa dokter spesialis mata
- Rasio dokter mata: **1:250.000** (WHO rekomendasikan 1:50.000)
- Kamera fundus: **Rp 100–500 juta** — tidak terjangkau Puskesmas

---

## 4. ARSITEKTUR SISTEM — 3 STAGE

```
STAGE 1: DETEKSI (CNN)
  Input  : Foto fundus retina
  Model  : EfficientNet-B4 + SE-Block + 4 output head
  Output : DR Grade, Glaukoma, AMD, Katarak + Grad-CAM heatmap
  Untuk  : Kader Puskesmas (RUJUK / PANTAU / AMAN)

STAGE 2: PREDIKSI PROGRESI (SURVIVAL ANALYSIS)
  Input  : Output CNN + data klinis (usia, HbA1c, tekanan darah)
  Model  : Cox Proportional Hazard Regression
  Output : Risk score 0-100 + prediksi waktu ke kebutaan
  Contoh : "Tanpa intervensi → kebutaan dalam 18 bulan"

STAGE 3: PEMETAAN EPIDEMIOLOGI (OPSIONAL)
  Input  : Riskesdas 2023 + BPS + data dokter mata Perdami
  Metode : Spatial regression + Moran's I
  Output : Peta choropleth 514 kab/kota prioritas skrining
```

---

## 5. GENUINE NOVELTY (6 POIN)

| # | Novelty | Status |
|---|---|---|
| 1 | Multi-disease CNN + Grad-CAM untuk kader Puskesmas | Belum ada di Indonesia |
| 2 | Explainable output — rekomendasi aksi bukan angka | Belum ada |
| 3 | Survival Analysis untuk prediksi WAKTU ke kebutaan | Belum digabung CNN retina manapun |
| 4 | Peta epidemiologi nasional 514 kab/kota | Belum ada |
| 5 | Laporan klinis otomatis untuk alur rujukan | Belum ada |
| 6 | Sistem 3-stage terintegrasi (individu → klinis → nasional) | Belum ada |

---

## 6. DATASET

| Dataset | Sumber | Isi | Prioritas |
|---|---|---|---|
| ODIR-5K | Kaggle (andrewmvd) | 5.000 pasien, 8 label penyakit, multi-label | **UTAMA** |
| EyePACS (resized) | Kaggle | ~35.000 gambar, DR Grade 0-4 | Tambahan |
| APTOS 2019 | Kaggle | 3.662 gambar DR | Tambahan |
| RFMiD 2.0 | IEEE DataPort | 3.200 gambar, 46 kondisi | Tambahan |

**Fokus awal: ODIR-5K saja** (feasible di Colab Free dalam 6 hari)

---

## 7. TECH STACK

| Komponen | Tools |
|---|---|
| CNN Training | PyTorch + timm (EfficientNet-B4) |
| Grad-CAM | pytorch-grad-cam |
| Survival Analysis | lifelines (CoxPHFitter) |
| Geospasial (opsional) | geopandas + folium |
| Laporan klinis | reportlab |
| Platform training | Google Colab Free (T4 GPU) |

---

## 8. PRIORITAS PENGERJAAN (14–20 Juni 2026)

### WAJIB
- [ ] Download ODIR-5K di Kaggle / Colab
- [ ] Training EfficientNet-B4 multi-head → dapat AUC nyata
- [ ] Implementasi Grad-CAM → contoh gambar heatmap
- [ ] Cox Regression sederhana → kurva Kaplan-Meier

### OPSIONAL (jika waktu cukup setelah 18 Juni)
- [ ] Peta geospasial Indonesia dari Riskesdas + BPS
- [ ] Laporan klinis otomatis PDF

---

## 9. POSITIONING vs KOMPETITOR

| | Voxify 2025 (Juara 1 SEC) | CNN Retina Biasa | VisionGuard+ |
|---|---|---|---|
| Multi-disease | Tidak | Tidak | Ya (4 sekaligus) |
| Explainable untuk kader | Tidak | Tidak | Ya (Grad-CAM) |
| Prediksi temporal | Tidak | Tidak | Ya (Cox Regression) |
| Peta nasional | Tidak | Tidak | Ya (514 kab/kota) |

---

## 10. KALIMAT KUNCI UNTUK JURI

**Pembuka:**
> "80% penderita glaukoma di Indonesia tidak tahu mereka sakit. VisionGuard+ hadir untuk mengubah itu."

**Beda dari Voxify:**
> "Voxify mendeteksi penyakit individu. VisionGuard+ bergerak tiga level: deteksi, prediksi kapan kritis, dan peta prioritas nasional."

**Penutup:**
> "VisionGuard+ adalah kompas berbasis data untuk Kemenkes: di mana kebutaan paling mengancam dan di mana intervensi harus dimulai hari ini."

---

## 11. FILE PENTING DI PROJECT

| File | Isi |
|---|---|
| `memoryccnn.md` | Detail lengkap VisionGuard+ (arsitektur, kode, referensi) |
| `VisionGuard_Laporan_Tim.pdf` | Laporan tim untuk presentasi |
| `generate_laporan.py` | Script Python untuk generate PDF laporan |

---

*Dokumen ini fokus 100% pada VisionGuard+ AI untuk SEC Satria Data 2026.*

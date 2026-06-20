# VisionGuard+ AI — SEC Satria Data 2026

Sistem prediksi risiko kebutaan berbasis Explainable Multi-Disease CNN dan Survival Analysis untuk pemetaan epidemiologi dan prioritas skrining penglihatan nasional.

> **Kompetisi:** Statistics Essay Competition (SEC) — Satria Data 2026 | Sub-tema: Kesehatan
> **Dataset:** ODIR-5K (6.392 gambar fundus retina) | **Platform:** Google Colab T4 GPU

---

## Dokumen

| File | Isi |
|---|---|
| [HASIL_VISIONGUARD.md](HASIL_VISIONGUARD.md) | Analisis lengkap hasil training, metrik evaluasi, Grad-CAM, dan Risk Stratification — bahan utama esai SEC |
| [VisionGuard_Colab.ipynb](VisionGuard_Colab.ipynb) | Notebook training lengkap untuk Google Colab (EfficientNet-B4, Grad-CAM, ROC, Risk Stratification) |
| [CLAUDE.md](CLAUDE.md) | Dokumentasi proyek: arsitektur sistem, novelty, dataset, dan prioritas pengerjaan |
| [memoryccnn.md](memoryccnn.md) | Detail teknis arsitektur VisionGuard+ (kode, referensi, metodologi) |

---

## Hasil Utama

| Metrik | Nilai |
|---|---|
| AUC-ROC rata-rata | **0.8887** (Good) |
| AUC Katarak | **0.9272** (Excellent) |
| Presisi RUJUK | **97.4%** |
| Efisiensi rujukan | **95.9%** pengurangan beban spesialis |
| Best epoch | 5 / 20 (early stopping) |
| Validation set | 959 gambar retina |

## Arsitektur Sistem

```
Input: Foto fundus retina
  ↓
Stage 1: EfficientNet-B4 (DR, Glaukoma, Katarak, AMD) + Grad-CAM
  ↓
Stage 2: Cox Regression → prediksi waktu ke kebutaan
  ↓
Output: AMAN / PANTAU / RUJUK
```

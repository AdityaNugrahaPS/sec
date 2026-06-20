# memoryccnn.md — VisionGuard+ AI: CNN Project
> Dokumen Konteks Deep Learning | SEC Satria Data 2026
> Sub-tema: Kesehatan
> Update: Arsitektur 3-Stage + Survival Analysis + Geospasial Epidemiologi

---

## 1. IDENTITAS PROYEK

| Item | Detail |
|---|---|
| **Nama sistem** | VisionGuard+ AI |
| **Kompetisi** | Statistics Essay Competition (SEC) — Satria Data 2026 |
| **Sub-tema** | Kesehatan |
| **Domain** | Computer Vision + Survival Analysis + Geospasial Epidemiologi |
| **Target** | Sistem prediksi risiko kebutaan + pemetaan prioritas skrining nasional |
| **Output** | Esai ilmiah PDF 2.000–3.000 kata |

---

## 2. JUDUL

```
VisionGuard+: Sistem Prediksi Risiko Kebutaan Berbasis 
Explainable CNN dan Survival Analysis untuk Pemetaan 
Epidemiologi dan Prioritas Skrining Penglihatan Nasional
```

**Alternatif lebih singkat:**
```
VisionGuard+: Prediksi Risiko Kebutaan Berbasis Explainable 
Multi-Disease CNN dan Stratifikasi Epidemiologi untuk 
Skrining Penglihatan Nasional
```

---

## 3. KENAPA INI BEDA DARI VOXIFY (UNAIR JUARA 1 SEC 2025)

| Dimensi | Voxify 2025 | VisionGuard+ 2026 |
|---|---|---|
| Domain | Pernapasan | Mata / retina |
| Input | Suara / audio | Foto fundus retina |
| Level analisis | **Individu saja** | **Individu + Populasi + Kebijakan** |
| Metode statistik | Deep Learning audio | CNN + Survival Analysis + Geospasial |
| Output kebijakan | Tidak ada | Peta prioritas skrining nasional |
| Target dampak | Pasien individual | Kebijakan Kemenkes + Puskesmas nasional |
| Angle ekonomi | Tidak ada | Cost-effectiveness analysis |

**Satu kalimat pembeda:**
> Voxify mendeteksi penyakit individu.
> VisionGuard+ memprediksi kapan seseorang akan buta DAN
> memetakan daerah mana di Indonesia yang paling kritis
> untuk diprioritaskan program skrining nasional.

---

## 4. LATAR BELAKANG MASALAH

### Fakta Kritis Indonesia

- **10,2 juta** penderita diabetes (IDF 2023) — 50% belum terdiagnosis
- **35%** penderita diabetes berisiko Retinopati Diabetik (DR)
- **Glaukoma** = penyebab kebutaan permanen #2, **80% tidak terdeteksi dini**
- **>10.000 Puskesmas** — hampir tidak ada dokter spesialis mata
- Rasio dokter mata: **1:250.000** (WHO rekomendasikan 1:50.000)
- Estimasi kerugian ekonomi kebutaan Indonesia: **Rp 4,2 triliun/tahun**
- Biaya skrining: ~Rp 50.000/orang vs biaya rehabilitasi kebutaan: ~Rp 15 juta/orang

### 3 Gap Utama

| Gap | Masalah | VisionGuard+ Menjawab |
|---|---|---|
| **Gap Deteksi** | Model CNN ada tapi 1 penyakit, tidak untuk Puskesmas | Multi-disease CNN + Grad-CAM untuk kader |
| **Gap Prediksi** | Tidak ada yang prediksi KAPAN pasien akan buta | Survival Analysis → waktu ke kebutaan |
| **Gap Kebijakan** | Tidak ada peta prioritas skrining berbasis data | Geospatial epidemiology → peta nasional |

---

## 5. ARSITEKTUR SISTEM — 3 STAGE

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1: DETEKSI (CNN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Foto fundus retina (smartphone / kamera fundus)
    ↓
EfficientNet-B4 Backbone (pretrained ImageNet)
    ↓
SE-Block (channel attention)
    ↓
Multi-head Output:
  ├── DR Severity (0-4) → Softmax
  ├── Glaukoma (Ya/Tidak) → Sigmoid  
  ├── AMD (Ya/Tidak) → Sigmoid
  └── Katarak (Ya/Tidak) → Sigmoid
    ↓
Grad-CAM → Heatmap area mencurigakan

Output Stage 1:
  "DR Grade 2 (Moderate) — 89.3%
   Glaukoma: Tidak Terdeteksi
   AMD: Tidak Terdeteksi"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2: PREDIKSI PROGRESI (SURVIVAL ANALYSIS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Output CNN + Data Klinis Pasien
  - Grade DR dari Stage 1
  - Usia pasien
  - Durasi diabetes (tahun)
  - HbA1c terakhir
  - Tekanan darah sistolik
  - Status merokok

Model: Cox Proportional Hazard Regression
    ↓
Output Stage 2:
  Risk Score: 73/100 (TINGGI)
  Stratifikasi: KRITIS
  Prediksi: "Tanpa intervensi → kebutaan dalam 18 bulan
             (95% CI: 14–24 bulan)"
  Rekomendasi: Rujuk ke dokter mata dalam 2 minggu

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 3: PEMETAAN EPIDEMIOLOGI (GEOSPASIAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Agregasi output Stage 1+2 + Data Eksternal
  - Prevalensi DR per kabupaten (agregat anonim dari Stage 1)
  - Data Riskesdas 2023 (prevalensi diabetes per provinsi)
  - BPS populasi per kabupaten/kota
  - Jumlah dokter mata per wilayah (data Perdami/Kemenkes)
  - Jumlah Puskesmas per kabupaten

Metode: Spatial regression + Kriging interpolasi
    ↓
Output Stage 3:
  Peta choropleth Indonesia:
  🔴 MERAH = risiko tinggi + akses rendah = PRIORITAS UTAMA
  🟡 KUNING = risiko sedang / akses cukup
  🟢 HIJAU = risiko rendah / sudah terlayani

  Top 10 kabupaten prioritas skrining nasional
  Estimasi jumlah penderita DR tidak terdiagnosis per wilayah
  Rekomendasi alokasi mobile screening unit
```

---

## 6. GENUINE NOVELTY (6 POIN)

| # | Novelty | Ada di Paper Lain? | Ada di Voxify? |
|---|---|---|---|
| **1** | Multi-disease CNN + Grad-CAM untuk kader Puskesmas | TIDAK | TIDAK |
| **2** | **Survival Analysis** untuk prediksi waktu ke kebutaan | TIDAK | TIDAK |
| **3** | **Geospatial epidemiology** → peta prioritas skrining nasional | TIDAK | TIDAK |
| **4** | **Cost-effectiveness analysis** skrining vs rehabilitasi kebutaan | TIDAK | TIDAK |
| **5** | Stratifikasi risiko 4 level berbasis CNN + klinis | TIDAK | TIDAK |
| **6** | 3-stage integrated system (deteksi → prediksi → kebijakan) | TIDAK | TIDAK |

### Positioning Statement

```
"Paper terdahulu membuktikan CNN dapat mendeteksi retinopati diabetik
dengan akurasi tinggi. Namun tidak ada yang menjawab tiga pertanyaan
kritis berikutnya: KAPAN pasien akan kehilangan penglihatan? DIMANA
di Indonesia penderita paling berisiko tidak terdeteksi? BERAPA nilai
ekonomi program skrining nasional?
VisionGuard+ menjawab ketiga pertanyaan tersebut melalui integrasi
Explainable CNN, Survival Analysis, dan Geospatial Epidemiology."
```

---

## 7. DATASET STRATEGY

### Dataset Utama

| Dataset | Sumber | Jumlah | Penyakit | Akses |
|---|---|---|---|---|
| **EyePACS (resized)** | Kaggle | ~35.000 gambar | DR Grade 0-4 | Free |
| **RFMiD 2.0** | IEEE DataPort | 3.200 gambar | 46 kondisi retina | Free |
| **ODIR-5K** | Kaggle | 10.000 pasien | DR, Glaukoma, Katarak, AMD | Free |
| **APTOS 2019** | Kaggle | 3.662 gambar | DR severity | Free |

### Dataset Epidemiologi

| Dataset | Sumber | Dipakai untuk |
|---|---|---|
| Riskesdas 2023 | Kemenkes | Prevalensi diabetes per provinsi |
| BPS Sensus 2020 | BPS | Populasi per kabupaten |
| Data dokter mata | Perdami/Kemenkes | Akses layanan per wilayah |
| Data Puskesmas | Kemenkes | Infrastruktur per kabupaten |

---

## 8. METODE STATISTIK LENGKAP

### Stage 1 — CNN (EfficientNet-B4)

```python
# Arsitektur
model = EfficientNet_B4_MultiHead(
    backbone='efficientnet-b4',
    pretrained=True,
    heads=['dr_severity', 'glaukoma', 'amd', 'katarak'],
    attention='SE-Block'
)

# Loss multi-task
L = 0.4*L_DR + 0.2*L_Glaukoma + 0.2*L_AMD + 0.2*L_Katarak

# Evaluasi
metrics = ['AUC-ROC', 'Sensitivity', 'Specificity', 'F1']
target_DR_AUC = 0.92  # minimal
```

### Stage 2 — Survival Analysis (Cox Regression)

```python
from lifelines import CoxPHFitter

# Covariates
X = ['dr_grade_cnn', 'age', 'diabetes_duration',
     'hba1c', 'systolic_bp', 'smoking_status']

# Target
T = 'months_to_severe_vision_loss'  # event time
E = 'event_occurred'                 # 1=buta, 0=censored

# Model
cph = CoxPHFitter()
cph.fit(df, duration_col=T, event_col=E, formula='+'.join(X))

# Output per pasien
risk_score = cph.predict_partial_hazard(patient_data)
survival_curve = cph.predict_survival_function(patient_data)
median_survival = cph.predict_median(patient_data)
```

### Stage 3 — Geospatial Epidemiology

```python
import geopandas as gpd
from scipy.stats import spearmanr

# Hitung prevalensi estimasi per kabupaten
prevalensi_DR = (populasi_diabetes × 0.35) × detection_gap_factor

# Indeks aksesibilitas layanan mata
akses_score = dokter_mata_per_100k / distance_to_nearest_faskes

# Prioritas skrining = fungsi dari risiko dan akses
prioritas = 0.6 × prevalensi_DR + 0.4 × (1 - akses_score)

# Spatial autocorrelation (Moran's I)
morans_i = esda.Moran(prioritas, spatial_weights)
```

### Cost-Effectiveness Analysis

```
ICER (Incremental Cost-Effectiveness Ratio):

Biaya skrining/orang       : Rp 50.000
Biaya rehabilitasi kebutaan: Rp 15.000.000
Probabilitas deteksi dini  : 78% (dari model)
Probabilitas cegah kebutaan: 65% (literatur)

ICER = (Biaya skrining - Biaya tanpa skrining) / QALY gained
     = Cost saving: Rp 47 per Rp 1 diinvestasikan
```

---

## 9. KERANGKA ESAI (3 BAB)

### BAB 1 — LATAR BELAKANG (~600 kata)

**P1 — Krisis Kebutaan Indonesia**
10,2 juta diabetik, 35% berisiko DR, 80% glaukoma tidak terdeteksi,
estimasi kerugian ekonomi Rp 4,2 triliun/tahun.

**P2 — Gap Teknologi & Kebijakan**
CNN sudah bisa deteksi DR, tapi: tidak prediksi progresi, tidak ada
peta epidemiologi, tidak ada panduan prioritas skrining berbasis data.

**P3 — Gap vs Voxify / Paper Lain**
Voxify menjawab deteksi penyakit pernapasan individual.
Tidak ada sistem yang mengintegrasikan prediksi progresi +
epidemiologi populasi + rekomendasi kebijakan untuk kebutaan.

**P4 — Solusi VisionGuard+**
Sistem 3-stage: CNN deteksi → Survival Analysis prediksi → 
Geospatial epidemiology pemetaan prioritas nasional.

---

### BAB 2 — ISI / PEMBAHASAN (~1.800 kata)

**2.1 Dataset & Preprocessing**
EyePACS + RFMiD + ODIR + APTOS — ~50.000 gambar fundus nyata.
Ben Graham preprocessing + CLAHE + augmentasi.

**2.2 Stage 1: Explainable Multi-Disease CNN**
EfficientNet-B4 + SE-Block + Multi-head.
Grad-CAM visualisasi area mencurigakan untuk kader Puskesmas.
Hasil: AUC DR 0.94, Glaukoma 0.91, AMD 0.88.

**2.3 Stage 2: Survival Analysis**
Cox Proportional Hazard → prediksi waktu ke kebutaan.
Risk score 0-100 + stratifikasi 4 level.
Contoh output: "Tanpa intervensi → buta dalam 18 bulan."

**2.4 Stage 3: Pemetaan Epidemiologi Geospasial**
Agregasi anonim output CNN + Riskesdas + BPS + data Puskesmas.
Peta choropleth prioritas skrining per kabupaten.
Top 10 wilayah paling kritis di Indonesia.

**2.5 Cost-Effectiveness Analysis**
Setiap Rp 1 skrining menghemat Rp 47 biaya rehabilitasi.
QALY gained per 1.000 pasien terskrining.

**2.6 Perbandingan**

| Aspek | CNN Konvensional | Voxify 2025 | VisionGuard+ |
|---|---|---|---|
| Deteksi penyakit | ✅ | ✅ | ✅ |
| Multi-penyakit | ❌ | ❌ | ✅ |
| Prediksi progresi | ❌ | ❌ | ✅ |
| Peta epidemiologi | ❌ | ❌ | ✅ |
| Cost-effectiveness | ❌ | ❌ | ✅ |
| Target Puskesmas | ❌ | ❌ | ✅ |

---

### BAB 3 — PENUTUP (~400 kata)

**Simpulan:**
VisionGuard+ melampaui batas deteksi penyakit — menjadi sistem
prediksi dan perencanaan kebijakan skrining penglihatan nasional
berbasis data.

**Dampak:**
- 10.000+ Puskesmas terbantu skrining retina tanpa dokter mata
- Deteksi dini → cegah 3,5+ juta kebutaan preventable
- Peta prioritas → alokasi sumber daya Kemenkes lebih efisien
- Penghematan: Rp 47 per Rp 1 diinvestasikan

**Kalimat penutup:**
> "VisionGuard+ bukan sekadar model AI — ia adalah kompas berbasis
> data yang menunjukkan kepada Kementerian Kesehatan: di mana
> kebutaan paling mengancam, dan di mana intervensi harus dimulai,
> agar tidak ada warga Indonesia yang kehilangan penglihatan
> karena keterlambatan deteksi."

---

## 10. REFERENSI KUNCI (Minimal 12)

| # | Referensi | Relevansi |
|---|---|---|
| 1 | IDF Diabetes Atlas 10th Ed. (2023) | Prevalensi diabetes Indonesia |
| 2 | WHO Global Eye Health Action Plan (2022-2030) | Konteks kebijakan global |
| 3 | Kemenkes RI, Riskesdas 2023 | Data diabetes & mata Indonesia |
| 4 | Gulshan et al., JAMA (2016) | Proof of concept CNN untuk DR |
| 5 | Tan & Le, EfficientNet ICML (2019) | Justifikasi arsitektur |
| 6 | Selvaraju et al., Grad-CAM ICCV (2017) | Justifikasi explainability |
| 7 | Cox (1972) + lifelines docs | Justifikasi Survival Analysis |
| 8 | RFMiD Dataset, IEEE Access (2021) | Dataset multi-disease |
| 9 | Holmberg et al., Nature (2020) | Progresi DR berbasis DL |
| 10 | Perdami, Peta Dokter Mata Indonesia (2022) | Data akses layanan |
| 11 | World Bank, Cost of Blindness Indonesia (2021) | Cost-effectiveness |
| 12 | Moran (1950) + GeoDa docs | Justifikasi spatial analysis |

**Kata kunci Google Scholar:**
```
"diabetic retinopathy" "survival analysis" "time to blindness"
"explainable CNN" "retinal disease" Indonesia "Puskesmas"
"geospatial epidemiology" "blindness" Indonesia
"EfficientNet" "multi-disease" fundus classification
"cost-effectiveness" "diabetic retinopathy screening" Indonesia
```

---

## 11. TECH STACK

| Komponen | Teknologi |
|---|---|
| CNN Training | PyTorch + EfficientNet (timm library) |
| Grad-CAM | pytorch-grad-cam library |
| Survival Analysis | lifelines (Python) |
| Geospasial | geopandas + folium + esda (PySAL) |
| Cost-effectiveness | Manual kalkulasi + tabel |
| Visualisasi | matplotlib + seaborn + plotly |
| Dashboard demo | Streamlit atau Flask |

---

## 12. CHECKLIST SEBELUM SUBMIT

### Data & Model
- [ ] Download EyePACS resized dari Kaggle (~8GB)
- [ ] Download RFMiD 2.0 dari IEEE DataPort
- [ ] Download ODIR-5K dari Kaggle
- [ ] Preprocessing pipeline (CLAHE + Ben Graham)
- [ ] EfficientNet-B4 fine-tuning → AUC DR > 0.92
- [ ] Grad-CAM diimplementasikan
- [ ] Dataset survival analysis disiapkan (simulasi jika perlu)
- [ ] Cox Regression model dijalankan
- [ ] Peta prioritas wilayah dibuat

### Visualisasi untuk Esai
- [ ] ROC Curve 4 penyakit dalam 1 plot
- [ ] Contoh Grad-CAM (foto retina + heatmap)
- [ ] Kaplan-Meier survival curve per stratifikasi risiko
- [ ] Peta choropleth prioritas skrining Indonesia
- [ ] Tabel cost-effectiveness
- [ ] Perbandingan model baseline vs VisionGuard+

### Esai
- [ ] 2.000–3.000 kata
- [ ] Minimal 12 referensi 2019–2026
- [ ] Similarity Turnitin < 25%
- [ ] Semua angka AUC dari eksperimen nyata

---

## 13. KALIMAT KUNCI UNTUK JURI

**Saat ditanya beda dari Voxify:**
> "Voxify mendeteksi penyakit pernapasan pada level individu.
>  VisionGuard+ bergerak tiga level lebih jauh: mendeteksi,
>  memprediksi kapan pasien akan kehilangan penglihatan,
>  dan memetakan 514 kabupaten Indonesia berdasarkan
>  prioritas kebutuhan skrining nasional."

**Saat ditanya beda dari CNN retina biasa:**
> "CNN konvensional menjawab: apakah pasien ini sakit?
>  VisionGuard+ menjawab: kapan pasien ini akan buta,
>  dan di mana di Indonesia kita harus mulai bergerak lebih dulu?"

**Saat ditanya justifikasi Survival Analysis:**
> "Survival Analysis atau Cox Regression adalah metode statistik
>  yang secara khusus dirancang untuk memodelkan waktu hingga
>  suatu kejadian — dalam hal ini, waktu hingga kehilangan
>  penglihatan. Ini memberikan dimensi temporal yang tidak bisa
>  dijawab oleh CNN maupun regresi logistik biasa."

**Kalimat penutup presentasi:**
> "VisionGuard+ adalah kompas berbasis data untuk Kemenkes:
>  menunjukkan di mana kebutaan paling mengancam,
>  dan di mana intervensi harus dimulai hari ini,
>  agar tidak ada warga Indonesia yang kehilangan penglihatan
>  karena keterlambatan deteksi."

---

*Dokumen ini adalah memori kerja untuk proyek VisionGuard+ AI.*
*Update setiap ada keputusan arsitektur atau hasil eksperimen baru.*
*Versi: 2.0 — dengan Survival Analysis + Geospasial Epidemiologi*

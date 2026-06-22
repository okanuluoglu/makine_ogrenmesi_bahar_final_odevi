# Radyomik Özelliklerle Papilodem İkili Sınıflandırması

**Makine Öğrenmesi Final Projesi - Teslim Paketi**

\---

## Özet Sonuç

* **Ortalama macro-F1 lideri:** ExtraTrees (0.847 +/- 0.060) - Ensemble (0.847 +/- 0.066) ile esit basta
* **En iyi tek tekrar:** Gradient Boosting, repeat 19, macro-F1 = 0.941
* **Friedman testi:** chi-kare = 22.62, p = 0.0009 (cok yuksek anlamlilik)
* **Toplam calisma suresi:** 85 dakika (20 outer repeat x 50 trial x 6 model + ensemble)
* **Hasta-bazli 70/10/20 uclu bolme:** Train 70%, Validation 10%, Test 20% - hicbir hasta birden fazla sete dusmez
* **Veri sizintisiz:** Tum CV donguleri patient-level, kalibrasyon hasta-bilincli (custom GroupAwareSigmoidCalibrator), uc adet pre-CV integrity assertion

\---

## Klasör Yapısı

|Klasör|İçerik|Spec Madde|
|-|-|-|
01_RAPOR/|Final rapor (DOCX)  
02_KOD/|Python kaynak kod (src, scripts, tests), requirements, README|madde 16|
03_NOTEBOOK/|Jupyter notebook (executed - tablo ve grafiklerle)|madde 16|
04_GRAFIKLER/|17 PNG grafik (300 DPI, A4 baski uyumlu)|madde 13|
05_SONUC_TABLOLARI/|CSV ve JSON sonuc tablolari|madde 16|
06_VERI/|Orijinal CSV veri dosyalari (normal + papilodem)|madde 2|

\---

## Detaylar (Spec Madde Eşleşmesi)

### Madde 5 - Veri Ön İşleme (10 puan)

* Median imputation: `02_KOD/src/preprocessing.py` (SimpleImputer)
* Low-variance filtering (esik=0.01): VarianceThreshold
* Pearson korelasyon |r|>0.95 elenmesi: CorrelationFilter (custom)
* RobustScaler ile olcekleme
* **Tamami sklearn Pipeline içinde, sadece eğitim verisinde fit edilir**

### Madde 6 - Özellik Seçimi (10 puan)

* MRMR uygulaması: `02_KOD/src/feature_selection.py`
* Relevance: Mutual Information
* Redundancy: Pearson korelasyonu
* k tunable: {20, 50, 100} (Optuna icinde)

### Madde 7 - Modelleme (20 puan)

6 temel sınıflandırıcı (`02_KOD/src/models.py`):

* LR (Lojistik Regresyon, saga, balanced)
* SVM (RBF kernel, balanced, probability=True)
* RF (Rastgele Orman, balanced\_subsample)
* ET (ExtraTrees, balanced\_subsample)
* GB (Gradient Boosting)
* KNN (K-En Yakin Komsu)

### Madde 8 - Hiperparametre Optimizasyonu (15 puan)

* Optuna (`02_KOD/src/optimization.py`)
* TPE sampler, seed=42
* Model basina 50 trial (spec minimumu)
* Amac: ortalama macro-F1

### Madde 9 - Çapraz Doğrulama

* Dış döngü: **GroupShuffleSplit ile 20 outer repeat**, hasta-bazlı 70/10/20 üçlü bölme
* İç döngü: StratifiedGroupKFold(5) Optuna her trial içinde
* Validation seti: hasta-bazlı ayrık, F1-maksimize eşik seçimi için
* Test seti: yalnızca nihai metrik raporlamada kullanıldı
* Aynı hasta asla iki sete düşmez (uc adet `isdisjoint` assertion her tekrarda)

### Madde 10 - Kalibrasyon

* Sigmoid (Platt) calibration
* **Sklearn 1.6 metadata routing bug'i icin manuel implementasyon**:
`02_KOD/src/calibration.py` -> `GroupAwareSigmoidCalibrator`

### Madde 11 - Ensemble

* Soft voting (`02_KOD/src/ensemble.py`)
* Üyeler: kalibre edilmiş RF + ET + GB

### Madde 12 - Performans Metrikleri (15 puan)

9 metrigin tamami hesaplandı:
Accuracy, Precision, Recall, F1, Macro-F1, ROC-AUC, PR-AUC, Balanced Accuracy, Brier

* Tablolar: `05_SONUC_TABLOLARI/metrics_summary.csv`

### Madde 13 - Grafikler (10 puan)

Zorunlu 6 grafik + ek grafikler:

1. ROC Curve - `04_GRAFIKLER/roc_curves.png`
2. Precision-Recall Curve - `pr_curves.png`
3. Confusion Matrix - `confusion_matrix.png`
4. Feature Importance - `feature_importance.png`
5. Calibration Curve - `calibration_curves.png`
6. Model Karşılaştırma - `model_comparison_macro_f1.png`, `model_comparison_roc_auc.png`

Ek grafikler:

* `shap_summary.png` - SHAP TreeExplainer özet grafiği
* `feature_stability.png` - MRMR seçim frekansı (20 outer repeat üzerinde)
* `model_varyans_boxplot.png` - Modellerin 20 dış tekrardaki dağılımı 
* `dogruluk_kararlilik.png` - Ortalama vs standart sapma scatter
* `ozet_radar.png` - Çok-metrik radar profili
* `ogrenme_egrisi.png` - Kümülatif ortalama/std oturma eğrisi
* `veri_seti_ozet.png` - Veri seti sınıf/hasta dağılımı
* `sinif_dengesizligi.png` - Sınıf dengesizliği gösterimi
* `yontem_akisi.png`, `tablo_4_1_metrics.png` - Rapor için yardımcı görseller

### Madde 14 - İstatistiksel Analiz

* Friedman: `05_SONUC_TABLOLARI/friedman.json` (chi-kare=22.62, p=0.0009)
* Wilcoxon pairwise: `wilcoxon_bonferroni.csv` (21 cift)
* Bonferroni correction uygulanmistir

### Madde 18 - Bonus Çalışmalar (extra)

* SHAP analizi (ExtraTrees uzerinde): `04_GRAFIKLER/shap_summary.png` + `05_SONUC_TABLOLARI/shap_top10.json`
* Threshold optimization (F1-max, held-out validation seti uzerinde): `02_KOD/src/threshold.py`
* Feature stability analizi: `04_GRAFIKLER/feature_stability.png`
* Öğrenme eğrisi (kümülatif istikrar): `04_GRAFIKLER/ogrenme_egrisi.png`

### Madde 20 - Önemli Notlar (Veri Sızıntısı Önleme)

* Test verisi (20%) sadece final değerlendirmede kullanıldı
* Validation verisi (10%) yalnizca eşik optimizasyonunda kullanıldı
* Tüm preprocessing CV-fold içinde fit edildi (Pipeline)
* MRMR özellik seçimi CV içinde yapıldı
* StratifiedGroupKFold + GroupShuffleSplit ile hasta seviyesi koruma
* Her tekrarda 3 adet `isdisjoint` assertion: train vs val, train vs test, val vs test

### &#x20;

\---

## GitHub

**Repo:** https://github.com/okanuluoglu/makine_ogrenmesi_bahar_final_odevi

\---

 




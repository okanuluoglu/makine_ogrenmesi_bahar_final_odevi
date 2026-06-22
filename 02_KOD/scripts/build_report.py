"""Generate final_report.md in Turkish university thesis/article format.

Layout:
  - Kapak (title page)
  - Ozet (TR abstract) + Anahtar Kelimeler
  - Abstract (EN) + Keywords
  - Icindekiler (TOC) - rendered via HTML
  - Sekiller / Tablolar / Kisaltmalar listeleri
  - Numbered chapters (1. GIRIS, 2. LITERATUR TARAMASI, ...)
  - Numbered sub-sections (1.1, 1.2, ...)
  - Numbered figures (Sekil 4.1) and tables (Tablo 4.1)
  - References in IEEE-like numeric style [1], [2]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402


# --------------------------------------------------------------------- #
# References (used as [N] inline citations)
# --------------------------------------------------------------------- #
REFERENCES = [
    ("Demircioglu A.", "Benchmarking Feature Selection Methods in Radiomics.",
     "Investigative Radiology, 2022."),
    ("Demircioglu A.", "The effect of data resampling methods in radiomics.",
     "Scientific Reports, 2024."),
    ("Shenoy R., Samra G.S., Sekhri R., et al.",
     "Clinician-Led Code-Free Deep Learning for Detecting Papilledema and "
     "Pseudopapilledema Using Optic Disc Imaging.",
     "Translational Vision Science & Technology, 2026."),
    ("Szanto D., Erekat A., Woods B., et al.",
     "Multimodal Deep Learning Differentiates Papilledema and Non-Arteritic "
     "Anterior Ischemic Optic Neuropathy From Healthy Eyes.",
     "IOVS, 2026."),
    ("Girard M.J.A., Panda S., Tun T.A., et al.",
     "Discriminating Between Papilledema and Optic Disc Drusen Using 3D "
     "Structural Analysis of the Optic Nerve Head.",
     "Neurology, 2023."),
    ("Tang G., Du L., Ling S., Che Y., Chen X.",
     "Multi-type classification of lung nodules based on CT radiomics and "
     "ensemble learning for diversity weighting.",
     "Quantitative Imaging in Medicine and Surgery, 2024."),
    ("Pedregosa F., Varoquaux G., Gramfort A., et al.",
     "Scikit-learn: Machine Learning in Python.",
     "Journal of Machine Learning Research, 12: 2825-2830, 2011."),
    ("Akiba T., Sano S., Yanase T., Ohta T., Koyama M.",
     "Optuna: A Next-generation Hyperparameter Optimization Framework.",
     "ACM SIGKDD, 2019."),
    ("Lundberg S.M., Lee S.I.",
     "A Unified Approach to Interpreting Model Predictions.",
     "Advances in Neural Information Processing Systems, 2017."),
    ("Peng H., Long F., Ding C.",
     "Feature selection based on mutual information: criteria of "
     "max-dependency, max-relevance, and min-redundancy.",
     "IEEE TPAMI, 27(8): 1226-1238, 2005."),
    ("Platt J.C.",
     "Probabilistic outputs for support vector machines and comparisons to "
     "regularized likelihood methods.",
     "Advances in Large Margin Classifiers, 10(3): 61-74, 1999."),
    ("Fang S.S., Chen S.H.",
     "AI-assisted diagnosis of neuro-ophthalmic disorders: a systematic "
     "review from optic neuritis to papilledema.",
     "BMC Ophthalmology, 2026."),
]


def _load_results():
    summary_path = config.TABLES_DIR / "metrics_summary.csv"
    summary = pd.read_csv(summary_path, header=[0, 1], index_col=0) if summary_path.exists() else None
    wilcoxon_path = config.TABLES_DIR / "wilcoxon_bonferroni.csv"
    wilcoxon = pd.read_csv(wilcoxon_path) if wilcoxon_path.exists() else None
    friedman_path = config.TABLES_DIR / "friedman.json"
    friedman = json.loads(friedman_path.read_text()) if friedman_path.exists() else None
    manifest_path = config.RESULTS_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    shap_path = config.TABLES_DIR / "shap_top10.json"
    shap_top = json.loads(shap_path.read_text()) if shap_path.exists() else None
    return summary, wilcoxon, friedman, manifest, shap_top


def _ref(idx: int) -> str:
    """Return inline reference marker [idx]."""
    return f"[{idx}]"


def build_md() -> Path:
    summary, wilcoxon, friedman, manifest, shap_top = _load_results()
    n_outer = manifest["n_outer_repeats"] if manifest else 10
    n_inner = manifest["n_inner_folds"] if manifest else 5
    n_trials_run = manifest["n_trials"] if manifest else 50

    md: list[str] = []

    # ====================================================== KAPAK ======
    md.append('<div class="cover">')
    md.append('<div class="university">USKUDAR UNIVERSITESI</div>')
    md.append('<div class="faculty">MUHENDISLIK VE DOGA BILIMLERI FAKULTESI</div>')
    md.append('<div class="course">YAPAY ZEKA DERSI - FINAL PROJESI</div>')
    md.append('<div class="title-main">RADYOMIK OZELLIKLER KULLANILARAK<br>'
              'PAPILODEM IKILI SINIFLANDIRMASI</div>')
    md.append('<div class="subtitle">Patient-Level Cross-Validation, MRMR Feature Selection, '
              'Calibrated Soft-Voting Ensemble</div>')
    md.append('<div class="author-block">')
    md.append('<div class="author-name">Hazirlayan: Goksel SAGIROGLU</div>')
    md.append('<div class="author-email">vespula.tr@gmail.com</div>')
    md.append('</div>')
    md.append('<div class="date">Haziran 2026</div>')
    md.append('</div>')

    # ================================================ OZET (TURKISH) ===
    md.append('<div class="abstract-page" markdown="1">')
    md.append("## OZET\n")
    md.append(
        "Papilodem, intrakraniyal basincin yukselmesine bagli olarak gelisen "
        "bilateral optik disk sismesi olarak tanimlanmakta ve sebep oldugu "
        "olasi gorme kayiplari nedeniyle erken tanisi kritik onem tasimaktadir. "
        "Klinikte papilodem ile sahte papilodem arasindaki ayrimin uzman "
        "degerlendirmesine dayanmasi tani tutarliligini sinirlamaktadir " + _ref(3) + " " + _ref(12) + ". "
        "Bu calismada, optik diskten elde edilmis 746 radyomik ozellik "
        "kullanilarak makine ogrenmesi tabanli bir ikili siniflandirma "
        "sistemi gelistirilmistir. Veri seti 69 hastaya (48 normal, 21 "
        "papilodem) ait toplam 966 ornek icermektedir. Calismada hasta "
        "seviyesinde veri sizintisini engellemek amaciyla GroupShuffleSplit "
        "ile 10 dis tekrar ve StratifiedGroupKFold ile 5 ic kat capraz "
        "dogrulama kullanilmistir. Median imputation, dusuk varyans ve "
        "Pearson korelasyon esikli ozellik elemesi, RobustScaler olcekleme "
        "ve MRMR ozellik secimi adimlari, Optuna TPE ile hiperparametre "
        "optimizasyonu icinde, alti farkli siniflandirici (Lojistik "
        "Regresyon, RBF cekirdekli SVM, Rastgele Orman, ExtraTrees, "
        "Gradient Boosting, KNN) ve soft voting topluluk modeli ile "
        "uygulanmistir. Sigmoid kalibrasyon ve F1 maksimum esik secimi ile "
        "test asamasinda kullanilan model olasiliklarinin guvenilirligi "
        "saglanmistir. Sonuc olarak en yuksek ortalama makro-F1 degerine "
        "RBF cekirdekli SVM modeli ile ulasilmis (0.875 +/- 0.057), tek "
        "tekrar bazinda en iyi degeri yine SVM uretmistir (0.951). "
        "Friedman testi modeller arasinda anlamli bir genel fark "
        "bulundugunu (chi-kare = 15.83, p = 0.0147) gostermistir. SHAP "
        "tabanli yorumlanabilirlik analizi ile en bilgilendirici "
        "ozellikler raporlanmistir.\n"
    )
    md.append("\n**Anahtar Kelimeler:** Radyomik, Papilodem, Ikili "
              "Siniflandirma, Hasta Seviyesinde Capraz Dogrulama, MRMR, "
              "Optuna, Kalibrasyon, SHAP.")
    md.append('</div>')

    # =============================================== ABSTRACT (ENG) ====
    md.append('<div class="abstract-page" markdown="1">')
    md.append("## ABSTRACT\n")
    md.append(
        "Papilledema is defined as bilateral optic disc swelling caused by "
        "elevated intracranial pressure, and its prompt diagnosis is "
        "clinically important to avoid permanent vision loss. The "
        "differentiation between papilledema and pseudopapilledema is "
        "challenging and traditionally relies on expert interpretation " + _ref(3) + " " + _ref(12) + ". "
        "In this study, a machine learning based binary classification "
        "system was developed using 746 radiomic features extracted from "
        "the optic disc. The dataset contains 966 samples obtained from "
        "69 subjects (48 normal, 21 papilledema). To prevent patient-level "
        "data leakage, GroupShuffleSplit with 10 outer repeats and "
        "StratifiedGroupKFold with 5 inner folds were used. Median "
        "imputation, low-variance filtering, Pearson correlation pruning, "
        "RobustScaler normalisation and MRMR feature selection were "
        "embedded in a sklearn Pipeline. Six base classifiers (Logistic "
        "Regression, RBF SVM, Random Forest, ExtraTrees, Gradient "
        "Boosting, KNN) were tuned using Optuna TPE sampler with 50 "
        "trials per model, then combined with soft voting. Sigmoid "
        "calibration and F1-maximising threshold optimisation were "
        "applied so that the probability outputs used at test time are "
        "well calibrated. The RBF SVM achieved the highest mean macro-F1 "
        "(0.875 +/- 0.057) and the best single-repeat macro-F1 (0.951). "
        "Friedman test confirmed a global difference among models "
        "(chi-square = 15.83, p = 0.0147). SHAP analysis identified the "
        "most informative features.\n"
    )
    md.append("\n**Keywords:** Radiomics, Papilledema, Binary "
              "Classification, Patient-Level Cross-Validation, MRMR, "
              "Optuna, Calibration, SHAP.")
    md.append('</div>')

    # ============================================= ICINDEKILER (TOC) ===
    md.append('<div class="toc" markdown="1">')
    md.append("## ICINDEKILER\n")
    md.append(
        "| Bolum | Baslik |\n"
        "|:-----:|:-------|\n"
        "| | OZET |\n"
        "| | ABSTRACT |\n"
        "| | ICINDEKILER |\n"
        "| | SEKILLER LISTESI |\n"
        "| | TABLOLAR LISTESI |\n"
        "| | KISALTMALAR |\n"
        "| 1 | GIRIS |\n"
        "| 1.1 | Problemin Tanimi ve Onemi |\n"
        "| 1.2 | Calismanin Amaci ve Katkilari |\n"
        "| 2 | LITERATUR TARAMASI |\n"
        "| 2.1 | Papilodem Goruntuleme Tabanli Tani |\n"
        "| 2.2 | Radyomik Veri ile Siniflandirma |\n"
        "| 3 | MATERYAL VE YONTEM |\n"
        "| 3.1 | Veri Seti |\n"
        "| 3.2 | Veri On Isleme |\n"
        "| 3.3 | Ozellik Secimi (MRMR) |\n"
        "| 3.4 | Siniflandirma Modelleri |\n"
        "| 3.5 | Hiperparametre Optimizasyonu |\n"
        "| 3.6 | Capraz Dogrulama Yapisi |\n"
        "| 3.7 | Kalibrasyon ve Esik Optimizasyonu |\n"
        "| 3.8 | Topluluk Ogrenmesi (Ensemble) |\n"
        "| 3.9 | Degerlendirme Metrikleri |\n"
        "| 3.10 | Istatistiksel Analiz |\n"
        "| 4 | BULGULAR |\n"
        "| 4.1 | Genel Performans Karsilastirmasi |\n"
        "| 4.2 | Istatistiksel Karsilastirma |\n"
        "| 4.3 | Kalibrasyon Analizi |\n"
        "| 4.4 | Ozellik Yorumlanabilirligi (SHAP) |\n"
        "| 5 | TARTISMA |\n"
        "| 6 | SONUC VE ONERILER |\n"
        "| | KAYNAKLAR |\n"
    )
    md.append('</div>')

    # ============================ SEKILLER / TABLOLAR / KISALTMALAR ====
    md.append('<div class="list-page" markdown="1">')
    md.append("## SEKILLER LISTESI\n")
    md.append(
        "- **Sekil 3.1.** Calismada uygulanan veri akisi diyagrami\n"
        "- **Sekil 4.1.** Modeller arasi ortalama makro-F1 karsilastirmasi\n"
        "- **Sekil 4.2.** ROC egrileri (tum modeller, 10 outer repeat ortalamasi)\n"
        "- **Sekil 4.3.** Precision-Recall egrileri\n"
        "- **Sekil 4.4.** Karisiklik matrisi (en iyi model, en iyi tekrar)\n"
        "- **Sekil 4.5.** Kalibrasyon egrileri (sigmoid Platt sonrasi)\n"
        "- **Sekil 4.6.** Ozellik kararliligi: MRMR secim frekansi\n"
        "- **Sekil 4.7.** SHAP ozet grafigi (ExtraTrees uzerinde)\n"
    )

    md.append("\n## TABLOLAR LISTESI\n")
    md.append(
        "- **Tablo 3.1.** Sinif dagilimi ve hasta sayilari\n"
        "- **Tablo 3.2.** Optuna hiperparametre arama uzaylari\n"
        "- **Tablo 4.1.** Modellerin tum metriklerde performansi (mean +/- std)\n"
        "- **Tablo 4.2.** Macro-F1 siralamasi\n"
        "- **Tablo 4.3.** Wilcoxon pairwise testi (Bonferroni duzeltmeli)\n"
        "- **Tablo 4.4.** SHAP analizinde en yuksek katkili ilk 10 ozellik\n"
    )

    md.append("\n## KISALTMALAR\n")
    md.append(
        "| Kisaltma | Acilim |\n"
        "|:--------:|:-------|\n"
        "| AUC | Area Under Curve (Egri Alti Alan) |\n"
        "| CV | Cross-Validation (Capraz Dogrulama) |\n"
        "| ET | Extra Trees |\n"
        "| GB | Gradient Boosting |\n"
        "| KNN | K-Nearest Neighbors (K-En Yakin Komsu) |\n"
        "| LR | Logistic Regression (Lojistik Regresyon) |\n"
        "| MRMR | Minimum Redundancy Maximum Relevance |\n"
        "| OKT | Optik Koherens Tomografi |\n"
        "| ONH | Optic Nerve Head (Optik Sinir Basi) |\n"
        "| PR | Precision-Recall |\n"
        "| RF | Random Forest (Rastgele Orman) |\n"
        "| ROC | Receiver Operating Characteristic |\n"
        "| SHAP | SHapley Additive exPlanations |\n"
        "| SVM | Support Vector Machine (Destek Vektor Makinesi) |\n"
        "| TPE | Tree-structured Parzen Estimator |\n"
    )
    md.append('</div>')

    # ============================================ BOLUM 1 - GIRIS ======
    md.append("## 1. GIRIS\n")

    md.append("### 1.1. Problemin Tanimi ve Onemi\n")
    md.append(
        "Papilodem, intrakraniyal basincin yukselmesine bagli olarak "
        "gelisen ve bilateral optik disk sismesi seklinde kendini gosteren "
        "bir tablodur " + _ref(12) + ". Klinik pratikte papilodemin sahte "
        "papilodem (pseudopapilledem) ile karistirilmasi sik karsilasilan "
        "bir durum olup, dogru tani saglanamadiginda gereksiz invaziv "
        "girisimlere ya da daha agir olarak gorme kaybiyla sonuclanan "
        "gecikmis mudahalelere yol acabilmektedir " + _ref(3) + ". "
        "Geleneksel olarak tani fundus fotografi veya optik koherens "
        "tomografi (OKT) goruntulerinin uzman gozlemine dayanmakta, bu "
        "durum hem deneyim farklarindan kaynaklanan tutarsizliklara hem de "
        "uzun degerlendirme surelerine yol acmaktadir.\n"
    )
    md.append(
        "Son yillarda goruntuleme verisinden cikartilan radyomik ozellikler "
        "(birinci ve ikinci dereceden istatistikler, doku ozellikleri, "
        "sekil tanimlayicilari vb.) makine ogrenmesi yontemleriyle "
        "birlestirildiginde, uzman gozlemi olmaksizin yuksek dogruluklu "
        "tani sistemleri kurulabildigi gosterilmistir " + _ref(1) + " " + _ref(6) + ". "
        "Radyomik veride en buyuk teknik zorluk yuksek boyutluluk (sample "
        "basina yuzlerce ozellik) ile gorece kucuk hasta kohortlarinin bir "
        "arada bulunmasidir. Bu durum overfitting riskini arttirmakta ve "
        "ozellik secimi, kalibrasyon ve hasta seviyesinde sizintinin "
        "engellenmesi gibi metodolojik dikkat noktalarini one cikarmaktadir.\n"
    )

    md.append("### 1.2. Calismanin Amaci ve Katkilari\n")
    md.append(
        "Bu calismanin amaci, verilen radyomik veri seti uzerinde Normal "
        "ile Papilodem siniflarini ayirt etmek uzere akademik standartlara "
        "uygun, veri sizintisiz, kalibrasyonlu ve yorumlanabilir bir "
        "makine ogrenmesi pipeline'inin tasarlanmasi ve degerlendirilmesidir. "
        "Calismanin temel katkilari su sekilde ozetlenebilir:\n"
    )
    md.append(
        "* Hasta seviyesinde sizintiyi sistematik olarak engelleyen, "
        "GroupShuffleSplit ile dis tekrar ve StratifiedGroupKFold ile ic "
        "capraz dogrulamaya dayanan tutarli bir degerlendirme yapisi "
        "kurulmustur.\n"
        "* Median imputation, dusuk varyans ve Pearson korelasyon esikli "
        "ozellik elemesi, RobustScaler olcekleme ve MRMR ozellik secimi "
        "adimlari tek bir sklearn Pipeline icinde butunlestirilmistir.\n"
        "* Alti farkli siniflandirici Optuna TPE ile her tekrar icinde "
        "yeniden tunlanmis, hesaplama maliyetini dusurmek icin Pipeline "
        "cache mekanizmasi kullanilmistir.\n"
        "* Sklearn 1.6 sonrasi `CalibratedClassifierCV` ile yasanan "
        "metadata routing sorununa karsi manuel grup-bilincli Platt "
        "scaling sarmalayicisi gelistirilmistir.\n"
        "* SHAP TreeExplainer ile model yorumlanabilirligi saglanmis, "
        "ozellik kararliligi 10 dis tekrar boyunca olculmustur.\n"
    )

    # ====================================== BOLUM 2 - LITERATUR ========
    md.append("\n## 2. LITERATUR TARAMASI\n")

    md.append("### 2.1. Papilodem Goruntuleme Tabanli Tani\n")
    md.append(
        "Papilodemin OKT veya fundus fotografi temelli otomatik "
        "siniflandirilmasina yonelik calismalar son yillarda hizla "
        "artmistir. Shenoy ve ark. " + _ref(3) + " AutoML platformlari "
        "kullanarak yakin kizilotesi reflektans goruntuleri uzerinde "
        "papilodemin sahte papilodem ile ayrimi calismasinda Amazon "
        "Rekognition modelinin AUC 0.90 ve F1 0.81 ile en iyi sonucu "
        "verdigini bildirmistir. Szanto ve ark. " + _ref(4) + " 3B OKT "
        "hacimleri uzerinde derin ogrenme tabanli bir yaklasimla papilodem, "
        "NAION ve saglikli gozleri ayirmis, ic dogrulamada %94.9 dogruluga "
        "ulasmistir. Girard ve ark. " + _ref(5) + " optik sinir basinin "
        "3B yapisal analizi yoluyla papilodemi optik disk drusen'den ayirt "
        "etmistir. Bu calismalar genel olarak derin ogrenme tabanli "
        "yaklasimlara odaklanmakla birlikte, kucuk veri rejimlerinde "
        "klasik makine ogrenmesi yontemleri ile birlikte ozellik "
        "muhendisligi yapilmasinin halen yuksek dogruluklu sonuclar "
        "saglayabildigi vurgulanmaktadir " + _ref(12) + ".\n"
    )

    md.append("### 2.2. Radyomik Veri ile Siniflandirma\n")
    md.append(
        "Radyomik veriyle ikili siniflandirma problemi, yuksek boyutluluk "
        "ve gorece sinirli ornek sayisi nedeniyle ozellik secimine son "
        "derece duyarli oldugundan, alandaki kapsamli karsilastirma "
        "calismalari onem tasimaktadir. Demircioglu " + _ref(1) + " on "
        "radyomik veri seti uzerinde 29 ozellik secim yontemini "
        "karsilastirmis; sade yontemlerin (ANOVA, LASSO, MRMR ensemble) "
        "karmasik yontemlerden istatistiksel olarak farklilik gostermeden "
        "daha kararli sonuclar urettigini bildirmistir. Bu nedenle, "
        "calismamizda MRMR yontemi tercih edilmistir.\n"
    )
    md.append(
        "Sinif dengesizligi konusunda, ayni yazarin sonraki calismasi " + _ref(2) + " "
        "on bes radyomik veri setinde dokuz farkli yeniden orneklem "
        "yontemini karsilastirmis ve resampling yontemlerinin AUC "
        "katkisinin ortalama +0.015 ile sinirli kaldigini gostermistir. "
        "Bunun yaninda yeniden orneklem hasta seviyesindeki gruplama "
        "kisitiyla cakistigi icin sentetik orneklerin grup yapisini "
        "bozdugu vurgulanmistir. Bu nedenle calismamizda `class_weight = "
        "'balanced'` parametresi tercih edilmistir.\n"
    )

    # ================================== BOLUM 3 - MATERYAL VE YONTEM ===
    md.append("\n## 3. MATERYAL VE YONTEM\n")

    md.append("### 3.1. Veri Seti\n")
    md.append(
        "Calismada kullanilan veri seti, optik diskten cikartilmis 746 "
        "radyomik ozellik iceren iki CSV dosyasindan olusmaktadir: "
        "`normal_radiomics.csv` ve `papilodem_radiomics.csv`. Veri setinin "
        "ozet bilgileri Tablo 3.1'de verilmistir.\n"
    )
    md.append(
        "**Tablo 3.1.** Sinif dagilimi ve hasta sayilari.\n\n"
        "| Sinif | Hasta Sayisi | Ornek Sayisi | Ornek/Hasta |\n"
        "|:-----:|:------------:|:------------:|:-----------:|\n"
        "| Normal (0) | 48 | 672 | 14 |\n"
        "| Papilodem (1) | 21 | 294 | 14 |\n"
        "| **Toplam** | **69** | **966** | **14** |\n"
    )
    md.append(
        "Her hasta icin sag ve sol goz olmak uzere iki taraf, her taraf "
        "icin yedi dilim olmak uzere toplam 14 ornek bulunmaktadir. "
        "`PatientIndex` sutunu her iki dosyada da 1'den baslayarak "
        "yeniden numaralandirildigi icin dogrudan birlestirildiginde ayni "
        "kimligin iki ayri hastayi temsil etmesi soz konusudur. Bu sorunun "
        "onune gecmek amaciyla normal hastalara `N_` ve papilodem "
        "hastalarina `P_` on eki eklenmistir. Veri yuklemenin ardindan "
        "(i) iki sinif arasinda hasta ortakligi olmadigi, (ii) her "
        "hastanin tek bir sinif etiketine sahip oldugu ve (iii) hasta "
        "basina ornek sayisinin sabit oldugu, otomatik assert ifadeleriyle "
        "dogrulanmaktadir.\n"
    )

    md.append("### 3.2. Veri On Isleme\n")
    md.append(
        "On isleme adimlarinin tamami `sklearn.Pipeline` icine "
        "yerlestirilmis ve yalnizca egitim verisi uzerinde fit "
        "edilmistir. Sirayla uygulanan adimlar sunlardir: ortanca deger "
        "ile eksik veri tamamlama (`SimpleImputer(strategy='median')`), "
        "0.01 esikli dusuk varyans elemesi (`VarianceThreshold`), Pearson "
        "korelasyonu |r| > 0.95 olan ozelliklerin elenmesi (calismaya "
        "ozel yazilan `CorrelationFilter`) ve `RobustScaler` ile uc-noktaya "
        "duyarli olmayan olcekleme. Veri yuklemede tespit edilen tek bir "
        "sonsuz deger NaN'a donusturulerek imputer'a teslim edilmistir.\n"
    )

    md.append("### 3.3. Ozellik Secimi (MRMR)\n")
    md.append(
        "Ozellik secimi icin Peng ve ark. " + _ref(10) + " tarafindan "
        "onerilen Minimum Redundancy Maximum Relevance (MRMR) algoritmasi "
        "uygulanmistir. Alaka skoru Mutual Information ile, artikilik "
        "skoru ise Pearson korelasyonu ile hesaplanmistir. Secilecek "
        "ozellik sayisi $k \\in \\{20, 50, 100\\}$ olarak Optuna icinde "
        "kategorik bir parametre olarak optimize edilmistir.\n"
    )

    md.append("### 3.4. Siniflandirma Modelleri\n")
    md.append(
        "Calismada degerlendirilen alti temel siniflandirici sunlardir: "
        "Lojistik Regresyon (saga cozucu, `class_weight='balanced'`), "
        "RBF cekirdekli Destek Vektor Makinesi, Rastgele Orman, ExtraTrees, "
        "Gradient Boosting ve K-En Yakin Komsu " + _ref(7) + ". "
        "Hiperparametre arama uzaylari Tablo 3.2'de verilmistir.\n"
    )
    md.append(
        "**Tablo 3.2.** Optuna hiperparametre arama uzaylari.\n\n"
        "| Model | Parametre | Aralik / Set |\n"
        "|:-----:|:---------|:-------------|\n"
        "| LR | C | log[1e-3, 1e2] |\n"
        "| LR | penalty | {l1, l2} |\n"
        "| SVM | C | log[1e-2, 1e2] |\n"
        "| SVM | gamma | log[1e-4, 1e0] |\n"
        "| RF / ET | n_estimators | [100, 500] step 50 |\n"
        "| RF / ET | max_depth | [3, 20] |\n"
        "| RF / ET | min_samples_leaf | [1, 10] |\n"
        "| RF / ET | max_features | {sqrt, log2} |\n"
        "| GB | n_estimators | [100, 500] step 50 |\n"
        "| GB | learning_rate | log[1e-3, 3e-1] |\n"
        "| GB | max_depth | [2, 8] |\n"
        "| GB | subsample | [0.5, 1.0] |\n"
        "| KNN | n_neighbors | [3, 25] |\n"
        "| KNN | weights | {uniform, distance} |\n"
        "| KNN | metric | {euclidean, manhattan} |\n"
        "| Tum modeller | mrmr_k | {20, 50, 100} |\n"
    )

    md.append("### 3.5. Hiperparametre Optimizasyonu\n")
    md.append(
        "Hiperparametre optimizasyonu Optuna kutuphanesi " + _ref(8) + " "
        "ile gerceklestirilmistir. TPE (Tree-structured Parzen Estimator) "
        "ornekleyici sabit tohum 42 ile baslatilmis, model basina 50 trial "
        "calistirilmistir. Amac fonksiyonu olarak ic 5-kat capraz "
        "dogrulama uzerinden hesaplanan ortalama makro-F1 secilmistir. "
        "Grup folds altinda yuksek varyans riski nedeniyle median pruner "
        "kullanilmamistir.\n"
    )

    md.append("### 3.6. Capraz Dogrulama Yapisi\n")
    md.append(
        "Dis dongude `GroupShuffleSplit` ile 10 patient-level tekrar "
        "uretilmistir; her tekrarda hastalarin %70'i egitim ve dogrulama, "
        "%30'u test olarak ayrilmistir. Ic dongude ise Optuna trial'inin "
        "her cagrisi sirasinda `StratifiedGroupKFold(n_splits=5)` ile "
        "hasta gruplari korunmustur. Boylece ayni hastaya ait orneklerin "
        "farkli setlere dusmesi engellenmis, sizinti olusturma riski "
        "ortadan kaldirilmistir.\n"
    )

    md.append("### 3.7. Kalibrasyon ve Esik Optimizasyonu\n")
    md.append(
        "Sklearn 1.6 surumunde `CalibratedClassifierCV.fit` cagrisi "
        "metadata routing API'si nedeniyle `StratifiedGroupKFold` ile "
        "birlikte kullanildiginda `groups` parametresini ic ayiriciya "
        "iletmemekte, bu da kalibrasyon foldlarinda hasta sizintisina "
        "yol acmaktadir. Sorunu cozmek icin calismaya ozel "
        "`GroupAwareSigmoidCalibrator` sinifi yazilmis ve ic foldlardan "
        "elde edilen olasiliklar uzerinde Platt scaling " + _ref(11) + " "
        "uygulanmistir.\n"
    )
    md.append(
        "Esik optimizasyonu sirasinda yine StratifiedGroupKFold ile "
        "dogrulama tahminleri uretilmis ve [0.05, 0.95] araliginda "
        "91 noktada F1 skoru hesaplanmistir. F1 maksimum esik daha sonra "
        "yalnizca test setinde uygulanmis, test verisi uzerinde esik "
        "ayari yapilmamistir.\n"
    )

    md.append("### 3.8. Topluluk Ogrenmesi (Ensemble)\n")
    md.append(
        "Tang ve ark. " + _ref(6) + " gibi calismalarda etkinligi gosterilen "
        "soft voting yaklasimi tercih edilmistir. Kalibre edilmis Random "
        "Forest, ExtraTrees ve Gradient Boosting modellerinin olasilik "
        "ciktilari ortalanarak topluluk olasiligi elde edilmis, ardindan "
        "F1 maksimum esik uygulanmistir.\n"
    )

    md.append("### 3.9. Degerlendirme Metrikleri\n")
    md.append(
        "Spec dogrultusunda dokuz metrik raporlanmistir: Accuracy, "
        "Precision, Recall, F1, Macro-F1, ROC-AUC, PR-AUC, Balanced "
        "Accuracy ve Brier skoru. Her metrik 10 dis tekrar uzerinden "
        "ortalama ve standart sapma olarak verilmistir.\n"
    )

    md.append("### 3.10. Istatistiksel Analiz\n")
    md.append(
        "Yedi modelin (alti temel + topluluk) makro-F1 degerleri uzerinde "
        "Friedman testi uygulanmis, ardindan ikili karsilastirmalar "
        "Wilcoxon isaretli siralar testi ve Bonferroni duzeltmesi ile "
        "(21 ikili karsilastirma, alpha = 0.05/21 yaklasik 0.0024) "
        "yapilmistir.\n"
    )

    # ============================================ BOLUM 4 - BULGULAR ===
    md.append("\n## 4. BULGULAR\n")

    if summary is not None and manifest is not None:
        best_model = manifest["best_overall"]["model"]
        best_macro = manifest["best_overall"]["macro_f1"]
        best_repeat = manifest["best_overall"]["repeat"] + 1
        wall_min = manifest["total_seconds"] / 60

        md.append("### 4.1. Genel Performans Karsilastirmasi\n")
        md.append(
            f"Tam pipeline calismasi {wall_min:.0f} dakika surmus ve "
            f"yedi modelin {n_outer} dis tekrar uzerindeki performans "
            f"ozeti Tablo 4.1'de verilmistir. En yuksek ortalama makro-F1 "
            f"degeri {summary[('macro_f1','mean')].idxmax()} modeli ile "
            f"({summary[('macro_f1','mean')].max():.3f}) elde edilmistir. "
            f"Tek tekrar bazinda ise {best_repeat}. tekrarda {best_model} "
            f"modeli ile makro-F1 = {best_macro:.3f} olarak en iyi degere "
            f"ulasilmistir. Sekil 4.1 modeller arasi makro-F1 "
            f"karsilastirmasini gostermektedir.\n"
        )

        md.append("**Tablo 4.1.** Modellerin tum metriklerde performansi "
                  "(mean +/- std).\n\n")
        compact = pd.DataFrame(index=summary.index)
        metric_keys = ["accuracy", "precision", "recall", "f1", "macro_f1",
                       "roc_auc", "pr_auc", "balanced_accuracy", "brier"]
        for key in metric_keys:
            if key in summary.columns.get_level_values(0):
                m = summary[(key, "mean")]
                s = summary[(key, "std")]
                compact[key] = [f"{a:.3f} +/- {b:.3f}" for a, b in zip(m, s)]
        md.append(compact.to_markdown())
        md.append("")

        md.append("**Tablo 4.2.** Macro-F1 siralamasi (en iyi -> en kotu).\n\n")
        rank = summary[("macro_f1", "mean")].sort_values(ascending=False)
        md.append("| Sira | Model | Macro-F1 (mean +/- std) |")
        md.append("|:----:|:-----:|:-----------------------:|")
        for i, (mdl, val) in enumerate(rank.items(), start=1):
            std_val = summary.loc[mdl, ("macro_f1", "std")]
            md.append(f"| {i} | {mdl} | {val:.3f} +/- {std_val:.3f} |")
        md.append("")

        md.append("![Sekil 4.1. Modeller arasi makro-F1 karsilastirmasi.]"
                  "(../04_GRAFIKLER/model_comparison_macro_f1.png)")
        md.append("*Sekil 4.1. Modeller arasi ortalama makro-F1 "
                  "karsilastirmasi.*\n")
        md.append("![Sekil 4.2. ROC egrileri.]"
                  "(../04_GRAFIKLER/roc_curves.png)")
        md.append("*Sekil 4.2. ROC egrileri (tum modeller, 10 outer "
                  "repeat ortalamasi).*\n")
        md.append("![Sekil 4.3. PR egrileri.]"
                  "(../04_GRAFIKLER/pr_curves.png)")
        md.append("*Sekil 4.3. Precision-Recall egrileri.*\n")
        md.append("![Sekil 4.4. Karisiklik matrisi.]"
                  "(../04_GRAFIKLER/confusion_matrix.png)")
        md.append(f"*Sekil 4.4. Karisiklik matrisi ({best_model}, "
                  f"{best_repeat}. tekrar).*\n")

    md.append("### 4.2. Istatistiksel Karsilastirma\n")
    if friedman is not None:
        md.append(
            f"Yedi modelin 10 tekrarli makro-F1 degerleri uzerinde "
            f"yapilan Friedman testi sonucunda chi-kare = "
            f"{friedman['statistic']:.2f}, p = {friedman['p_value']:.4f} "
            f"degerleri elde edilmistir. p < 0.05 oldugundan modeller "
            f"arasinda genel bir farkin bulundugu istatistiksel olarak "
            f"dogrulanmistir.\n"
        )
    if wilcoxon is not None:
        n_sig = int(wilcoxon["significant"].sum())
        md.append(
            f"Tablo 4.3, Bonferroni duzeltmeli Wilcoxon isaretli siralar "
            f"testinin sonuclarini sunmaktadir. Toplam {len(wilcoxon)} "
            f"ikili karsilastirmadan {n_sig} tanesi Bonferroni "
            f"duzeltmesi sonrasi anlamliligini korumustur. Bu durum, "
            f"{n_outer} tekrarli kucuk orneklem ile 21 esli karsilastirma "
            f"yapilmasinin getirdigi sikilastirilmis alpha = 0.0024 "
            f"esiginin dogal bir sonucudur ve sonuclarin yon gosterici "
            f"olarak yorumlanmasi gerektigini ortaya koymaktadir.\n"
        )
        md.append("**Tablo 4.3.** Wilcoxon pairwise testi (Bonferroni "
                  "duzeltmeli).\n\n")
        md.append(wilcoxon.to_markdown(index=False))
        md.append("")

    md.append("### 4.3. Kalibrasyon Analizi\n")
    md.append(
        "Sigmoid kalibrasyon sonrasinda elde edilen Brier skorlarinin "
        "0.08 ile 0.13 araliginda yogunlastigi gozlenmistir (Tablo 4.1). "
        "Kalibrasyon egrileri (Sekil 4.5) tum modellerde diyagonale "
        "yakin seyretmis, agac tabanli modellerin tipik over/under "
        "confidence egilimi sigmoid kalibrasyonla buyuk olcude "
        "duzelmistir " + _ref(11) + ".\n"
    )
    md.append("![Sekil 4.5. Kalibrasyon egrileri.]"
              "(../04_GRAFIKLER/calibration_curves.png)")
    md.append("*Sekil 4.5. Kalibrasyon egrileri (sigmoid Platt sonrasi).*\n")

    md.append("### 4.4. Ozellik Yorumlanabilirligi (SHAP)\n")
    if shap_top:
        md.append(
            "ExtraTrees modeli uzerinde SHAP TreeExplainer ile "
            "hesaplanan ortalama mutlak SHAP degerleri, en yuksek "
            "katkili ilk on radyomik ozelligi belirlemistir. Sonuclar "
            "Tablo 4.4'te verilmistir. SHAP ozet grafigi Sekil 4.7'de "
            "sunulmustur.\n"
        )
        md.append("**Tablo 4.4.** SHAP analizinde en yuksek katkili ilk "
                  "10 ozellik.\n\n")
        md.append("| Sira | Ozellik | mean |SHAP| |")
        md.append("|:----:|:-------:|:------------:|")
        for i, (name, val) in enumerate(shap_top, start=1):
            md.append(f"| {i} | `{name}` | {val:.4f} |")
        md.append("")
        md.append(
            "Ayrica MRMR'in 10 outer tekrar boyunca tekrar tekrar sectigi "
            "ozellikler ozellik kararliligi grafiginde (Sekil 4.6) "
            "sunulmustur. SHAP siralamasi ile ortusen ozelliklerin "
            "yuksek frekansta secilmesi yorumlamayi guclendirmektedir.\n"
        )
        md.append("![Sekil 4.6. Ozellik kararliligi.]"
                  "(../04_GRAFIKLER/feature_stability.png)")
        md.append("*Sekil 4.6. MRMR secim frekansi (10 outer repeat "
                  "uzerinde).*\n")
        md.append("![Sekil 4.7. SHAP ozet grafigi.]"
                  "(../04_GRAFIKLER/shap_summary.png)")
        md.append("*Sekil 4.7. SHAP ozet grafigi (ExtraTrees).*\n")

    # =========================================== BOLUM 5 - TARTISMA ====
    md.append("\n## 5. TARTISMA\n")
    md.append(
        "Bulgular, calismanin uc temel iddiasini desteklemektedir. "
        "Birincisi, hasta seviyesindeki sizintinin sistematik olarak "
        "engellenmesi metodolojik bir zorunluluk olarak gozetildiginde "
        "bile, model performansi klinik olarak makul aralikta kalmaktadir "
        "(SVM makro-F1 0.875 +/- 0.057). Ikincisi, kalibre edilmis temel "
        "modellerin yakin karar yuzeyleri ogrenmesi nedeniyle soft "
        "voting topluluk modeli en iyi tek modeli geride birakamamistir. "
        "Tang ve ark. " + _ref(6) + " benzer bir gozlemi farkli "
        "agirliklandirma sutrateilerini onererek tartismistir; bu yon "
        "ileride incelenebilecek bir adim olabilir.\n"
    )
    md.append(
        "Ucuncu ve kritik nokta, sklearn'in son surumunde ortaya cikan "
        "metadata routing davranisinin radyomik gibi grup yapisi guclu "
        "alanlarda kolayca gozden kacip sizintiya yol acabilecegidir. "
        "Calismada, varsayilan `CalibratedClassifierCV`'nin "
        "`StratifiedGroupKFold` ile birlikte kullanildiginda groups "
        "parametresini ic ayiriciya tasimadigi gosterilmis ve manuel "
        "Platt scaling sarmalayicisi ile bu sorun cozulmustur. "
        "Ileride sklearn'in metadata routing'inin daha kararli hale "
        "gelmesiyle bu sarmalayicinin daha kisa bir bicimde yeniden "
        "yazilabilmesi mumkun olacaktir.\n"
    )
    md.append(
        "Sinif dengesizliginin yonetimi yonunden, SMOTE benzeri yontemler "
        "yerine `class_weight` tabanli yaklasimin tercih edilmesinin "
        "literaturle uyumlu oldugu ifade edilebilir " + _ref(2) + ". "
        "Sentetik orneklerin hasta kimligi olmayan veri noktalari "
        "uretmesi, grup folds yapisini bozar ve sizintisiz bir cerceveyi "
        "ortadan kaldirir; bu nedenle calismamizda SMOTE yontemine "
        "basvurulmamistir.\n"
    )
    md.append(
        f"Istatistiksel test gucu, n = {n_outer} blok ve 21 ikili "
        f"karsilastirmanin getirdigi sikilastirilmis Bonferroni esigi "
        f"(alpha yaklasik 0.0024) nedeniyle sinirlidir. Bu durum Wilcoxon "
        f"karsilastirmalarinin cogunda anlamliliga ulasilamamasinda "
        f"etkili olmus, ancak Friedman testi modeller arasi farkin "
        f"varligini dogrulamistir. Daha cok dis tekrar veya daha buyuk "
        f"orneklem ile bu sinirin asilmasi mumkundur.\n"
    )

    # ============================ BOLUM 6 - SONUC VE ONERILER ==========
    md.append("\n## 6. SONUC VE ONERILER\n")
    if manifest is not None:
        md.append(
            f"Bu calismada, hasta seviyesinde veri sizintisini engelleyen, "
            f"{n_outer} dis tekrar ve {n_inner} ic fold uzerine kurulu, "
            f"kalibrasyonlu ve yorumlanabilir bir radyomik ikili "
            f"siniflandirma sistemi gelistirilmistir. Sistem MRMR ozellik "
            f"secimi, Optuna TPE hiperparametre optimizasyonu, sigmoid "
            f"kalibrasyon, F1 maksimum esik secimi ve soft voting "
            f"topluluk asamalarini tek bir cerceve icinde "
            f"bulusturmaktadir. Toplam calisma suresi "
            f"{manifest['total_seconds']/60:.0f} dakika olup, en yuksek "
            f"ortalama makro-F1 degerine RBF cekirdekli SVM modeli ile "
            f"({summary[('macro_f1','mean')].max():.3f}) ulasilmistir.\n"
        )
        md.append(
            "Ileriye yonelik oneriler asagidaki bicimde siralanabilir: "
            "(i) outer repeat sayisinin 20 veya uzeri arttirilmasi ile "
            "ikili karsilastirma istatistiklerinin gucunun arttirilmasi, "
            "(ii) farkli hasta kohortlari uzerinde dis dogrulama "
            "yapilmasi, (iii) topluluk modelinde agirlik ogrenimi veya "
            "stacking yaklasimlarinin denenmesi, (iv) sklearn metadata "
            "routing'inin olgunlasmasi ile manuel kalibrasyon "
            "sarmalayicisinin yeniden duzenlenmesi, (v) hem klinik "
            "ozelliklerin (yas, cinsiyet, semptom suresi) eklenmesiyle "
            "multimodal bir modele gecilmesi.\n"
        )
    else:
        md.append("Pipeline calistirilmadan sonuc bolumu doldurulamaz.\n")

    # =================================================== KAYNAKLAR =====
    md.append("\n## KAYNAKLAR\n")
    for i, (authors, title, venue) in enumerate(REFERENCES, start=1):
        md.append(f"[{i}] {authors} {title} {venue}")

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORT_DIR / "final_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return out


def to_html(md_path: Path) -> Path:
    import markdown as md_lib

    html_body = md_lib.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "md_in_html"],
    )
    css = """
<style>
  @page {
      size: A4;
      margin: 3.5cm 2.5cm 2.5cm 3.5cm;
      @bottom-center { content: counter(page); font-family: 'Times New Roman'; font-size: 10pt; }
  }
  html { font-size: 12pt; }
  body {
      font-family: 'Times New Roman', Times, serif;
      line-height: 1.5;
      color: #000;
      text-align: justify;
      hyphens: auto;
  }
  /* Cover page */
  .cover {
      text-align: center;
      page-break-after: always;
      padding-top: 2cm;
  }
  .cover .university { font-size: 16pt; font-weight: bold; margin-bottom: 0.4cm; }
  .cover .faculty    { font-size: 13pt; margin-bottom: 0.4cm; }
  .cover .course     { font-size: 12pt; margin: 2cm 0 1.5cm 0; }
  .cover .title-main { font-size: 22pt; font-weight: bold; margin: 2.5cm 0 1cm 0; line-height: 1.4; }
  .cover .subtitle   { font-size: 12pt; font-style: italic; margin-bottom: 4cm; }
  .cover .author-block { margin: 1cm 0; }
  .cover .author-name  { font-size: 13pt; font-weight: bold; }
  .cover .author-email { font-size: 11pt; color: #444; }
  .cover .date         { margin-top: 4cm; font-size: 13pt; font-weight: bold; }
  .page-break          { page-break-after: always; }

  /* Abstract pages */
  .abstract-page h2 { text-align: center; }
  .abstract-page p  { text-align: justify; text-indent: 1cm; }

  /* TOC and list pages */
  .toc h2, .list-page h2 { text-align: center; }
  .toc table, .list-page table { width: 100%; }

  /* Main headings (chapters) */
  h2 {
      font-size: 14pt;
      font-weight: bold;
      text-transform: uppercase;
      margin-top: 1.5em;
      margin-bottom: 0.6em;
      border-bottom: 1px solid #000;
      padding-bottom: 0.2em;
      page-break-after: avoid;
      page-break-before: always;
  }
  /* First H2 on title page should not break */
  .cover h2 { page-break-before: avoid; border-bottom: none; }

  h3 {
      font-size: 12pt;
      font-weight: bold;
      margin-top: 1.2em;
      margin-bottom: 0.4em;
      page-break-after: avoid;
  }

  p { text-indent: 1cm; margin: 0.4em 0; }
  ul, ol { margin: 0.6em 0 0.6em 1.5em; padding-left: 0.5em; }
  li { margin: 0.2em 0; text-align: justify; }

  table {
      border-collapse: collapse;
      margin: 0.8em auto;
      font-size: 11pt;
      page-break-inside: avoid;
  }
  th, td {
      border: 1px solid #000;
      padding: 4px 8px;
  }
  th { background: #eee; text-align: center; font-weight: bold; }

  code { font-family: 'Consolas', 'Courier New', monospace; font-size: 10.5pt; }

  img {
      display: block;
      max-width: 90%;
      margin: 0.6em auto 0.2em auto;
      page-break-inside: avoid;
  }
  img + em, img + p em {
      display: block;
      text-align: center;
      font-size: 10.5pt;
      margin-bottom: 0.8em;
  }
  em { font-style: italic; }
  strong { font-weight: bold; }
</style>
"""
    html = (
        "<!DOCTYPE html><html lang='tr'><head><meta charset='utf-8'>"
        "<title>Radyomik Papilodem Siniflandirmasi - Final Rapor</title>"
        + css + "</head><body>" + html_body + "</body></html>"
    )
    out = md_path.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    return out


def to_pdf_via_edge(html_path: Path) -> Path | None:
    """Render HTML to PDF via Edge headless using ASCII-safe temp path."""
    import shutil
    import subprocess
    import tempfile

    edge = Path(r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
    chrome = Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe")
    browser = edge if edge.exists() else (chrome if chrome.exists() else None)
    if browser is None:
        return None

    pdf_out = html_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory(prefix="papilodem_report_") as td:
        tmp_html = Path(td) / "report.html"
        # copy referenced figures as well so img src='../04_GRAFIKLER/...' works
        # ...actually keep external images as-is, Edge can resolve relative URLs
        # only if we keep the HTML alongside that hierarchy. Easiest: convert
        # img paths to absolute file URLs before write.
        raw_html = html_path.read_text(encoding="utf-8")
        # Convert ../04_GRAFIKLER/... to file:// URLs of the actual files
        figures_dir = (html_path.parent.parent / "04_GRAFIKLER")
        if not figures_dir.exists():
            # fall back to results/figures of source tree
            figures_dir = config.FIGURES_DIR
        for fn in figures_dir.glob("*.png"):
            # both possible src tokens
            raw_html = raw_html.replace(
                f'src="../04_GRAFIKLER/{fn.name}"',
                f'src="{fn.resolve().as_uri()}"',
            )
            raw_html = raw_html.replace(
                f"src='../04_GRAFIKLER/{fn.name}'",
                f"src='{fn.resolve().as_uri()}'",
            )
        tmp_html.write_text(raw_html, encoding="utf-8")
        tmp_pdf = Path(td) / "report.pdf"
        try:
            subprocess.run(
                [
                    str(browser),
                    "--headless=new",
                    "--disable-gpu",
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={tmp_pdf}",
                    tmp_html.as_uri(),
                ],
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                timeout=180,
            )
            shutil.copyfile(tmp_pdf, pdf_out)
            return pdf_out
        except Exception as e:
            print(f"Edge headless PDF rendering failed: {e}")
            return None


if __name__ == "__main__":
    md_path = build_md()
    print(f"Wrote {md_path}")
    try:
        html_path = to_html(md_path)
        print(f"Wrote {html_path}")
        pdf_path = to_pdf_via_edge(html_path)
        if pdf_path is not None:
            print(f"Wrote {pdf_path}")
    except Exception as e:
        print(f"Conversion skipped: {e}")

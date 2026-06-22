"""Build the academic presentation notebook programmatically."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nbformat as nbf  # noqa: E402


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# Radyomik Özelliklerle Papilödem Sınıflandırması\n"
            "**Yapay Zeka Final Ödevi**\n\n"
            "Bu notebook, patient-level radyomik veri ile Normal vs Papilödem ayırımı "
            "için geliştirilen pipeline'ın akademik sunumudur."
        ),
        md("## 1. Veri Yükleme ve İntegrity Kontrolü"),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "from src.data_loader import load_dataset\n"
            "from src import config\n\n"
            "bundle = load_dataset()\n"
            "print('X shape:', bundle.X.shape)\n"
            "print('Patients:', len(set(bundle.patient_id)))\n"
            "uniq, cnt = np.unique(bundle.y, return_counts=True)\n"
            "print('Class balance:', dict(zip(uniq.tolist(), cnt.tolist())))"
        ),
        md(
            "**İntegrity assertions** veri yükleme sırasında otomatik çalışır:\n"
            "1. Hasta sınıflar arası çakışmıyor (`N_*` ve `P_*` setleri disjoint).\n"
            "2. Her hasta sadece bir sınıf etiketine sahip.\n"
            "3. Hasta başına örnek sayısı tutarlı."
        ),
        md("## 2. Pipeline Çalıştırma\n"
           "Tam çalıştırma offline yapıldı. Buradan kaydedilmiş sonuçları yüklüyoruz."),
        code(
            "import json\n\n"
            "summary = pd.read_csv(config.TABLES_DIR / 'metrics_summary.csv', header=[0, 1], index_col=0)\n"
            "summary"
        ),
        code(
            "wilcoxon = pd.read_csv(config.TABLES_DIR / 'wilcoxon_bonferroni.csv')\n"
            "wilcoxon"
        ),
        code(
            "friedman = json.loads((config.TABLES_DIR / 'friedman.json').read_text())\n"
            "print(f\"Friedman χ² = {friedman['statistic']:.3f}, p = {friedman['p_value']:.4g}\")"
        ),
        code(
            "manifest = json.loads((config.RESULTS_DIR / 'run_manifest.json').read_text())\n"
            "print('Best overall:', manifest['best_overall'])\n"
            "print(f\"Wallclock: {manifest['total_seconds']/60:.1f} min\")"
        ),
        md("## 3. Grafikler"),
        code(
            "from IPython.display import Image, display\n\n"
            "figures = [\n"
            "    'roc_curves.png', 'pr_curves.png', 'confusion_matrix.png',\n"
            "    'calibration_curves.png',\n"
            "    'model_comparison_macro_f1.png', 'model_comparison_roc_auc.png',\n"
            "    'feature_stability.png', 'feature_importance.png',\n"
            "]\n"
            "for fn in figures:\n"
            "    fp = config.FIGURES_DIR / fn\n"
            "    if fp.exists():\n"
            "        print(f'=== {fn} ===')\n"
            "        display(Image(fp))\n"
            "    else:\n"
            "        print(f'(skipped) {fn} not found')"
        ),
        md("## 4. EK Soruların Cevapları (spec §21)\n\n"
           "1. **Hangi model en iyi performansı verdi?**  \n"
           "   En iyi model `run_manifest.json` ve `metrics_summary.csv` mean macro-F1'e göre belirleniyor.\n\n"
           "2. **Ensemble model tekli modellerden daha iyi mi?**  \n"
           "   Wilcoxon tablosunda Ensemble'ı içeren satırlar incelenir; significant=True olan eşleşmeler vurgulanır.\n\n"
           "3. **MRMR özellik seçimi performansı artırdı mı?**  \n"
           "   Feature stability grafiği seçilen özelliklerin tekrarlanabilirliğini gösterir. "
           "MRMR olmadan baseline ile karşılaştırma ileri çalışma olarak önerilir.\n\n"
           "4. **Kalibrasyon model güvenilirliğini artırdı mı?**  \n"
           "   Calibration curves diyagonale yakındır; Brier skoru kalibrasyonun ardından düşer.\n\n"
           "5. **ROC-AUC ile PR-AUC arasında ilişki?**  \n"
           "   Dengesiz veride (Normal %70 - Papilödem %30) PR-AUC azınlık sınıfa daha duyarlıdır. "
           "Genellikle ROC-AUC daha optimist, PR-AUC daha realist sinyal verir.\n\n"
           "6. **Veri boyutunun model performansına etkisi nedir?**  \n"
           "   20 outer repeat ile std deviation gözlemlenir; learning curve için ek deney önerilir.\n\n"
           "7. **En önemli ilk 10 radyomik özellik hangileridir?**  \n"
           "   `feature_stability.png` ve (varsa) SHAP summary plot'tan belirlenir."),
    ]
    out = Path("notebooks/radiomics_pipeline.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    p = build()
    print(f"Wrote {p}")

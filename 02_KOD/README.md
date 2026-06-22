# Radyomik Özelliklerle Papilödem Sınıflandırması

Makine Öğrenmesi Final Ödevi, patient-level radiomik ikili sınıflandırma (Normal vs Papilödem).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Smoke Test

```bash
python scripts/smoke_test.py
```

## Full Pipeline Run

```bash
python scripts/run_pipeline.py
```

## Notebook

```bash
jupyter notebook notebooks/radiomics_pipeline.ipynb
```

Detaylı tasarım: [`docs/superpowers/specs/2026-06-03-radiomics-papilodem-classification-design.md`](docs/superpowers/specs/2026-06-03-radiomics-papilodem-classification-design.md)
Implementasyon planı: [`docs/superpowers/plans/2026-06-03-radiomics-papilodem-pipeline.md`](docs/superpowers/plans/2026-06-03-radiomics-papilodem-pipeline.md)

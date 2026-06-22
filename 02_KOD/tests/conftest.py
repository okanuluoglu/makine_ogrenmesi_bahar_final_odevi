"""Shared pytest fixtures: synthetic mini-dataset and real-data sample."""
import numpy as np
import pandas as pd
import pytest

from src import config

RNG = np.random.default_rng(0)


@pytest.fixture
def synthetic_dataset():
    """Tiny 60-sample synthetic dataset mimicking radiomic data structure.

    10 patients × 6 samples each (3 per side × 2 sides), 20 features.
    Half normal (5 patients), half papilodem (5 patients).
    """
    n_patients_per_class = 5
    samples_per_side = 3
    n_features = 20
    rows = []
    for cls, prefix in [(0, "N"), (1, "P")]:
        for p in range(1, n_patients_per_class + 1):
            pid = f"{prefix}_{p}"
            for side in ["Right", "Left"]:
                for _ in range(samples_per_side):
                    feats = RNG.normal(loc=cls * 1.5, scale=1.0, size=n_features)
                    row = {"patient_id": pid, "SideStandard": side, "label": cls}
                    row.update({f"Feature_{i+1:04d}": v for i, v in enumerate(feats)})
                    rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_xy(synthetic_dataset):
    """Returns (X, y, groups, side_array) from synthetic_dataset."""
    df = synthetic_dataset
    feat_cols = [c for c in df.columns if c.startswith("Feature_")]
    X = df[feat_cols].values
    y = df["label"].values
    groups = df["patient_id"].values
    side = df["SideStandard"].values
    return X, y, groups, side


@pytest.fixture(scope="session")
def real_data_paths():
    """Paths to real CSV data. Skips test if not present."""
    if not config.NORMAL_CSV.exists() or not config.PAPILODEM_CSV.exists():
        pytest.skip("Real data not present in data/raw/")
    return config.NORMAL_CSV, config.PAPILODEM_CSV

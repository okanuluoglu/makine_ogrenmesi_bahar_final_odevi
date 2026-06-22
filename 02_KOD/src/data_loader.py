"""Load radiomic CSVs, merge with global patient IDs, encode SideStandard."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DataBundle:
    """Container for fully-loaded dataset."""
    X: np.ndarray                # (n_samples, n_features)
    y: np.ndarray                # (n_samples,) binary labels
    patient_id: np.ndarray       # (n_samples,) string IDs ("N_1", "P_3", ...)
    side: np.ndarray             # (n_samples,) original "Right"/"Left"
    feature_names: list[str]     # column names of X
    df: pd.DataFrame             # full merged dataframe for debugging/EDA


def _prefix_patient_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add a `patient_id` column of form '<prefix>_<PatientIndex>'."""
    out = df.copy()
    out["patient_id"] = prefix + "_" + out["PatientIndex"].astype(str)
    return out


def _build_combined(normal_csv: Path, papil_csv: Path) -> pd.DataFrame:
    """Read both CSVs, prefix patient IDs, add labels, concatenate."""
    normal = pd.read_csv(normal_csv)
    papil = pd.read_csv(papil_csv)
    normal = _prefix_patient_ids(normal, "N")
    papil = _prefix_patient_ids(papil, "P")
    normal["label"] = 0
    papil["label"] = 1
    combined = pd.concat([normal, papil], ignore_index=True)
    return combined


def _assert_integrity(df: pd.DataFrame) -> None:
    """Three pre-CV integrity checks. Raises AssertionError on violation."""
    normal_pids = set(df.loc[df["label"] == 0, "patient_id"])
    papil_pids = set(df.loc[df["label"] == 1, "patient_id"])
    overlap = normal_pids & papil_pids
    assert not overlap, f"Patient ID overlap across classes: {overlap}"

    labels_per_pid = df.groupby("patient_id")["label"].nunique()
    bad = labels_per_pid[labels_per_pid > 1]
    assert bad.empty, f"Patients with mixed labels: {bad.to_dict()}"

    sizes = df.groupby("patient_id").size()
    common = sizes.mode().iloc[0]
    deviants = sizes[sizes != common]
    if len(deviants) > 0:
        warnings.warn(
            f"{len(deviants)} patients deviate from typical sample count {common}: "
            f"{deviants.to_dict()}",
            stacklevel=2,
        )


def load_dataset(
    normal_csv: Path | None = None,
    papil_csv: Path | None = None,
) -> DataBundle:
    """Load both CSVs, one-hot-encode SideStandard, run integrity asserts."""
    from src import config
    normal_csv = normal_csv or config.NORMAL_CSV
    papil_csv = papil_csv or config.PAPILODEM_CSV

    combined = _build_combined(normal_csv, papil_csv)
    _assert_integrity(combined)

    side_dummies = pd.get_dummies(combined["SideStandard"], prefix="side").astype(int)
    feature_cols = [c for c in combined.columns if c.startswith("Feature_")]
    X_df = pd.concat([combined[feature_cols], side_dummies], axis=1)
    X = X_df.to_numpy(dtype=np.float64)

    # Convert any infinity values to NaN so the imputer can replace them with the median.
    # Real radiomic CSVs occasionally contain inf from undefined statistics
    # (e.g. division by zero in a ratio feature).
    inf_count = int(np.isinf(X).sum())
    if inf_count > 0:
        warnings.warn(
            f"Replacing {inf_count} infinity values with NaN for median imputation",
            stacklevel=2,
        )
        X = np.where(np.isinf(X), np.nan, X)

    return DataBundle(
        X=X,
        y=combined["label"].to_numpy(dtype=np.int64),
        patient_id=combined["patient_id"].to_numpy(),
        side=combined["SideStandard"].to_numpy(),
        feature_names=list(X_df.columns),
        df=combined,
    )

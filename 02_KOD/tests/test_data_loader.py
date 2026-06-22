import pandas as pd

from src.data_loader import _build_combined, _prefix_patient_ids, load_dataset


def test_prefix_patient_ids():
    df = pd.DataFrame({"PatientIndex": [1, 2, 1]})
    out = _prefix_patient_ids(df, "N")
    assert list(out["patient_id"]) == ["N_1", "N_2", "N_1"]


def test_build_combined_disjoint(tmp_path):
    normal = pd.DataFrame({
        "PatientIndex": [1, 1, 2],
        "SideStandard": ["Right", "Left", "Right"],
        "Feature_0001": [1.0, 2.0, 3.0],
        "Feature_0002": [4.0, 5.0, 6.0],
    })
    papil = pd.DataFrame({
        "PatientIndex": [1, 2, 2],
        "SideStandard": ["Right", "Right", "Left"],
        "Feature_0001": [7.0, 8.0, 9.0],
        "Feature_0002": [10.0, 11.0, 12.0],
    })
    n_path = tmp_path / "normal.csv"
    p_path = tmp_path / "papil.csv"
    normal.to_csv(n_path, index=False)
    papil.to_csv(p_path, index=False)

    combined = _build_combined(n_path, p_path)

    assert set(combined["patient_id"]) == {"N_1", "N_2", "P_1", "P_2"}
    assert combined.loc[combined["label"] == 0, "patient_id"].str.startswith("N_").all()
    assert combined.loc[combined["label"] == 1, "patient_id"].str.startswith("P_").all()
    assert "Feature_0001" in combined.columns
    assert "label" in combined.columns


def test_load_real_dataset(real_data_paths):
    n_path, p_path = real_data_paths
    bundle = load_dataset(n_path, p_path)
    assert bundle.X.shape[0] == 966
    assert bundle.X.shape[1] >= 746
    assert bundle.y.sum() == 294
    assert len(set(bundle.patient_id)) == 69
    assert (bundle.y == 0).sum() == 672

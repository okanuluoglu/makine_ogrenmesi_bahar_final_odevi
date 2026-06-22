from pathlib import Path

from src import config


def test_seeds_consistent():
    assert config.SEED == 42


def test_outer_repeats():
    assert config.N_OUTER_REPEATS == 20
    assert config.N_INNER_FOLDS == 5
    assert config.N_TRIALS == 50


def test_thresholds():
    assert config.VARIANCE_THRESHOLD == 0.01
    assert config.CORRELATION_THRESHOLD == 0.95
    assert config.TEST_SIZE == 0.30


def test_mrmr_k_candidates():
    assert config.MRMR_K_CANDIDATES == [20, 50, 100]


def test_paths_exist():
    assert isinstance(config.ROOT_DIR, Path)
    assert config.DATA_DIR.name == "data"
    assert config.RESULTS_DIR.name == "results"
    assert config.BASE_MODEL_NAMES == ["LR", "SVM", "RF", "ET", "GB", "KNN"]

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.ensemble import build_soft_voting_ensemble


def test_ensemble_predicts(synthetic_xy):
    X, y, _, _ = synthetic_xy
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    lr = LogisticRegression(max_iter=500).fit(X, y)
    ens = build_soft_voting_ensemble({"rf": rf, "lr": lr})
    ens.fit(X, y)
    proba = ens.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)

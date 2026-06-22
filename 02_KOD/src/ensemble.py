"""Soft-voting ensemble builder."""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier


def build_soft_voting_ensemble(
    members: dict[str, BaseEstimator],
) -> VotingClassifier:
    """Compose a VotingClassifier(voting='soft') from named members.

    Members should already be calibrated for best ensemble behavior.
    """
    return VotingClassifier(
        estimators=list(members.items()),
        voting="soft",
        n_jobs=1,
    )

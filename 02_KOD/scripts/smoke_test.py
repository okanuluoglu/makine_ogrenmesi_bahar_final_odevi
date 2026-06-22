"""Smoke test: 2 outer repeats × 6 models × 3 trials. ~5 min wallclock."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_full_pipeline  # noqa: E402


def main() -> int:
    print("Running smoke test: 2 outer repeats, 3 Optuna trials per model.")
    result = run_full_pipeline(
        n_outer_repeats=2, n_trials=3, n_inner_folds=3,
        save_artifacts=False,
    )
    summary = result["summary"]
    print("\n=== Smoke Test Summary (macro_f1) ===")
    print(summary["macro_f1"])
    print(
        f"\nBest model: {result['best_overall']['model']} "
        f"(macro_f1={result['best_overall']['macro_f1']:.4f})"
    )
    print(f"Wallclock: {result['total_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

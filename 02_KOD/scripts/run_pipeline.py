"""Full pipeline run: 20 outer repeats × 6 models × 50 Optuna trials each."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.pipeline import run_full_pipeline  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=config.N_OUTER_REPEATS)
    p.add_argument("--trials", type=int, default=config.N_TRIALS)
    p.add_argument("--inner-folds", type=int, default=config.N_INNER_FOLDS)
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from results/checkpoint.joblib if present",
    )
    p.add_argument(
        "--fresh", action="store_true",
        help="Delete any existing checkpoint before starting (forces fresh run)",
    )
    args = p.parse_args()

    if args.fresh:
        ckpt = config.RESULTS_DIR / "checkpoint.joblib"
        if ckpt.exists():
            ckpt.unlink()
            print(f"Deleted existing checkpoint: {ckpt}")

    print(
        f"Full run: {args.repeats} outer repeats x {args.trials} trials per model "
        f"x {args.inner_folds} inner folds (resume={args.resume})"
    )
    result = run_full_pipeline(
        n_outer_repeats=args.repeats,
        n_trials=args.trials,
        n_inner_folds=args.inner_folds,
        save_artifacts=True,
        resume=args.resume,
    )
    print("\n=== Final Summary (macro_f1) ===")
    print(result["summary"]["macro_f1"])
    print(
        f"\nBest: {result['best_overall']['model']} "
        f"(macro_f1={result['best_overall']['macro_f1']:.4f})"
    )
    print(f"Friedman p-value: {result['friedman'][1]:.4g}")
    print(f"Wallclock: {result['total_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

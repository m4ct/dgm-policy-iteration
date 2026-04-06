"""
run_all.py — Run all exercises sequentially.

Usage:
    python run_all.py          # run everything
    python run_all.py 1 3 4    # run only exercises 1.1, 1.2, 3, 4
"""

import sys
import os
import time
import subprocess

EXERCISES = [
    ("1.1", "exercise_1_1.py"),
    ("1.2", "exercise_1_2.py"),
    ("2.1", "exercise_2_1.py"),
    ("2.2", "exercise_2_2.py"),
    ("3",   "exercise_3.py"),
    ("4",   "exercise_4.py"),
]


def main():
    # Parse optional filter from CLI args
    if len(sys.argv) > 1:
        requested = set(sys.argv[1:])
        exercises = [(label, script) for label, script in EXERCISES if label in requested]
        if not exercises:
            print(f"No matching exercises. Available: {[l for l, _ in EXERCISES]}")
            sys.exit(1)
    else:
        exercises = EXERCISES

    script_dir = os.path.dirname(os.path.abspath(__file__))
    total_start = time.time()
    results = []

    print("=" * 64)
    print("  Stochastic Control — Running all exercises")
    print("=" * 64)

    for label, script in exercises:
        print(f"\n{'━' * 64}")
        print(f"  Exercise {label}  ({script})")
        print(f"{'━' * 64}\n")

        start = time.time()
        ret = subprocess.run(
            [sys.executable, os.path.join(script_dir, script)],
            cwd=script_dir,
        )
        elapsed = time.time() - start

        if ret.returncode == 0:
            results.append((label, "OK", elapsed))
            print(f"\n  ✓ Exercise {label} completed in {elapsed:.1f}s")
        else:
            results.append((label, "FAILED", elapsed))
            print(f"\n  ✗ Exercise {label} failed after {elapsed:.1f}s "
                  f"(exit code {ret.returncode})")

    # Summary
    total = time.time() - total_start
    print(f"\n{'=' * 64}")
    print("  Summary")
    print(f"{'=' * 64}")
    for label, status, elapsed in results:
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} Exercise {label:>4s}  {status:<6s}  {elapsed:>7.1f}s")
    print(f"{'─' * 64}")
    print(f"  Total: {total:.1f}s")
    print(f"{'=' * 64}")

    if any(s == "FAILED" for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

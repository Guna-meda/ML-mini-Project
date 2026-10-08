"""Run Part 1 (reference reproduction) and then Part 2 (Gradient Boosting)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Part 1: Reference reproduction", "project.src.reproduction"),
    ("Part 2: Gradient Boosting extension", "project.src.part2"),
]

for label, module in STEPS:
    print(f"\n=== {label} ===")
    result = subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

print("\nAll project parts finished successfully.")

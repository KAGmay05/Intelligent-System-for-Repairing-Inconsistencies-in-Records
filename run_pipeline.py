"""
run_pipeline.py
---------------
One-command runner for the full inconsistency detection and repair pipeline.

Steps:
  1. data/seed.py              — generate synthetic dataset
  2. detector/structural_rules.py — detect structural inconsistencies
  3. detector/llm_detector.py  — detect textual inconsistencies  (skippable)
  4. detector/repair_optimizer.py — solve minimum-cost repair with CP-SAT
  5. validate.py               — confirm 0 structural issues remain
  6. evaluate.py               — print precision / recall / F1

Usage:
    python run_pipeline.py
    python run_pipeline.py --seed 42
    python run_pipeline.py --skip-textual   # reuse existing textual_inconsistencies.json
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def run_step(label: str, cmd: list) -> bool:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {label}")
    print(f"{'═' * width}")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\n[FAIL] El paso '{label}' terminó con código {result.returncode}. Pipeline abortado.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo de detección y reparación")
    parser.add_argument("--seed",         type=int, default=42,  help="Semilla para el generador del dataset")
    parser.add_argument("--skip-textual", action="store_true",
                        help="Omitir detección textual (reutiliza textual_inconsistencies.json existente)")
    args = parser.parse_args()

    py = sys.executable

    steps = [
        ("1/5  Generar dataset",                      [py, "data/seed.py", "--seed", str(args.seed)]),
        ("2/5  Detectar inconsistencias estructurales", [py, "detector/structural_rules.py"]),
        ("4/5  Optimizar reparaciones (CP-SAT)",       [py, "detector/repair_optimizer.py"]),
        ("5/5  Validar dataset reparado",              [py, "validate.py"]),
        ("     Evaluar precisión / recall / F1",       [py, "evaluate.py"]),
    ]

    if not args.skip_textual:
        steps.insert(2, (
            "3/5  Detectar inconsistencias textuales",
            [py, "detector/llm_detector.py"],
        ))

    for label, cmd in steps:
        if not run_step(label, cmd):
            sys.exit(1)

    print(f"\n{'═' * 60}")
    print("  ✔  Pipeline completado con éxito.")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()

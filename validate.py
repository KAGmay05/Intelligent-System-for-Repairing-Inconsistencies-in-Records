"""
validate.py
-----------
Verifies that repaired_dataset.json contains zero structural inconsistencies.
Run after repair_optimizer.py to confirm the repair plan was fully effective.

Exit codes:
  0 — no structural inconsistencies remain
  1 — inconsistencies still present (or file not found)

Usage:
    python validate.py
    python validate.py --repaired data/repaired_dataset.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Import detector from the detector sub-package
sys.path.insert(0, str(BASE_DIR / "detector"))
from structural_rules import detect_all_structural  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Verificar que el dataset reparado no tiene inconsistencias estructurales"
    )
    parser.add_argument(
        "--repaired",
        default=str(DATA_DIR / "repaired_dataset.json"),
        help="Ruta al dataset reparado (default: data/repaired_dataset.json)",
    )
    args = parser.parse_args()

    path = Path(args.repaired)
    if not path.exists():
        print(f"ERROR: {path} no encontrado. Ejecuta repair_optimizer.py primero.")
        sys.exit(1)

    with path.open("r", encoding="utf-8") as fh:
        repaired = json.load(fh)

    issues = detect_all_structural(repaired)

    if not issues:
        print("✔  Validación exitosa: 0 inconsistencias estructurales en el dataset reparado.")
        sys.exit(0)

    by_type = Counter(i.get("type") for i in issues)
    print(f"✘  Validación fallida: {len(issues)} inconsistencias estructurales aún presentes.")
    for issue_type, count in sorted(by_type.items()):
        print(f"   - {issue_type}: {count}")
    sys.exit(1)


if __name__ == "__main__":
    main()

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
  6. evaluate.py               — print precision / recall / F1 / MCC

Usage:
    python run_pipeline.py
    python run_pipeline.py --seed 42
    python run_pipeline.py --skip-textual   # reuse existing data/textual_inconsistencies.json
    python run_pipeline.py --verbose        # show solver decision details
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def run_step(label: str, cmd: list, verbose: bool = False) -> tuple[bool, float]:
    """Run one pipeline step. Returns (success, elapsed_seconds)."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {label}")
    print(f"{'═' * width}")
    env_extra = {"VERBOSE": "1"} if verbose else {}
    import os
    env = {**os.environ, **env_extra}
    t0     = time.monotonic()
    result = subprocess.run(cmd, cwd=BASE_DIR, env=env)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        print(f"\n[FAIL] El paso '{label}' terminó con código {result.returncode}. Pipeline abortado.")
        return False, elapsed
    return True, elapsed


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo de detección y reparación")
    parser.add_argument("--seed",         type=int, default=42,  help="Semilla para el generador del dataset")
    parser.add_argument("--skip-textual", action="store_true",
                        help="Omitir detección textual (reutiliza data/textual_inconsistencies.json existente)")
    parser.add_argument("--verbose",      action="store_true",
                        help="Mostrar decisiones detalladas del solver y alternativas rechazadas")
    parser.add_argument("--llm-timeout",  type=int, default=None,
                        help="Timeout por request textual LLM en segundos (p.ej. 60)")
    parser.add_argument("--llm-retries",  type=int, default=None,
                        help="Reintentos por request textual LLM (p.ej. 4)")
    parser.add_argument("--llm-subset",   type=int, default=20,
                        help="Reportes coherentes a muestrear para la detección textual además de "
                             "los que tienen ground truth (default 20). El LLM local es lento "
                             "(~1 min/reporte en CPU), así que correr los 156 tardaría horas. "
                             "Usa --llm-subset 0 para procesar TODOS los reportes.")
    args = parser.parse_args()

    py = sys.executable

    steps = [
        ("1/5  Generar dataset",                      [py, "data/seed.py", "--seed", str(args.seed)]),
        ("2/5  Detectar inconsistencias estructurales", [py, "detector/structural_rules.py"]),
        ("4/5  Optimizar reparaciones (CP-SAT)",       [py, "detector/repair_optimizer.py"]),
        ("5/5  Validar dataset reparado",              [py, "validate.py"]),
        ("     Evaluar precisión / recall / F1 / MCC", [py, "evaluate.py"]),
    ]

    if not args.skip_textual:
        textual_cmd = [py, "detector/llm_detector.py"]
        if args.llm_subset and args.llm_subset > 0:
            textual_cmd += ["--eval-subset", str(args.llm_subset)]
        if args.llm_timeout is not None:
            textual_cmd += ["--timeout", str(args.llm_timeout)]
        if args.llm_retries is not None:
            textual_cmd += ["--retries", str(args.llm_retries)]
        steps.insert(2, (
            "3/5  Detectar inconsistencias textuales",
            textual_cmd,
        ))

    detector_mode = os.environ.get("DETECTOR_MODE", "llm")
    log_lines = [f"Pipeline ejecutado: {datetime.now().isoformat()}",
                 f"  seed={args.seed}  skip_textual={args.skip_textual}  verbose={args.verbose}  detector_mode={detector_mode}  llm_timeout={args.llm_timeout}  llm_retries={args.llm_retries}"]

    for label, cmd in steps:
        if args.verbose:
            # Pass the verbose flag explicitly to the optimizer step
            if "repair_optimizer" in cmd[-1]:
                cmd = cmd + ["--verbose"]
        ok, elapsed = run_step(label, cmd, verbose=args.verbose)
        status = "OK" if ok else "FAIL"
        log_lines.append(f"  [{status}] {label}  ({elapsed:.1f}s)")
        if not ok:
            _write_log(log_lines)
            sys.exit(1)

    print(f"\n{'═' * 60}")
    print("  ✔  Pipeline completado con éxito.")
    print(f"{'═' * 60}\n")
    log_lines.append("  Resultado: ÉXITO")
    _write_log(log_lines)


def _write_log(lines: list) -> None:
    log_path = DATA_DIR / "pipeline_run.log"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n" + "─" * 60 + "\n")
    print(f"Log guardado en: {log_path}")


if __name__ == "__main__":
    main()

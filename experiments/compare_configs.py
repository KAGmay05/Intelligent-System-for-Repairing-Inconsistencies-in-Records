"""
experiments/compare_configs.py
--------------------------------
Compares three detector configurations on the SAME dataset:

    rule   — only structural rules (no LLM)
    hybrid — structural rules + LLM (LLM enriches but rules still run)
    llm    — full LLM-based textual detection (default production mode)

For each mode it:
  1. Runs the textual detector  (unless mode=rule, which skips it)
  2. Runs the CP-SAT repair optimizer
  3. Runs evaluation
  4. Captures precision / recall / F1 / MCC and total repair cost

Results are saved to:
    data/config_comparison.json

and a comparison table is printed to stdout.

Usage:
    python experiments/compare_configs.py
    python experiments/compare_configs.py --seed 42        # regenerate dataset first
    python experiments/compare_configs.py --no-regen       # use existing data/dataset.json
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PY       = sys.executable

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CONFIGS    = ("rule", "hybrid", "llm")
OPTIMIZERS = ("cpsat", "greedy", "sa")


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _run(cmd: list, env: dict | None = None) -> bool:
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, cwd=ROOT_DIR, env=merged_env)
    return result.returncode == 0


def _load(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=4, ensure_ascii=False)


def _ollama_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3):
            return True
    except (urllib.error.URLError, Exception):
        return False


# ─────────────────────────────────────────────────────────────────
# PER-CONFIG RUNNER
# ─────────────────────────────────────────────────────────────────

def run_config(mode: str, optimizer: str, dataset_path: Path, out_dir: Path) -> dict | None:
    """
    Run textual detection (if applicable), repair optimizer, and evaluation
    for one DETECTOR_MODE.  Returns metrics dict or None on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    incon_path   = DATA_DIR / "inconsistencies.json"   # structural is shared
    textual_path = out_dir / "textual_inconsistencies.json"
    plan_path    = out_dir / "repair_plan.json"
    repaired     = out_dir / "repaired_dataset.json"
    eval_path    = out_dir / "evaluation_report.json"

    env = {"DETECTOR_MODE": mode}

    # Step 1: textual detection (skip for rule-only mode)
    if mode == "rule":
        # create empty textual file so downstream steps don't fail
        _save(textual_path, [])
    else:
        print(f"    Detección textual ({mode})...", end=" ", flush=True)
        ok = _run([PY, "detector/llm_detector.py",
                   "--dataset", str(dataset_path),
                   "--output",  str(textual_path)], env=env)
        if not ok:
            print("FAIL")
            return None
        print("OK")

    # Step 2: repair optimizer
    print(f"    Repair optimizer ({optimizer})...", end=" ", flush=True)
    ok = _run([PY, "detector/repair_optimizer.py",
               "--dataset",         str(dataset_path),
               "--textual",         str(textual_path),
               "--optimizer",       optimizer,
               "--output-plan",     str(plan_path),
               "--output-repaired", str(repaired)], env=env)
    if not ok:
        print("FAIL")
        return None
    print("OK")

    # Step 3: evaluate
    print("    Evaluación...", end=" ", flush=True)
    ok = _run([PY, "evaluate.py",
               "--dataset",    str(dataset_path),
               "--structural", str(incon_path),
               "--textual",    str(textual_path),
               "--output",     str(eval_path)], env=env)
    if not ok:
        print("FAIL")
        return None
    print("OK")

    if not eval_path.exists():
        return None

    report = _load(eval_path)
    g = report.get("global", {})

    # total repair cost from repair plan
    total_cost = 0
    if plan_path.exists():
        plan = _load(plan_path)
        total_cost = (plan.get("summary") or {}).get("total_cost",
                      plan.get("total_cost", 0))

    return {
        "mode":          mode,
        "optimizer":     optimizer,
        "config_id":     f"{mode}+{optimizer}",
        "precision":     g.get("precision", 0.0),
        "recall":        g.get("recall",    0.0),
        "f1":            g.get("f1",        0.0),
        "mcc":           g.get("mcc",       0.0),
        "tp":            g.get("tp",        0),
        "fp":            g.get("fp",        0),
        "fn":            g.get("fn",        0),
        "total_cost":    total_cost,
        "by_type":       report.get("by_type", {}),
        "behavior":      report.get("behavior_analysis", {}),
    }


# ─────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    print(f"\n{'═'*76}")
    print("  COMPARACIÓN DE CONFIGURACIONES")
    print(f"{'═'*76}")
    hdr = f"  {'Modo':<8} {'Opt':<7} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>6}  {'Coste':>7}"
    print(hdr)
    print(f"  {'─'*72}")
    for r in results:
        print(
            f"  {r['mode']:<8} {r['optimizer']:<7} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}"
            f"  {r['precision']:>6.4f} {r['recall']:>6.4f} {r['f1']:>6.4f}"
            f" {r['mcc']:>6.3f}  {r['total_cost']:>7}"
        )
    print(f"  {'─'*72}")

    if len(results) > 1:
        best_f1   = max(results, key=lambda x: x["f1"])
        best_mcc  = max(results, key=lambda x: x["mcc"])
        best_cost = min(results, key=lambda x: x["total_cost"])
        print(f"\n  Mejor F1     : {best_f1['config_id']}  (F1={best_f1['f1']:.4f})")
        print(f"  Mejor MCC    : {best_mcc['config_id']}  (MCC={best_mcc['mcc']:.3f})")
        print(f"  Menor coste  : {best_cost['config_id']}  (coste={best_cost['total_cost']})")
    print(f"{'═'*76}\n")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compara modos rule / hybrid / llm sobre el mismo dataset"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla para (re)generar dataset antes de comparar")
    parser.add_argument("--no-regen", action="store_true",
                        help="Usar dataset existente en data/ sin regenerar")
    parser.add_argument("--modes", nargs="+", default=list(CONFIGS),
                        choices=list(CONFIGS),
                        help="Modos a comparar (default: rule hybrid llm)")
    parser.add_argument("--optimizers", nargs="+", default=list(OPTIMIZERS),
                        choices=list(OPTIMIZERS),
                        help="Optimizadores a comparar (default: cpsat greedy sa)")
    args = parser.parse_args()

    compare_dir  = DATA_DIR / "compare"
    dataset_path = DATA_DIR / "dataset.json"
    structural_path = DATA_DIR / "inconsistencies.json"

    # ── (Re)generate dataset ──────────────────────────────────────────────
    if not args.no_regen:
        seed = args.seed or 42
        print(f"Generando dataset con semilla {seed}...")
        if not _run([PY, "data/seed.py", "--seed", str(seed)]):
            print("ERROR: no se pudo generar el dataset.")
            sys.exit(1)

        # Structural detection (shared across all modes)
        print("Detectando inconsistencias estructurales...")
        if not _run([PY, "detector/structural_rules.py",
                     "--dataset", str(dataset_path),
                     "--output", str(structural_path)]):
            print("ERROR: fallo en structural_rules.")
            sys.exit(1)
    else:
        if not dataset_path.exists():
            print(f"ERROR: {dataset_path} no existe. Ejecuta sin --no-regen primero.")
            sys.exit(1)
        print(f"Usando dataset existente: {dataset_path}")
        if not structural_path.exists():
            print(f"[aviso] {structural_path} no existe. Regenerando inconsistencias estructurales...")
            if not _run([PY, "detector/structural_rules.py",
                         "--dataset", str(dataset_path),
                         "--output", str(structural_path)]):
                print("ERROR: no se pudo regenerar inconsistencies.json en modo --no-regen.")
                sys.exit(1)

    # ── Check Ollama ──────────────────────────────────────────────────────
    ollama_ok = _ollama_available()
    if not ollama_ok:
        print(f"\n[aviso] Ollama no disponible en {OLLAMA_URL}.")
        print("  Los modos 'hybrid' y 'llm' serán omitidos.")
        print("  Para habilitarlos, ejecuta: ollama serve\n")

    # ── Run each config ───────────────────────────────────────────────────
    all_results = []
    for mode in args.modes:
        if mode in ("hybrid", "llm") and not ollama_ok:
            print(f"\n── Modo '{mode}' — OMITIDO (Ollama no disponible)")
            continue

        for optimizer in args.optimizers:
            print(f"\n── Config '{mode}+{optimizer}' ─────────────────────────")
            out_dir = compare_dir / f"{mode}_{optimizer}"
            result  = run_config(mode, optimizer, dataset_path, out_dir)
            if result:
                all_results.append(result)
                print(f"  → P={result['precision']:.3f}  R={result['recall']:.3f}  "
                      f"F1={result['f1']:.3f}  MCC={result['mcc']:.3f}  coste={result['total_cost']}")
            else:
                print(f"  → FALLÓ")

    if not all_results:
        print("\nNingún modo completó con éxito.")
        sys.exit(1)

    # ── Save and print ────────────────────────────────────────────────────
    out_path = DATA_DIR / "config_comparison.json"
    _save(out_path, {
        "modes_attempted": args.modes,
        "optimizers_attempted": args.optimizers,
        "modes_ok":        [r["mode"] for r in all_results],
        "configs_ok":      [r["config_id"] for r in all_results],
        "results":         all_results,
    })

    print_table(all_results)
    print(f"Comparación guardada en: {out_path}")


if __name__ == "__main__":
    main()
